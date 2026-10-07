use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use anyhow::Result;
use eframe::egui::{
    self, Color32, ColorImage, RichText, Sense, Stroke, TextureHandle, TextureOptions, pos2, vec2,
};

use crate::analysis::{self, Analysis};
use crate::export::{self, CutMode, ExportItem};
use crate::ffmpeg::{self, VideoInfo};
use crate::scenes::{self, DetectMode, Params, Scene, SceneKind};

const THUMB_W: u32 = 192;
const ROW_HEIGHT: f32 = 112.0;

/// Messages from background threads. `generation` ties results to the video they were
/// started for, so late results from a previously opened file are dropped.
enum Msg {
    AnalysisProgress(f32),
    AnalysisDone { generation: u64, path: PathBuf, result: Result<(VideoInfo, Analysis)> },
    Thumb { generation: u64, frame: usize, size: [usize; 2], rgba: Vec<u8> },
    ExportProgress(usize),
    ExportDone(Result<Vec<PathBuf>>),
}

enum Task {
    Idle,
    Analyzing(f32),
    Exporting { done: usize, total: usize },
}

struct Loaded {
    path: PathBuf,
    info: VideoInfo,
    analysis: Analysis,
    cuts: Vec<usize>,
    scenes: Vec<Scene>,
    thumb_tx: Sender<usize>,
}

pub struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    generation: u64,
    task: Task,
    cancel: Arc<AtomicBool>,
    status: String,

    params: Params,
    loaded: Option<Loaded>,
    /// Cut frames the user merged away.
    merged: HashSet<usize>,
    /// Per-scene user choices, keyed by the scene's start frame.
    kind_overrides: HashMap<usize, SceneKind>,
    excluded: HashSet<usize>,

    thumbs: HashMap<usize, TextureHandle>,
    thumbs_requested: HashSet<usize>,

    out_dir: Option<PathBuf>,
    cut_mode: CutMode,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        Self {
            tx,
            rx,
            generation: 0,
            task: Task::Idle,
            cancel: Arc::new(AtomicBool::new(false)),
            status: "Open a video to begin (or drop one on the window).".into(),
            params: Params::default(),
            loaded: None,
            merged: HashSet::new(),
            kind_overrides: HashMap::new(),
            excluded: HashSet::new(),
            thumbs: HashMap::new(),
            thumbs_requested: HashSet::new(),
            out_dir: None,
            cut_mode: CutMode::Exact,
        }
    }

    fn busy(&self) -> bool {
        !matches!(self.task, Task::Idle)
    }

    fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.cancel.store(true, Ordering::Relaxed);
        self.generation += 1;
        self.loaded = None;
        self.merged.clear();
        self.kind_overrides.clear();
        self.excluded.clear();
        self.thumbs.clear();
        self.thumbs_requested.clear();
        self.out_dir = Some(default_out_dir(&path));
        self.cancel = Arc::new(AtomicBool::new(false));
        self.task = Task::Analyzing(0.0);
        self.status = format!("Analyzing {}…", path.display());

        let (tx, ctx, cancel, generation) = (self.tx.clone(), ctx.clone(), self.cancel.clone(), self.generation);
        std::thread::spawn(move || {
            let result = ffmpeg::probe(&path).and_then(|info| {
                let analysis = analysis::analyze(&path, &info, &cancel, |p| {
                    let _ = tx.send(Msg::AnalysisProgress(p));
                    ctx.request_repaint();
                })?;
                Ok((info, analysis))
            });
            let _ = tx.send(Msg::AnalysisDone { generation, path, result });
            ctx.request_repaint();
        });
    }

    fn on_analysis_done(&mut self, ctx: &egui::Context, path: PathBuf, info: VideoInfo, analysis: Analysis) {
        let thumb_tx = spawn_thumb_worker(ctx.clone(), self.tx.clone(), self.generation, path.clone(), &info, analysis.fps);
        self.status = format!(
            "{} · {}×{} · {:.2} fps · {}{}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            info.width,
            info.height,
            info.fps,
            fmt_time(info.duration),
            if info.has_audio { " · audio" } else { "" },
        );
        self.loaded = Some(Loaded { path, info, analysis, cuts: Vec::new(), scenes: Vec::new(), thumb_tx });
        self.recompute();
    }

    /// Re-run cut detection on the cached scores. Cheap; called whenever settings change.
    fn recompute(&mut self) {
        let Some(l) = &mut self.loaded else { return };
        l.cuts = scenes::detect_cuts(&l.analysis.diffs, l.analysis.fps, &self.params);
        l.scenes = scenes::build_scenes(
            &l.analysis.diffs,
            l.analysis.frame_count(),
            &l.cuts,
            &self.merged,
            self.params.still_threshold,
        );
    }

    fn effective_kind(&self, scene: &Scene) -> SceneKind {
        self.kind_overrides.get(&scene.start).copied().unwrap_or(scene.kind)
    }

    fn start_export(&mut self, ctx: &egui::Context) {
        let (Some(l), Some(out_dir)) = (&self.loaded, self.out_dir.clone()) else { return };
        let items: Vec<ExportItem> = l
            .scenes
            .iter()
            .enumerate()
            .filter(|(_, s)| !self.excluded.contains(&s.start))
            .map(|(index, s)| ExportItem {
                index,
                kind: self.effective_kind(s),
                start: l.analysis.frame_time(s.start),
                end: l.analysis.frame_time(s.end),
            })
            .collect();
        if items.is_empty() {
            self.status = "Nothing selected to export.".into();
            return;
        }

        self.cancel = Arc::new(AtomicBool::new(false));
        self.task = Task::Exporting { done: 0, total: items.len() };
        self.status = format!("Exporting to {}…", out_dir.display());
        let (tx, ctx, cancel) = (self.tx.clone(), ctx.clone(), self.cancel.clone());
        let (input, mode) = (l.path.clone(), self.cut_mode);
        std::thread::spawn(move || {
            let result = export::export_all(&input, &out_dir, &items, mode, &cancel, |done| {
                let _ = tx.send(Msg::ExportProgress(done));
                ctx.request_repaint();
            });
            let _ = tx.send(Msg::ExportDone(result));
            ctx.request_repaint();
        });
    }

    fn handle_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::AnalysisProgress(p) => {
                    if let Task::Analyzing(_) = self.task {
                        self.task = Task::Analyzing(p);
                    }
                }
                Msg::AnalysisDone { generation, path, result } if generation == self.generation => {
                    self.task = Task::Idle;
                    match result {
                        Ok((info, analysis)) => self.on_analysis_done(ctx, path, info, analysis),
                        Err(e) => self.status = format!("Analysis failed: {e:#}"),
                    }
                }
                Msg::Thumb { generation, frame, size, rgba } if generation == self.generation => {
                    let image = ColorImage::from_rgba_unmultiplied(size, &rgba);
                    let tex = ctx.load_texture(format!("thumb-{frame}"), image, TextureOptions::LINEAR);
                    self.thumbs.insert(frame, tex);
                }
                Msg::AnalysisDone { .. } | Msg::Thumb { .. } => {}
                Msg::ExportProgress(done) => {
                    if let Task::Exporting { total, .. } = self.task {
                        self.task = Task::Exporting { done, total };
                    }
                }
                Msg::ExportDone(result) => {
                    self.task = Task::Idle;
                    self.status = match result {
                        Ok(files) => format!("Exported {} files.", files.len()),
                        Err(e) => format!("Export failed: {e:#}"),
                    };
                }
            }
        }
    }

    // ---- UI ---------------------------------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.busy(), egui::Button::new("Open video…")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Video", &["mp4", "mov", "mkv", "avi", "webm", "m4v", "wmv"])
                    .add_filter("All files", &["*"])
                    .pick_file()
                {
                    self.open(ui.ctx(), path);
                }
            }
            match self.task {
                Task::Analyzing(p) => {
                    ui.add(egui::ProgressBar::new(p).desired_width(200.0).show_percentage());
                    if ui.button("Cancel").clicked() {
                        self.cancel.store(true, Ordering::Relaxed);
                    }
                }
                Task::Exporting { done, total } => {
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total as f32)
                            .desired_width(200.0)
                            .text(format!("{done}/{total}")),
                    );
                    if ui.button("Cancel").clicked() {
                        self.cancel.store(true, Ordering::Relaxed);
                    }
                }
                Task::Idle => {}
            }
            ui.label(&self.status);
        });
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        let p = &mut self.params;
        ui.horizontal(|ui| {
            ui.label("Cut detection:");
            changed |= ui.radio_value(&mut p.mode, DetectMode::Adaptive, "Adaptive").changed();
            changed |= ui.radio_value(&mut p.mode, DetectMode::Fixed, "Fixed threshold").changed();
        });
        egui::Grid::new("settings").num_columns(4).spacing([16.0, 4.0]).show(ui, |ui| {
            match p.mode {
                DetectMode::Fixed => {
                    ui.label("Cut threshold");
                    changed |= ui.add(egui::Slider::new(&mut p.cut_threshold, 1.0..=120.0)).changed();
                }
                DetectMode::Adaptive => {
                    ui.label("Sensitivity ratio");
                    changed |= ui.add(egui::Slider::new(&mut p.adaptive_ratio, 1.2..=10.0)).changed();
                }
            }
            ui.label("Min scene length (s)");
            changed |= ui.add(egui::Slider::new(&mut p.min_scene_secs, 0.0..=5.0)).changed();
            ui.end_row();

            if p.mode == DetectMode::Adaptive {
                ui.label("Ignore changes below");
                changed |= ui.add(egui::Slider::new(&mut p.adaptive_floor, 0.0..=60.0)).changed();
            } else {
                ui.label("");
                ui.label("");
            }
            ui.label("Still if motion below")
                .on_hover_text("Scenes whose median frame-to-frame change is below this are exported as a single image.");
            changed |= ui.add(egui::Slider::new(&mut p.still_threshold, 0.0..=10.0)).changed();
            ui.end_row();
        });
        if changed {
            self.recompute();
        }

        let Some(l) = &self.loaded else { return };
        difference_graph(ui, l, &self.params, &self.merged);
        let n = l.scenes.len();
        let stills = l.scenes.iter().filter(|s| self.effective_kind(s) == SceneKind::Still).count();
        ui.horizontal(|ui| {
            ui.label(format!("{n} scenes · {stills} stills · {} clips", n - stills));
            if !self.merged.is_empty() && ui.button(format!("Undo {} merges", self.merged.len())).clicked() {
                self.merged.clear();
                self.recompute();
            }
        });
    }

    fn export_bar(&mut self, ui: &mut egui::Ui) {
        // Wraps on narrow windows. The export button comes first so it's never pushed
        // off-screen, and the path goes last so it can be truncated to whatever space is left.
        ui.horizontal_wrapped(|ui| {
            let count = self
                .loaded
                .as_ref()
                .map_or(0, |l| l.scenes.iter().filter(|s| !self.excluded.contains(&s.start)).count());
            let can_export = !self.busy() && count > 0 && self.out_dir.is_some();
            if ui.add_enabled(can_export, egui::Button::new(format!("Export {count} scenes"))).clicked() {
                self.start_export(ui.ctx());
            }
            ui.separator();
            ui.radio_value(&mut self.cut_mode, CutMode::Exact, "Exact (re-encode)");
            ui.radio_value(&mut self.cut_mode, CutMode::Fast, "Fast (copy, keyframe-aligned)");
            ui.separator();
            ui.label("Output:");
            if ui.button("Change…").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.out_dir = Some(dir);
                }
            }
            let shown = self.out_dir.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "—".into());
            ui.add(egui::Label::new(RichText::new(&shown).monospace()).truncate()).on_hover_text(shown);
        });
    }

    fn scene_list(&mut self, ui: &mut egui::Ui) {
        let Some(l) = &self.loaded else {
            ui.centered_and_justified(|ui| ui.label(RichText::new("No video loaded").weak()));
            return;
        };
        let thumb_h = thumb_height(&l.info);
        let n = l.scenes.len();
        let mut merge_at = None;

        egui::ScrollArea::vertical().auto_shrink(false).show_rows(ui, ROW_HEIGHT, n, |ui, range| {
            for i in range {
                let scene = &l.scenes[i];
                let thumb_frame = scene.middle();
                if self.thumbs_requested.insert(thumb_frame) {
                    let _ = l.thumb_tx.send(thumb_frame);
                }

                ui.horizontal(|ui| {
                    ui.set_height(ROW_HEIGHT);
                    let size = vec2(THUMB_W as f32, thumb_h as f32);
                    match self.thumbs.get(&thumb_frame) {
                        Some(tex) => {
                            ui.add(egui::Image::from_texture((tex.id(), size)));
                        }
                        None => {
                            let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
                            ui.painter().rect_filled(rect, 4.0, ui.visuals().faint_bg_color);
                            ui.put(rect, egui::Spinner::new());
                        }
                    }

                    ui.vertical(|ui| {
                        let (start, end) = (l.analysis.frame_time(scene.start), l.analysis.frame_time(scene.end));
                        ui.label(RichText::new(format!("#{}", i + 1)).strong());
                        ui.label(format!("{} – {}  ({:.1}s)", fmt_time(start), fmt_time(end), end - start));
                        ui.label(RichText::new(format!("motion {:.2}", scene.motion)).weak());

                        ui.horizontal(|ui| {
                            let mut include = !self.excluded.contains(&scene.start);
                            if ui.checkbox(&mut include, "Export").changed() {
                                if include {
                                    self.excluded.remove(&scene.start);
                                } else {
                                    self.excluded.insert(scene.start);
                                }
                            }
                            let mut kind = self.kind_overrides.get(&scene.start).copied().unwrap_or(scene.kind);
                            let before = kind;
                            ui.selectable_value(&mut kind, SceneKind::Video, "🎞 Video");
                            ui.selectable_value(&mut kind, SceneKind::Still, "🖼 Still");
                            if kind != before {
                                self.kind_overrides.insert(scene.start, kind);
                            }
                            if i + 1 < n && ui.button("Merge with next").clicked() {
                                merge_at = Some(scene.end);
                            }
                        });
                    });
                });
                ui.separator();
            }
        });

        if let Some(cut) = merge_at {
            self.merged.insert(cut);
            self.recompute();
        }
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_messages(ctx);
        let dropped = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(path) = dropped {
            if !self.busy() {
                self.open(ctx, path);
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(4.0);
        });
        egui::Panel::top("settings").show(ui, |ui| {
            ui.add_space(4.0);
            self.settings(ui);
            ui.add_space(4.0);
        });
        egui::Panel::bottom("export").show(ui, |ui| {
            ui.add_space(4.0);
            self.export_bar(ui);
            ui.add_space(4.0);
        });
        egui::CentralPanel::default().show(ui, |ui| self.scene_list(ui));
    }
}

/// Plot of the per-frame difference, with cut markers and the active threshold.
fn difference_graph(ui: &mut egui::Ui, l: &Loaded, params: &Params, merged: &HashSet<usize>) {
    let diffs = &l.analysis.diffs;
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 70.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    if diffs.is_empty() {
        return;
    }

    let y_max = diffs.iter().copied().fold(0.0f32, f32::max).clamp(10.0, 120.0);
    let to_y = |v: f32| rect.bottom() - (v / y_max).min(1.0) * rect.height();
    let to_x = |frame: usize| rect.left() + frame as f32 / diffs.len() as f32 * rect.width();

    // One bar per pixel column, showing the largest difference in that column.
    let cols = rect.width().max(1.0) as usize;
    let line = Stroke::new(1.0, ui.visuals().text_color().gamma_multiply(0.6));
    for c in 0..cols {
        let lo = c * diffs.len() / cols;
        let hi = ((c + 1) * diffs.len() / cols).max(lo + 1).min(diffs.len());
        let v = diffs[lo..hi].iter().copied().fold(0.0f32, f32::max);
        let x = rect.left() + c as f32 + 0.5;
        painter.vline(x, to_y(v)..=rect.bottom(), line);
    }

    let accent = Color32::from_rgb(230, 120, 40);
    for &cut in &l.cuts {
        let color = if merged.contains(&cut) { Color32::GRAY } else { accent };
        painter.vline(to_x(cut), rect.y_range(), Stroke::new(1.0, color.gamma_multiply(0.7)));
    }
    let threshold = match params.mode {
        DetectMode::Fixed => params.cut_threshold,
        DetectMode::Adaptive => params.adaptive_floor,
    };
    painter.hline(rect.x_range(), to_y(threshold), Stroke::new(1.0, Color32::from_rgb(80, 160, 230)));

    if let Some(pos) = response.hover_pos() {
        let frame = (((pos.x - rect.left()) / rect.width()) * diffs.len() as f32) as usize;
        let frame = frame.min(diffs.len() - 1);
        painter.vline(pos.x, rect.y_range(), Stroke::new(1.0, ui.visuals().strong_text_color()));
        painter.text(
            pos2(pos.x + 4.0, rect.top() + 2.0),
            egui::Align2::LEFT_TOP,
            format!("{}  diff {:.1}", fmt_time(l.analysis.frame_time(frame)), diffs[frame]),
            egui::FontId::monospace(11.0),
            ui.visuals().strong_text_color(),
        );
    }
}

/// Background thread that decodes thumbnails on request. Exits when the sender is dropped
/// (i.e. when another video is opened).
fn spawn_thumb_worker(
    ctx: egui::Context,
    tx: Sender<Msg>,
    generation: u64,
    path: PathBuf,
    info: &VideoInfo,
    fps: f64,
) -> Sender<usize> {
    let (req_tx, req_rx) = channel::<usize>();
    let h = thumb_height(info);
    std::thread::spawn(move || {
        let mut queue = Vec::new();
        while let Ok(first) = req_rx.recv() {
            queue.push(first);
            // Serve the most recent requests first: they're what's on screen now.
            loop {
                queue.extend(req_rx.try_iter());
                let Some(frame) = queue.pop() else { break };
                match ffmpeg::grab_frame_rgba(&path, frame as f64 / fps, THUMB_W, h) {
                    Ok(rgba) => {
                        let msg = Msg::Thumb { generation, frame, size: [THUMB_W as usize, h as usize], rgba };
                        if tx.send(msg).is_err() {
                            return;
                        }
                        ctx.request_repaint();
                    }
                    Err(_) => continue,
                }
            }
        }
    });
    req_tx
}

fn thumb_height(info: &VideoInfo) -> u32 {
    if info.width == 0 {
        return THUMB_W * 9 / 16;
    }
    // Even number, which keeps ffmpeg's scaler happy.
    ((THUMB_W as f64 * info.height as f64 / info.width as f64).round() as u32 / 2 * 2).clamp(2, ROW_HEIGHT as u32 - 4)
}

fn default_out_dir(input: &Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    input.parent().unwrap_or(Path::new(".")).join(format!("{stem}_scenes"))
}

fn fmt_time(secs: f64) -> String {
    let secs = secs.max(0.0);
    let m = (secs / 60.0).floor() as u64;
    format!("{m}:{:04.1}", secs - m as f64 * 60.0)
}
