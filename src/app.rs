use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use anyhow::Result;
use eframe::egui::{
    self, Color32, ColorImage, Key, Rect, RichText, Sense, Shape, Stroke, TextureHandle, TextureOptions, pos2, vec2,
};

use crate::analysis::{self, Analysis};
use crate::export::{self, CutMode, ExportItem};
use crate::ffmpeg::{self, VideoInfo};
use crate::player::Player;
use crate::scenes::{self, DetectMode, Params, Scene, SceneKind};

const THUMB_W: u32 = 192;
const ROW_HEIGHT: f32 = 112.0;

const ACCENT: Color32 = Color32::from_rgb(230, 120, 40);
const VIDEO_COLOR: Color32 = Color32::from_rgb(70, 130, 200);
const STILL_COLOR: Color32 = Color32::from_rgb(90, 170, 110);

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

/// Player requests collected while drawing the UI and applied afterwards, which keeps the
/// drawing code free of borrow conflicts.
enum Action {
    Seek(usize),
    Step(i64),
    TogglePlay,
    /// Select scene by index and jump to it, optionally starting playback.
    SelectScene { index: usize, play: bool },
}

struct Loaded {
    path: PathBuf,
    info: VideoInfo,
    analysis: Analysis,
    cuts: Vec<usize>,
    scenes: Vec<Scene>,
    thumb_tx: Sender<usize>,
    player: Player,
}

impl Loaded {
    /// Index of the scene containing `frame`.
    fn scene_at(&self, frame: usize) -> usize {
        self.scenes.partition_point(|s| s.start <= frame).saturating_sub(1)
    }
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
    /// Detected cuts the user merged away.
    merged: HashSet<usize>,
    /// Per-scene user choices, keyed by `Scene::id`.
    kind_overrides: HashMap<usize, SceneKind>,
    excluded: HashSet<usize>,
    /// `Scene::id` of the selected scene.
    selected: Option<usize>,
    loop_scene: bool,

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
            selected: None,
            loop_scene: true,
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
        self.selected = None;
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
        let player = Player::new(&path, &info, analysis.fps, analysis.frame_count());
        self.status = format!(
            "{} · {}×{} · {:.2} fps · {}{}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            info.width,
            info.height,
            info.fps,
            fmt_time(info.duration),
            if info.has_audio { " · audio" } else { "" },
        );
        self.loaded = Some(Loaded { path, info, analysis, cuts: Vec::new(), scenes: Vec::new(), thumb_tx, player });
        self.recompute();
    }

    /// Re-run cut detection on the cached scores. Cheap; called whenever settings change.
    fn recompute(&mut self) {
        let Some(l) = &mut self.loaded else { return };
        l.cuts = scenes::detect_cuts(&l.analysis.diffs, l.analysis.fps, &self.params);
        l.scenes = scenes::build_scenes(&l.analysis.diffs, l.analysis.frame_count(), &l.cuts, &self.merged, &self.params);
        if self.selected.is_some_and(|id| !l.scenes.iter().any(|s| s.id == id)) {
            self.selected = None;
        }
        self.sync_loop_range();
    }

    /// Keep the player's loop range on the selected scene's exported frames.
    fn sync_loop_range(&mut self) {
        let Some(l) = &mut self.loaded else { return };
        l.player.loop_range = self
            .selected
            .filter(|_| self.loop_scene)
            .and_then(|id| l.scenes.iter().find(|s| s.id == id))
            .map(|s| s.keep.clone())
            .filter(|r| !r.is_empty());
    }

    fn effective_kind(&self, scene: &Scene) -> SceneKind {
        self.kind_overrides.get(&scene.id).copied().unwrap_or(scene.kind)
    }

    fn will_export(&self, scene: &Scene) -> bool {
        !scene.keep.is_empty() && !self.excluded.contains(&scene.id)
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::SelectScene { index, play } => {
                let Some(scene) = self.loaded.as_ref().and_then(|l| l.scenes.get(index)) else { return };
                let (id, frame) = (scene.id, if scene.keep.is_empty() { scene.start } else { scene.keep.start });
                self.selected = Some(id);
                self.sync_loop_range();
                let player = &mut self.loaded.as_mut().unwrap().player;
                player.seek(frame);
                if play {
                    player.play();
                }
            }
            other => {
                let Some(l) = &mut self.loaded else { return };
                match other {
                    Action::Seek(frame) => l.player.seek(frame),
                    Action::Step(delta) => l.player.step(delta),
                    Action::TogglePlay => l.player.toggle(),
                    Action::SelectScene { .. } => unreachable!(),
                }
            }
        }
    }

    fn start_export(&mut self, ctx: &egui::Context) {
        let (Some(l), Some(out_dir)) = (&self.loaded, self.out_dir.clone()) else { return };
        let items: Vec<ExportItem> = l
            .scenes
            .iter()
            .enumerate()
            .filter(|(_, s)| self.will_export(s))
            .map(|(index, s)| ExportItem {
                index,
                kind: self.effective_kind(s),
                start: l.analysis.frame_time(s.keep.start),
                end: l.analysis.frame_time(s.keep.end),
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

    fn keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if self.loaded.is_none() || ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (space, left, right) =
            ctx.input(|i| (i.key_pressed(Key::Space), i.key_pressed(Key::ArrowLeft), i.key_pressed(Key::ArrowRight)));
        if space {
            self.apply(Action::TogglePlay);
        }
        if left {
            self.apply(Action::Step(-1));
        }
        if right {
            self.apply(Action::Step(1));
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

    fn settings(&mut self, ui: &mut egui::Ui) -> Option<Action> {
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

            ui.label("Cut offset (frames)").on_hover_text(
                "Moves every cut relative to the detected change.\n\
                 0: the new scene starts on the first changed frame.\n\
                 Negative: earlier. Positive: later.",
            );
            changed |= ui.add(egui::DragValue::new(&mut p.cut_offset).range(-60..=60).speed(0.1)).changed();
            ui.label("Drop frames before / after cut").on_hover_text(
                "Frames removed from the end of the outgoing scene and the start of the incoming one, \
                 e.g. to skip transition or blended frames. Not applied at the start or end of the video.",
            );
            ui.horizontal(|ui| {
                changed |= ui.add(egui::DragValue::new(&mut p.drop_before_cut).range(0..=600).speed(0.1)).changed();
                ui.label("/");
                changed |= ui.add(egui::DragValue::new(&mut p.drop_after_cut).range(0..=600).speed(0.1)).changed();
            });
            ui.end_row();
        });
        if changed {
            self.recompute();
        }

        let l = self.loaded.as_ref()?;
        let action = difference_graph(ui, l, &self.params, &self.merged).map(Action::Seek);
        let n = l.scenes.len();
        let stills = l.scenes.iter().filter(|s| self.effective_kind(s) == SceneKind::Still).count();
        let empty = l.scenes.iter().filter(|s| s.keep.is_empty()).count();
        ui.horizontal(|ui| {
            ui.label(format!("{n} scenes · {stills} stills · {} clips", n - stills));
            if empty > 0 {
                ui.label(RichText::new(format!("· {empty} trimmed to nothing")).color(ui.visuals().warn_fg_color));
            }
            if !self.merged.is_empty() && ui.button(format!("Undo {} merges", self.merged.len())).clicked() {
                self.merged.clear();
                self.recompute();
            }
        });
        action
    }

    fn export_bar(&mut self, ui: &mut egui::Ui) {
        // Wraps on narrow windows. The export button comes first so it's never pushed
        // off-screen, and the path goes last so it can be truncated to whatever space is left.
        ui.horizontal_wrapped(|ui| {
            let count = self.loaded.as_ref().map_or(0, |l| l.scenes.iter().filter(|s| self.will_export(s)).count());
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

    fn player_panel(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let Some(l) = &self.loaded else {
            ui.centered_and_justified(|ui| ui.label(RichText::new("No video loaded").weak()));
            return None;
        };
        let mut action = None;
        let player = &l.player;

        let size = vec2(ui.available_width(), ui.available_width() / player.aspect());
        match player.texture() {
            Some(tex) => {
                let resp = ui.add(egui::Image::from_texture((tex.id(), size)).sense(Sense::click()));
                if resp.clicked() {
                    action = Some(Action::TogglePlay);
                }
            }
            None => {
                let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
                ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                ui.put(rect, egui::Spinner::new());
            }
        }

        if let Some(frame) = timeline(ui, l, self.selected) {
            action = Some(Action::Seek(frame));
        }

        let current = player.current_frame();
        let here = l.scene_at(current);
        ui.horizontal_wrapped(|ui| {
            if ui.button("⏮").on_hover_text("Previous scene").clicked() {
                // Restart the current scene unless we're already at its beginning.
                let at_start = l.scenes.get(here).is_none_or(|s| current <= s.keep.start + 2);
                let index = if at_start { here.saturating_sub(1) } else { here };
                action = Some(Action::SelectScene { index, play: player.is_playing() });
            }
            if ui.button("◀").on_hover_text("Previous frame (←)").clicked() {
                action = Some(Action::Step(-1));
            }
            let play_label = if player.is_playing() { "⏸" } else { "▶" };
            if ui.button(play_label).on_hover_text("Play / pause (Space)").clicked() {
                action = Some(Action::TogglePlay);
            }
            if ui.button("▶|").on_hover_text("Next frame (→)").clicked() {
                action = Some(Action::Step(1));
            }
            if ui.button("⏭").on_hover_text("Next scene").clicked() && here + 1 < l.scenes.len() {
                action = Some(Action::SelectScene { index: here + 1, play: player.is_playing() });
            }
            ui.monospace(format!(
                "{} / {}  #{current}",
                fmt_time(current as f64 / player.fps()),
                fmt_time(player.frame_count() as f64 / player.fps())
            ));
        });
        if ui
            .checkbox(&mut self.loop_scene, "Loop selected scene")
            .on_hover_text("Plays exactly the frames that will be exported for the selected scene.")
            .changed()
        {
            self.sync_loop_range();
        }
        action
    }

    fn scene_list(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let Some(l) = &self.loaded else {
            ui.centered_and_justified(|ui| ui.label(RichText::new("No video loaded").weak()));
            return None;
        };
        let thumb_h = thumb_height(&l.info);
        let n = l.scenes.len();
        let mut action = None;
        let mut merge_at = None;

        egui::ScrollArea::vertical().auto_shrink(false).show_rows(ui, ROW_HEIGHT, n, |ui, range| {
            for i in range {
                let scene = &l.scenes[i];
                let thumb_frame = scene.middle();
                if self.thumbs_requested.insert(thumb_frame) {
                    let _ = l.thumb_tx.send(thumb_frame);
                }
                let is_selected = self.selected == Some(scene.id);
                let background = ui.painter().add(Shape::Noop);

                let row = ui.horizontal(|ui| {
                    ui.set_height(ROW_HEIGHT);
                    let size = vec2(THUMB_W as f32, thumb_h as f32);
                    let thumb = match self.thumbs.get(&thumb_frame) {
                        Some(tex) => ui.add(egui::Image::from_texture((tex.id(), size)).sense(Sense::click())),
                        None => {
                            let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
                            ui.painter().rect_filled(rect, 4.0, ui.visuals().faint_bg_color);
                            ui.put(rect, egui::Spinner::new());
                            resp
                        }
                    };
                    if thumb.on_hover_text("Click to preview").clicked() {
                        action = Some(Action::SelectScene { index: i, play: true });
                    }

                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("#{}", i + 1)).strong());
                            if scene.keep.is_empty() {
                                ui.label(RichText::new("trimmed to nothing").color(ui.visuals().warn_fg_color));
                            } else {
                                let (start, end) =
                                    (l.analysis.frame_time(scene.keep.start), l.analysis.frame_time(scene.keep.end));
                                ui.label(format!("{} – {}  ({:.1}s)", fmt_time(start), fmt_time(end), end - start));
                            }
                        });
                        ui.label(
                            RichText::new(format!(
                                "frames {}–{} · motion {:.2}",
                                scene.keep.start,
                                scene.keep.end.saturating_sub(1),
                                scene.motion
                            ))
                            .weak(),
                        );

                        ui.horizontal(|ui| {
                            let mut include = !self.excluded.contains(&scene.id);
                            let checkbox = ui.add_enabled(!scene.keep.is_empty(), egui::Checkbox::new(&mut include, "Export"));
                            if checkbox.changed() {
                                if include {
                                    self.excluded.remove(&scene.id);
                                } else {
                                    self.excluded.insert(scene.id);
                                }
                            }
                            let mut kind = self.kind_overrides.get(&scene.id).copied().unwrap_or(scene.kind);
                            let before = kind;
                            ui.selectable_value(&mut kind, SceneKind::Video, "🎞 Video");
                            ui.selectable_value(&mut kind, SceneKind::Still, "🖼 Still");
                            if kind != before {
                                self.kind_overrides.insert(scene.id, kind);
                            }
                            if ui.button("▶ Preview").clicked() {
                                action = Some(Action::SelectScene { index: i, play: true });
                            }
                            if i + 1 < n && ui.button("Merge with next").clicked() {
                                merge_at = Some(l.scenes[i + 1].id);
                            }
                        });
                    });
                });

                if is_selected {
                    let rect = row.response.rect.expand2(vec2(4.0, 2.0));
                    let fill = ui.visuals().selection.bg_fill.gamma_multiply(0.35);
                    ui.painter().set(background, Shape::rect_filled(rect, 4.0, fill));
                }
                ui.separator();
            }
        });

        if let Some(cut) = merge_at {
            self.merged.insert(cut);
            self.recompute();
        }
        action
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
        self.keyboard_shortcuts(ctx);
        if let Some(l) = &mut self.loaded {
            l.player.tick(ctx);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let mut actions = Vec::new();
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(4.0);
        });
        egui::Panel::top("settings").show(ui, |ui| {
            ui.add_space(4.0);
            actions.extend(self.settings(ui));
            ui.add_space(4.0);
        });
        egui::Panel::bottom("export").show(ui, |ui| {
            ui.add_space(4.0);
            self.export_bar(ui);
            ui.add_space(4.0);
        });
        egui::Panel::right("player").resizable(true).default_size(480.0).min_size(240.0).show(ui, |ui| {
            ui.add_space(4.0);
            actions.extend(self.player_panel(ui));
        });
        egui::CentralPanel::default().show(ui, |ui| actions.extend(self.scene_list(ui)));

        for action in actions {
            self.apply(action);
        }
    }
}

/// Plot of the per-frame difference, with cut markers and the active threshold.
/// Returns a frame to seek to when clicked or dragged.
fn difference_graph(ui: &mut egui::Ui, l: &Loaded, params: &Params, merged: &HashSet<usize>) -> Option<usize> {
    let diffs = &l.analysis.diffs;
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 70.0), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    if diffs.is_empty() {
        return None;
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

    for &cut in merged {
        painter.vline(to_x(cut), rect.y_range(), Stroke::new(1.0, Color32::GRAY.gamma_multiply(0.7)));
    }
    for scene in l.scenes.iter().skip(1) {
        painter.vline(to_x(scene.start), rect.y_range(), Stroke::new(1.0, ACCENT.gamma_multiply(0.7)));
    }
    let threshold = match params.mode {
        DetectMode::Fixed => params.cut_threshold,
        DetectMode::Adaptive => params.adaptive_floor,
    };
    painter.hline(rect.x_range(), to_y(threshold), Stroke::new(1.0, VIDEO_COLOR));
    painter.vline(to_x(l.player.current_frame()), rect.y_range(), Stroke::new(1.5, ui.visuals().strong_text_color()));

    let frame_at = |x: f32| ((((x - rect.left()) / rect.width()) * diffs.len() as f32) as usize).min(diffs.len() - 1);
    if let Some(pos) = response.hover_pos() {
        let frame = frame_at(pos.x);
        painter.vline(pos.x, rect.y_range(), Stroke::new(1.0, ui.visuals().weak_text_color()));
        painter.text(
            pos2(pos.x + 4.0, rect.top() + 2.0),
            egui::Align2::LEFT_TOP,
            format!("{}  diff {:.1}", fmt_time(l.analysis.frame_time(frame)), diffs[frame]),
            egui::FontId::monospace(11.0),
            ui.visuals().strong_text_color(),
        );
    }
    seek_target(&response, l, frame_at)
}

/// Scrubbable bar showing every scene's exported frames (blue = clip, green = still),
/// trimmed gaps, the selected scene, and the playhead.
fn timeline(ui: &mut egui::Ui, l: &Loaded, selected: Option<usize>) -> Option<usize> {
    let total = l.player.frame_count().max(1);
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let to_x = |frame: usize| rect.left() + frame as f32 / total as f32 * rect.width();

    for scene in &l.scenes {
        if scene.keep.is_empty() {
            continue;
        }
        let color = match scene.kind {
            SceneKind::Video => VIDEO_COLOR,
            SceneKind::Still => STILL_COLOR,
        };
        let r = Rect::from_x_y_ranges(to_x(scene.keep.start)..=to_x(scene.keep.end), rect.y_range()).shrink2(vec2(0.5, 3.0));
        painter.rect_filled(r, 1.0, color.gamma_multiply(0.6));
        if selected == Some(scene.id) {
            painter.rect_stroke(r.expand(2.0), 2.0, Stroke::new(1.5, ACCENT), egui::StrokeKind::Outside);
        }
    }
    let x = to_x(l.player.current_frame());
    painter.vline(x, rect.y_range(), Stroke::new(2.0, ui.visuals().strong_text_color()));

    let frame_at = |x: f32| ((((x - rect.left()) / rect.width()) * total as f32) as usize).min(total - 1);
    seek_target(&response, l, frame_at)
}

fn seek_target(response: &egui::Response, l: &Loaded, frame_at: impl Fn(f32) -> usize) -> Option<usize> {
    if !(response.clicked() || response.dragged()) {
        return None;
    }
    let frame = frame_at(response.interact_pointer_pos()?.x);
    // Avoid restarting the decoder every UI frame while the mouse is held still.
    (frame != l.player.current_frame()).then_some(frame)
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
