use std::collections::{HashMap, HashSet};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use anyhow::Result;
use eframe::egui::{
    self, Color32, ColorImage, RichText, Sense, Shape, TextureHandle, TextureOptions, vec2,
};

use crate::analysis::{self, Analysis};
use crate::cache;
use crate::editor::{self, Action, EditorView, fmt_time};
use crate::export::{self, CutMode, ExportItem};
use crate::ffmpeg::{self, VideoInfo};
use crate::player::Player;
use crate::project::{self, Edits, History, Project};
use crate::scenes::{self, CutEdits, CutId, CutSource, DetectMode, Detected, Params, ResolvedCuts, Scene, SceneKind};
use crate::sections::{self, Lock, Section, Span};

const THUMB_W: u32 = 192;
const STRIP_W: u32 = 160;
const ROW_HEIGHT: f32 = 120.0;
/// Height of a scene row's content; the separator sits below it.
const ROW_SEPARATOR_AT: f32 = 104.0;
/// Edits are saved this long after the last change.
const SAVE_DELAY: Duration = Duration::from_millis(800);
const NOTICE_DURATION: Duration = Duration::from_secs(4);


/// Messages from background threads. `generation` ties results to the video they were
/// started for, so late results from a previously opened file are dropped.
enum Msg {
    AnalysisProgress(f32),
    /// `cached` is true when the analysis came from the on-disk cache.
    AnalysisDone { generation: u64, path: PathBuf, result: Result<(VideoInfo, Analysis, bool)> },
    Frames { generation: u64, kind: FrameKind, start: usize, size: [usize; 2], frames: Vec<Vec<u8>> },
    ExportProgress(usize),
    ExportDone(Result<Vec<PathBuf>>),
    FfmpegChecked(Result<String>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    Thumb,
    Strip,
}

struct FrameJob {
    kind: FrameKind,
    start: usize,
    count: usize,
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
    /// Which settings apply where.
    spans: Vec<Span>,
    /// Cuts found by detection, before the user's edits.
    detected: Vec<Detected>,
    cuts: ResolvedCuts,
    scenes: Vec<Scene>,
    frame_tx: Sender<FrameJob>,
    player: Player,
}

impl Loaded {
    /// Index of the scene containing `frame`.
    fn scene_at(&self, frame: usize) -> usize {
        self.scenes.partition_point(|s| s.start <= frame).saturating_sub(1)
    }

    fn frame_count(&self) -> usize {
        self.analysis.frame_count()
    }
}

pub struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    generation: u64,
    task: Task,
    cancel: Arc<AtomicBool>,
    status: String,
    notice: Option<(String, Instant)>,
    /// Result of checking for ffmpeg at startup: its version, or why it can't be used.
    ffmpeg: Option<Result<String, String>>,

    params: Params,
    edits: Edits,
    history: History,
    loaded: Option<Loaded>,
    /// Video the current params/edits belong to (set as soon as a file is opened).
    video_path: Option<PathBuf>,
    dirty_since: Option<Instant>,
    /// Set when the saved project couldn't be read, so we don't overwrite it.
    save_blocked: bool,

    /// `Scene::id` of the selected scene.
    selected: Option<CutId>,
    /// Other end of a range of selected scenes (Shift+click).
    selected_to: Option<CutId>,
    /// Section whose settings the settings panel edits (`None`: the whole video).
    selected_section: Option<u32>,
    selected_cut: Option<CutId>,
    loop_scene: bool,
    view: EditorView,
    /// Scene list row to bring into view on the next frame.
    scroll_to_scene: Option<usize>,
    /// Rows fully visible in the scene list last frame.
    visible_rows: std::ops::Range<usize>,

    thumbs: HashMap<usize, TextureHandle>,
    thumbs_requested: HashSet<usize>,
    strip: HashMap<usize, TextureHandle>,
    strip_requested: HashSet<usize>,

    out_dir: Option<PathBuf>,
    cut_mode: CutMode,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let app = Self::empty();
        let (tx, ctx) = (app.tx.clone(), cc.egui_ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Msg::FfmpegChecked(ffmpeg::check()));
            ctx.request_repaint();
        });
        app
    }

    fn empty() -> Self {
        let (tx, rx) = channel();
        Self {
            tx,
            rx,
            generation: 0,
            task: Task::Idle,
            cancel: Arc::new(AtomicBool::new(false)),
            status: "Open a video to begin (or drop one on the window).".into(),
            notice: None,
            ffmpeg: None,
            params: Params::default(),
            edits: Edits::default(),
            history: History::default(),
            loaded: None,
            video_path: None,
            dirty_since: None,
            save_blocked: false,
            selected: None,
            selected_to: None,
            selected_section: None,
            selected_cut: None,
            loop_scene: true,
            view: EditorView::new(1),
            scroll_to_scene: None,
            visible_rows: 0..0,
            thumbs: HashMap::new(),
            thumbs_requested: HashSet::new(),
            strip: HashMap::new(),
            strip_requested: HashSet::new(),
            out_dir: None,
            cut_mode: CutMode::Exact,
        }
    }

    fn busy(&self) -> bool {
        !matches!(self.task, Task::Idle)
    }

    fn notify(&mut self, text: impl Into<String>) {
        self.notice = Some((text.into(), Instant::now()));
    }

    fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        if self.dirty_since.is_some() {
            self.save_project();
        }
        self.cancel.store(true, Ordering::Relaxed);
        self.generation += 1;
        self.loaded = None;
        self.selected = None;
        self.selected_to = None;
        self.selected_section = None;
        self.selected_cut = None;
        self.thumbs.clear();
        self.thumbs_requested.clear();
        self.strip.clear();
        self.strip_requested.clear();
        self.history = History::default();
        self.dirty_since = None;
        self.save_blocked = false;
        self.out_dir = Some(default_out_dir(&path));

        // Settings carry over from the previous video unless this one has its own saved.
        self.edits = Edits::default();
        match project::load(&path) {
            Ok(Some(p)) => {
                self.params = p.params;
                self.edits = p.edits;
                self.notify(format!("Loaded saved edits from {}", project::sidecar_path(&path).display()));
            }
            Ok(None) => {}
            Err(e) => {
                self.save_blocked = true;
                self.notify(format!("{e:#}. Edits won't be saved, to avoid overwriting it."));
            }
        }
        self.video_path = Some(path.clone());

        self.cancel = Arc::new(AtomicBool::new(false));
        self.task = Task::Analyzing(0.0);
        self.status = format!("Analyzing {}…", path.display());

        let (tx, ctx, cancel, generation) = (self.tx.clone(), ctx.clone(), self.cancel.clone(), self.generation);
        std::thread::spawn(move || {
            let result = ffmpeg::probe(&path).and_then(|info| {
                if let Some(analysis) = cache::load(&path) {
                    return Ok((info, analysis, true));
                }
                let analysis = analysis::analyze(&path, &info, &cancel, |p| {
                    let _ = tx.send(Msg::AnalysisProgress(p));
                    ctx.request_repaint();
                })?;
                // A failed cache write only costs a re-analysis next time.
                let _ = cache::save(&path, &analysis);
                Ok((info, analysis, false))
            });
            let _ = tx.send(Msg::AnalysisDone { generation, path, result });
            ctx.request_repaint();
        });
    }

    fn on_analysis_done(&mut self, ctx: &egui::Context, path: PathBuf, info: VideoInfo, analysis: Analysis) {
        let frame_tx = spawn_frame_worker(ctx.clone(), self.tx.clone(), self.generation, path.clone(), &info, analysis.fps);
        let player = Player::new(&path, &info, analysis.fps, analysis.frame_count());
        self.view = EditorView::new(analysis.frame_count());
        self.status = format!(
            "{} · {}×{} · {:.2} fps · {}{}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            info.width,
            info.height,
            info.fps,
            fmt_time(info.duration),
            if info.has_audio { " · audio" } else { "" },
        );
        self.loaded = Some(Loaded {
            path,
            info,
            analysis,
            spans: Vec::new(),
            detected: Vec::new(),
            cuts: ResolvedCuts::default(),
            scenes: Vec::new(),
            frame_tx,
            player,
        });
        self.recompute();
    }

    /// Re-run detection and apply edits. Cheap; called whenever settings or edits change.
    fn recompute(&mut self) {
        let Some(l) = &mut self.loaded else { return };
        let fc = l.frame_count();
        l.spans = sections::plan(&self.params, &self.edits.sections, fc);
        l.detected = scenes::detect(&l.analysis.diffs, l.analysis.fps, &l.spans);
        l.cuts = scenes::resolve(&l.detected, &self.edits.cuts, fc);
        l.scenes = scenes::build(&l.analysis.diffs, fc, &l.cuts.cuts, &l.spans);
        if self.selected.is_some_and(|id| !l.scenes.iter().any(|s| s.id == id)) {
            self.selected = None;
        }
        if self.selected.is_none() || self.selected_to.is_some_and(|id| !l.scenes.iter().any(|s| s.id == id)) {
            self.selected_to = None;
        }
        if self.selected_section.is_some_and(|id| self.edits.sections.get(id).is_none()) {
            self.selected_section = None;
        }
        if self.selected_cut.is_some_and(|id| l.cuts.get(id).is_none()) {
            self.selected_cut = None;
        }
        self.sync_loop_range();
    }

    fn mark_dirty(&mut self) {
        self.dirty_since = Some(Instant::now());
    }

    fn save_project(&mut self) {
        self.dirty_since = None;
        let Some(path) = &self.video_path else { return };
        if self.save_blocked {
            return;
        }
        let project = Project { params: self.params.clone(), edits: self.edits.clone(), ..Project::default() };
        if let Err(e) = project::save(path, &project) {
            self.notify(format!("Couldn't save edits: {e:#}"));
        }
    }

    /// Record an undo step, change the edits, and refresh.
    fn edit(&mut self, coalesce_key: Option<String>, change: impl FnOnce(&mut Edits)) {
        self.history.checkpoint(&self.edits, coalesce_key);
        change(&mut self.edits);
        self.recompute();
        self.mark_dirty();
    }

    /// Indices of the selected scenes.
    fn selection(&self) -> Option<RangeInclusive<usize>> {
        let l = self.loaded.as_ref()?;
        let index = |id: CutId| l.scenes.iter().position(|s| s.id == id);
        let a = index(self.selected?)?;
        let b = self.selected_to.and_then(index).unwrap_or(a);
        Some(a.min(b)..=a.max(b))
    }

    /// Keep the player's loop range on the selected scenes' exported frames.
    fn sync_loop_range(&mut self) {
        let selection = self.selection().filter(|_| self.loop_scene);
        let Some(l) = &mut self.loaded else { return };
        l.player.loop_range = selection
            .map(|r| l.scenes[*r.start()].keep.start..l.scenes[*r.end()].keep.end)
            .filter(|r| !r.is_empty());
    }

    fn effective_kind(&self, scene: &Scene) -> SceneKind {
        self.edits.kinds.get(&scene.id).copied().unwrap_or(scene.kind)
    }

    fn will_export(&self, scene: &Scene) -> bool {
        !scene.keep.is_empty() && !self.edits.excluded.contains(&scene.id)
    }

    /// Move a cut (no undo step), clamped between its neighbours, and preview the new first frame.
    fn move_cut(&mut self, id: CutId, frame: usize) {
        let Some(l) = &self.loaded else { return };
        let Some(cut) = l.cuts.get(id).copied() else { return };
        let range = l.cuts.movable_range(id, l.frame_count());
        let frame = frame.clamp(range.start, range.end - 1);
        if frame != cut.frame {
            if cut.ghost == Some(frame) {
                // Back at the detected position: it's no longer an edit.
                self.edits.cuts.reset(id);
            } else {
                self.edits.cuts.set_frame(id, frame);
            }
            self.recompute();
            self.mark_dirty();
        }
        if let Some(l) = &mut self.loaded {
            l.player.pause();
            l.player.seek(frame);
        }
    }

    fn apply(&mut self, action: Action) {
        let Some(l) = &mut self.loaded else { return };
        let position = l.player.position();
        match action {
            Action::Seek(frame) => l.player.seek(frame),
            Action::Step(delta) => l.player.step(delta),
            Action::TogglePlay => l.player.toggle(),
            Action::SelectScene { index, play } => {
                let Some(scene) = l.scenes.get(index) else { return };
                let (id, frame) = (scene.id, if scene.keep.is_empty() { scene.start } else { scene.keep.start });
                self.selected = Some(id);
                self.selected_to = None;
                self.scroll_to_scene = Some(index);
                self.sync_loop_range();
                let player = &mut self.loaded.as_mut().unwrap().player;
                player.seek(frame);
                if play {
                    player.play();
                }
            }
            Action::SelectSceneAt { index, frame } => {
                let Some(scene) = l.scenes.get(index) else { return };
                self.selected = Some(scene.id);
                self.selected_to = None;
                self.scroll_to_scene = Some(index);
                self.sync_loop_range();
                self.loaded.as_mut().unwrap().player.seek(frame);
            }
            Action::ExtendSelection(index) => {
                let Some(scene) = l.scenes.get(index) else { return };
                if self.selected.is_none() {
                    self.selected = Some(scene.id);
                } else {
                    self.selected_to = Some(scene.id);
                }
                self.scroll_to_scene = Some(index);
                self.sync_loop_range();
            }
            Action::SelectCut(id) => self.selected_cut = id,
            Action::Checkpoint => self.history.checkpoint(&self.edits, None),
            Action::MoveCut { id, frame } => self.move_cut(id, frame),
            Action::Split => self.apply(Action::SplitAt(position)),
            Action::SplitAt(frame) => {
                if frame == 0 || frame >= l.frame_count() {
                    self.notify("Can't split at the very start or end of the video.");
                } else if l.cuts.cuts.iter().any(|c| c.frame == frame) {
                    self.notify(format!("There's already a cut at frame {frame}."));
                } else {
                    let mut new_id = None;
                    self.edit(None, |e| new_id = Some(e.cuts.add(frame)));
                    self.selected_cut = new_id;
                }
            }
            Action::DeleteCut(id) => {
                self.edit(None, |e| e.cuts.delete(id));
                if self.selected_cut == Some(id) {
                    self.selected_cut = None;
                }
            }
            Action::DeleteSelected => match self.selected_cut {
                Some(id) => self.apply(Action::DeleteCut(id)),
                None => self.notify("Select a cut first: click its marker, or use [ and ] to jump to one."),
            },
            Action::ResetCut(id) => self.edit(None, |e| e.cuts.reset(id)),
            Action::ResetAllCuts => self.edit(None, |e| {
                e.cuts = CutEdits { next_manual_id: e.cuts.next_manual_id, ..CutEdits::default() };
            }),
            Action::Nudge(delta) => {
                let target = self
                    .selected_cut
                    .and_then(|id| l.cuts.get(id))
                    .or_else(|| l.cuts.cuts.iter().min_by_key(|c| c.frame.abs_diff(position)))
                    .copied();
                let Some(cut) = target else {
                    self.notify("There are no cuts to nudge.");
                    return;
                };
                self.selected_cut = Some(cut.id);
                self.history.checkpoint(&self.edits, Some(format!("nudge {}", cut.id)));
                self.move_cut(cut.id, cut.frame.saturating_add_signed(delta as isize).max(1));
            }
            Action::JumpCut(dir) => {
                let target = if dir < 0 {
                    l.cuts.cuts.iter().rev().find(|c| c.frame < position)
                } else {
                    l.cuts.cuts.iter().find(|c| c.frame > position)
                };
                if let Some(cut) = target.copied() {
                    self.selected_cut = Some(cut.id);
                    l.player.pause();
                    l.player.seek(cut.frame);
                }
            }
            Action::Undo | Action::Redo => {
                let done = if matches!(action, Action::Undo) {
                    self.history.undo(&mut self.edits)
                } else {
                    self.history.redo(&mut self.edits)
                };
                if done {
                    self.recompute();
                    self.mark_dirty();
                }
            }
            Action::ZoomToSelectedScene => {
                if let Some(r) = self.selection() {
                    let l = self.loaded.as_ref().unwrap();
                    self.view.show_range(l.scenes[*r.start()].start..l.scenes[*r.end()].end);
                }
            }
            Action::SetKind(id, kind) => self.edit(None, |e| {
                e.kinds.insert(id, kind);
            }),
            Action::SetExcluded(id, excluded) => self.edit(None, |e| {
                if excluded {
                    e.excluded.insert(id);
                } else {
                    e.excluded.remove(&id);
                }
            }),
            Action::SelectSection(id) => self.selected_section = id,
            Action::NewSection => {
                let Some(r) = self.selection() else {
                    self.notify("Select scenes first: click one, then Shift+click another to select a range.");
                    return;
                };
                let l = self.loaded.as_ref().unwrap();
                let mut sections = self.edits.sections.clone();
                match sections.add(l.scenes[*r.start()].start..l.scenes[*r.end()].end) {
                    Ok(id) => {
                        self.edit(None, |e| e.sections = sections);
                        self.selected_section = Some(id);
                    }
                    Err(e) => self.notify(e),
                }
            }
            Action::DeleteSection(id) => self.edit(None, |e| e.sections.remove(id)),
            Action::SetSectionRange { id, start, end } => {
                let fc = l.frame_count();
                let Some(section) = self.edits.sections.get(id).filter(|s| s.lock.is_none()) else { return };
                if (section.start, section.end) != (start, end) {
                    self.edits.sections.set_range(id, start..end, fc);
                    self.recompute();
                    self.mark_dirty();
                }
            }
            Action::ToggleLock(id) => {
                let Some(section) = self.edits.sections.get(id) else { return };
                let lock = match section.lock {
                    Some(_) => None,
                    // Freeze what the section shows now: its settings and detected cuts.
                    None => Some(Lock {
                        params: section.overrides.apply(&self.params),
                        cuts: l
                            .detected
                            .iter()
                            .filter(|d| (section.start..section.end).contains(&d.id))
                            .map(|d| (d.id, d.frame.max(0) as usize))
                            .collect(),
                    }),
                };
                self.edit(None, |e| {
                    if let Some(s) = e.sections.get_mut(id) {
                        s.lock = lock;
                    }
                });
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
            self.notify("Nothing selected to export.");
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
                        Ok((info, analysis, cached)) => {
                            self.on_analysis_done(ctx, path, info, analysis);
                            if cached {
                                self.status.push_str(" · analysis from cache");
                            }
                        }
                        Err(e) => self.status = format!("Analysis failed: {e:#}"),
                    }
                }
                Msg::Frames { generation, kind, start, size, frames } if generation == self.generation => {
                    let cache = match kind {
                        FrameKind::Thumb => &mut self.thumbs,
                        FrameKind::Strip => &mut self.strip,
                    };
                    for (i, rgba) in frames.iter().enumerate() {
                        let image = ColorImage::from_rgba_unmultiplied(size, rgba);
                        let name = format!("{}-{}", if kind == FrameKind::Thumb { "thumb" } else { "strip" }, start + i);
                        cache.insert(start + i, ctx.load_texture(name, image, TextureOptions::LINEAR));
                    }
                }
                Msg::AnalysisDone { .. } | Msg::Frames { .. } => {}
                Msg::ExportProgress(done) => {
                    if let Task::Exporting { total, .. } = self.task {
                        self.task = Task::Exporting { done, total };
                    }
                }
                Msg::FfmpegChecked(result) => self.ffmpeg = Some(result.map_err(|e| format!("{e:#}"))),
                Msg::ExportDone(result) => {
                    self.task = Task::Idle;
                    if let Some(l) = &self.loaded {
                        self.status = format!("{}", l.path.file_name().unwrap_or_default().to_string_lossy());
                    }
                    match result {
                        Ok(files) => self.notify(format!("Exported {} files.", files.len())),
                        Err(e) => self.notify(format!("Export failed: {e:#}")),
                    }
                }
            }
        }
    }

    fn keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if self.loaded.is_none() {
            return;
        }
        for action in editor::shortcut_actions(ctx) {
            self.apply(action);
        }
    }

    /// Ask the frame worker for any filmstrip frames in `window` it hasn't been asked for yet.
    fn request_strip_frames(&mut self, window: std::ops::Range<usize>) {
        let Some(l) = &self.loaded else { return };
        let missing: Vec<usize> = window.filter(|f| !self.strip.contains_key(f) && !self.strip_requested.contains(f)).collect();
        if let (Some(&first), Some(&last)) = (missing.first(), missing.last()) {
            self.strip_requested.extend(first..=last);
            let _ = l.frame_tx.send(FrameJob { kind: FrameKind::Strip, start: first, count: last - first + 1 });
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
            match &self.ffmpeg {
                Some(Err(e)) => {
                    ui.label(RichText::new(format!("⚠ {e}")).color(ui.visuals().error_fg_color));
                }
                Some(Ok(version)) => {
                    ui.label(&self.status).on_hover_text(version);
                }
                None => {
                    ui.label(&self.status);
                }
            }
            if let Some((text, at)) = &self.notice {
                if at.elapsed() < NOTICE_DURATION {
                    ui.label(RichText::new(text).color(ui.visuals().warn_fg_color));
                    ui.ctx().request_repaint_after(NOTICE_DURATION);
                }
            }
        });
    }

    /// Which settings the panel edits: the whole video or a section, plus section controls.
    fn section_picker(&self, ui: &mut egui::Ui, section: Option<&Section>, actions: &mut Vec<Action>) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Settings for:");
            if ui.selectable_label(section.is_none(), "Whole video").clicked() {
                actions.push(Action::SelectSection(None));
            }
            for (i, s) in self.edits.sections.list.iter().enumerate() {
                let text = format!("{}Section {}", if s.lock.is_some() { "🔒 " } else { "" }, i + 1);
                let selected = section.is_some_and(|x| x.id == s.id);
                if ui.selectable_label(selected, RichText::new(text).color(editor::SECTION_COLOR)).clicked() {
                    actions.push(Action::SelectSection(Some(s.id)));
                }
            }
            ui.separator();
            if ui
                .add_enabled(self.selection().is_some(), egui::Button::new("➕ New section"))
                .on_hover_text("Give the selected scenes their own settings.\nShift+click scenes to select several.")
                .clicked()
            {
                actions.push(Action::NewSection);
            }
            let Some(s) = section else {
                if !self.edits.sections.list.is_empty() {
                    ui.label(RichText::new("Sections keep the settings they change.").weak());
                }
                return;
            };
            let (text, hint) = match s.lock {
                Some(_) => ("🔓 Unlock", "Let setting changes affect this section again."),
                None => ("🔒 Lock", "Keep this section's cuts and settings as they are now,\nwhatever you change elsewhere."),
            };
            if ui.button(text).on_hover_text(hint).clicked() {
                actions.push(Action::ToggleLock(s.id));
            }
            if ui.button("🗑 Remove section").on_hover_text("This part goes back to the whole-video settings.").clicked() {
                actions.push(Action::DeleteSection(s.id));
            }
            let fps = self.loaded.as_ref().map_or(1.0, |l| l.analysis.fps);
            let state = match (&s.lock, s.overrides.count()) {
                (Some(_), _) => "locked: settings and detected cuts are frozen (cuts can still be edited by hand)".to_owned(),
                (None, 0) => "same as the whole video until you change a setting".to_owned(),
                (None, n) => format!("{n} settings changed, the rest follow the whole video"),
            };
            ui.label(RichText::new(format!("{} – {} · {state}", fmt_time(s.start as f64 / fps), fmt_time(s.end as f64 / fps))).weak());
        });
    }

    fn settings(&mut self, ui: &mut egui::Ui) -> Vec<Action> {
        let mut actions = Vec::new();
        let section = self.selected_section.and_then(|id| self.edits.sections.get(id)).cloned();
        self.section_picker(ui, section.as_ref(), &mut actions);

        let locked = section.as_ref().is_some_and(|s| s.lock.is_some());
        let before = match &section {
            Some(Section { lock: Some(lock), .. }) => lock.params.clone(),
            Some(s) => s.overrides.apply(&self.params),
            None => self.params.clone(),
        };
        let mut p = before.clone();
        let mut cleared = Vec::new();
        // Settings the section overrides are highlighted, with a button to go back to the whole-video value.
        let overridden = |name: &str| section.as_ref().is_some_and(|s| !locked && s.overrides.is_set(name));
        let mut label = |ui: &mut egui::Ui, name: &'static str, text: &str| -> egui::Response {
            if !overridden(name) {
                return ui.label(text);
            }
            ui.horizontal(|ui| {
                if ui.small_button("↺").on_hover_text("Use the whole-video setting").clicked() {
                    cleared.push(name);
                }
                ui.label(RichText::new(text).color(editor::SECTION_COLOR));
            })
            .response
        };

        ui.add_enabled_ui(!locked, |ui| {
            ui.horizontal(|ui| {
                label(ui, "mode", "Cut detection:");
                ui.radio_value(&mut p.mode, DetectMode::Adaptive, "Adaptive");
                ui.radio_value(&mut p.mode, DetectMode::Fixed, "Fixed threshold");
            });
            egui::Grid::new("settings").num_columns(4).spacing([16.0, 4.0]).show(ui, |ui| {
                match p.mode {
                    DetectMode::Fixed => {
                        label(ui, "cut_threshold", "Cut threshold");
                        ui.add(egui::Slider::new(&mut p.cut_threshold, 1.0..=120.0));
                    }
                    DetectMode::Adaptive => {
                        label(ui, "adaptive_ratio", "Sensitivity ratio");
                        ui.add(egui::Slider::new(&mut p.adaptive_ratio, 1.2..=10.0));
                    }
                }
                label(ui, "min_scene_secs", "Min scene length (s)");
                ui.add(egui::Slider::new(&mut p.min_scene_secs, 0.0..=5.0));
                ui.end_row();

                if p.mode == DetectMode::Adaptive {
                    label(ui, "adaptive_floor", "Ignore changes below");
                    ui.add(egui::Slider::new(&mut p.adaptive_floor, 0.0..=60.0));
                } else {
                    ui.label("");
                    ui.label("");
                }
                label(ui, "still_threshold", "Still if motion below")
                    .on_hover_text("Scenes whose median frame-to-frame change is below this are exported as a single image.");
                ui.add(egui::Slider::new(&mut p.still_threshold, 0.0..=10.0));
                ui.end_row();

                label(ui, "cut_offset", "Cut offset (frames)").on_hover_text(
                    "Moves every detected cut relative to the detected change.\n\
                     0: the new scene starts on the first changed frame.\n\
                     Negative: earlier. Positive: later.\n\
                     Cuts you moved or added by hand aren't affected.",
                );
                ui.add(egui::DragValue::new(&mut p.cut_offset).range(-60..=60).speed(0.1));
                // One label for both values: overriding either highlights it.
                let name = if overridden("drop_after_cut") { "drop_after_cut" } else { "drop_before_cut" };
                label(ui, name, "Drop frames before / after cut").on_hover_text(
                    "Frames removed from the end of the outgoing scene and the start of the incoming one, \
                     e.g. to skip transition or blended frames. Not applied at the start or end of the video.",
                );
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut p.drop_before_cut).range(0..=600).speed(0.1));
                    ui.label("/");
                    ui.add(egui::DragValue::new(&mut p.drop_after_cut).range(0..=600).speed(0.1));
                });
                ui.end_row();
            });
        });
        // The drop label resets both values.
        if cleared.contains(&"drop_before_cut") || cleared.contains(&"drop_after_cut") {
            cleared.extend(["drop_before_cut", "drop_after_cut"]);
        }

        match section {
            None if p != before => {
                self.params = p;
                self.recompute();
                self.mark_dirty();
            }
            Some(s) if !locked => {
                let mut overrides = s.overrides.clone();
                let changed = overrides.record(&before, &p);
                for name in cleared {
                    overrides.clear(name);
                }
                if overrides != s.overrides {
                    // Dragging one slider is one undo step.
                    let key = changed.map(|name| format!("section {} {name}", s.id));
                    self.edit(key, |e| {
                        if let Some(x) = e.sections.get_mut(s.id) {
                            x.overrides = overrides;
                        }
                    });
                }
            }
            _ => {}
        }
        actions
    }

    fn cut_editor(&mut self, ui: &mut egui::Ui) -> Vec<Action> {
        let Some(l) = &self.loaded else { return Vec::new() };
        let input = editor::GraphInput {
            diffs: &l.analysis.diffs,
            cuts: &l.cuts,
            frame_count: l.frame_count(),
            fps: l.analysis.fps,
            playhead: l.player.position(),
            selected_cut: self.selected_cut,
            thresholds: l
                .spans
                .iter()
                .map(|s| {
                    let t = match s.params.mode {
                        DetectMode::Fixed => s.params.cut_threshold,
                        DetectMode::Adaptive => s.params.adaptive_floor,
                    };
                    (s.range.clone(), s.frozen.is_none().then_some(t))
                })
                .collect(),
            sections: l
                .spans
                .iter()
                .filter_map(|s| Some((s.range.clone(), s.section? == self.selected_section.unwrap_or(u32::MAX))))
                .collect(),
        };
        let mut actions = editor::graph(ui, &mut self.view, &input);

        let toolbar = editor::ToolbarInput {
            selected_cut: self.selected_cut.and_then(|id| l.cuts.get(id)).copied(),
            has_selected_scene: self.selected.is_some(),
            has_cut_edits: !self.edits.cuts.is_empty(),
            can_undo: self.history.can_undo(),
            can_redo: self.history.can_redo(),
        };
        actions.extend(editor::toolbar(ui, &mut self.view, &toolbar));

        let n = l.scenes.len();
        let stills = l.scenes.iter().filter(|s| self.effective_kind(s) == SceneKind::Still).count();
        let empty = l.scenes.iter().filter(|s| s.keep.is_empty()).count();
        let edited = l.cuts.cuts.iter().filter(|c| c.source != CutSource::Detected).count() + l.cuts.removed.len();
        ui.horizontal(|ui| {
            ui.label(format!("{n} scenes · {stills} stills · {} clips", n - stills));
            if edited > 0 {
                ui.label(RichText::new(format!("· {edited} cut edits")).weak());
            }
            if empty > 0 {
                ui.label(RichText::new(format!("· {empty} trimmed to nothing")).color(ui.visuals().warn_fg_color));
            }
        });
        actions
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

    fn player_panel(&mut self, ui: &mut egui::Ui) -> Vec<Action> {
        let Some(l) = &self.loaded else {
            ui.centered_and_justified(|ui| ui.label(RichText::new("No video loaded").weak()));
            return Vec::new();
        };
        let mut actions = Vec::new();
        let player = &l.player;

        let size = vec2(ui.available_width(), ui.available_width() / player.aspect());
        match player.texture() {
            Some(tex) => {
                let resp = ui.add(egui::Image::from_texture((tex.id(), size)).sense(Sense::click()));
                if resp.clicked() {
                    actions.push(Action::TogglePlay);
                }
            }
            None => {
                let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
                ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                ui.put(rect, egui::Spinner::new());
            }
        }

        let snap_to: Vec<usize> = l.cuts.cuts.iter().map(|c| c.frame).collect();
        let bar = editor::SectionsInput {
            sections: &self.edits.sections,
            selected: self.selected_section,
            frame_count: l.frame_count(),
            fps: l.analysis.fps,
            snap_to: &snap_to,
        };
        actions.extend(editor::sections_bar(ui, &bar, &mut self.view));

        let overview = editor::OverviewInput {
            scenes: &l.scenes,
            edits: &self.edits,
            selected: self.selection(),
            frame_count: l.player.frame_count(),
            fps: l.analysis.fps,
            shown_frame: l.player.current_frame(),
            position: l.player.position(),
        };
        actions.extend(editor::overview(ui, &overview, &self.view));

        let current = player.current_frame();
        let here = l.scene_at(current);
        ui.horizontal_wrapped(|ui| {
            if ui.button("⏮").on_hover_text("Previous scene").clicked() {
                // Restart the current scene unless we're already at its beginning.
                let at_start = l.scenes.get(here).is_none_or(|s| current <= s.keep.start + 2);
                let index = if at_start { here.saturating_sub(1) } else { here };
                actions.push(Action::SelectScene { index, play: player.is_playing() });
            }
            if ui.button("◀").on_hover_text("Previous frame (←)").clicked() {
                actions.push(Action::Step(-1));
            }
            let play_label = if player.is_playing() { "⏸" } else { "▶" };
            if ui.button(play_label).on_hover_text("Play / pause (Space)").clicked() {
                actions.push(Action::TogglePlay);
            }
            if ui.button("▶|").on_hover_text("Next frame (→)").clicked() {
                actions.push(Action::Step(1));
            }
            if ui.button("⏭").on_hover_text("Next scene").clicked() && here + 1 < l.scenes.len() {
                actions.push(Action::SelectScene { index: here + 1, play: player.is_playing() });
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
        actions
    }

    fn filmstrip(&mut self, ui: &mut egui::Ui) -> Vec<Action> {
        let Some(l) = &self.loaded else { return Vec::new() };
        let Some(cut) = self.selected_cut.and_then(|id| l.cuts.get(id)).copied() else { return Vec::new() };
        let input = editor::StripInput {
            cut,
            frame_count: l.frame_count(),
            fps: l.analysis.fps,
            movable: l.cuts.movable_range(cut.id, l.frame_count()),
            frames: &self.strip,
            aspect: l.player.aspect(),
        };
        let (actions, window) = editor::filmstrip(ui, &input);
        ui.separator();
        self.request_strip_frames(window);
        actions
    }

    fn scene_list(&mut self, ui: &mut egui::Ui) -> Vec<Action> {
        let Some(l) = &self.loaded else {
            ui.centered_and_justified(|ui| ui.label(RichText::new("No video loaded").weak()));
            return Vec::new();
        };
        let thumb_h = scaled_height(&l.info, THUMB_W).min(ROW_SEPARATOR_AT as u32);
        let n = l.scenes.len();
        let mut actions = Vec::new();

        // Every row is exactly ROW_HEIGHT tall (its separator included), so a row's scroll
        // offset is exact and selecting a scene elsewhere can bring it into view.
        let pitch = ROW_HEIGHT + ui.spacing().item_spacing.y;
        let selection = self.selection();
        let mut area = egui::ScrollArea::vertical().auto_shrink(false);
        if let Some(i) = self.scroll_to_scene.take().filter(|i| !self.visible_rows.contains(i)) {
            area = area.vertical_scroll_offset((i as f32 - 1.0).max(0.0) * pitch);
        }
        let mut visible = 0..0;
        area.show_rows(ui, ROW_HEIGHT, n, |ui, range| {
            // `range` includes partly visible rows at either end.
            visible = range.start + 1..range.end.saturating_sub(1);
            for i in range {
                let scene = &l.scenes[i];
                let thumb_frame = scene.middle();
                if self.thumbs_requested.insert(thumb_frame) {
                    let _ = l.frame_tx.send(FrameJob { kind: FrameKind::Thumb, start: thumb_frame, count: 1 });
                }
                let is_selected = selection.as_ref().is_some_and(|r| r.contains(&i));
                let background = ui.painter().add(Shape::Noop);

                let row = ui.horizontal(|ui| {
                    ui.set_height(ROW_SEPARATOR_AT);
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
                    if thumb.on_hover_text("Click to preview, Shift+click to select a range").clicked() {
                        actions.push(if ui.input(|i| i.modifiers.shift) {
                            Action::ExtendSelection(i)
                        } else {
                            Action::SelectScene { index: i, play: true }
                        });
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
                            match l.cuts.get(scene.id).map(|c| c.source) {
                                Some(CutSource::Moved) => {
                                    ui.label(RichText::new("✎ moved cut").small().color(editor::MOVED_COLOR));
                                }
                                Some(CutSource::Manual) => {
                                    ui.label(RichText::new("✂ manual cut").small().color(editor::MANUAL_COLOR));
                                }
                                _ => {}
                            }
                            let section = l.spans.first().and_then(|_| sections::span_at(&l.spans, scene.start).section);
                            if let Some(n) = section.and_then(|id| self.edits.sections.number(id)) {
                                ui.label(RichText::new(format!("section {n}")).small().color(editor::SECTION_COLOR));
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
                            let mut include = !self.edits.excluded.contains(&scene.id);
                            let checkbox = ui.add_enabled(!scene.keep.is_empty(), egui::Checkbox::new(&mut include, "Export"));
                            if checkbox.changed() {
                                actions.push(Action::SetExcluded(scene.id, !include));
                            }
                            // Field access rather than `effective_kind` keeps this closure's borrows disjoint.
                            let current = self.edits.kinds.get(&scene.id).copied().unwrap_or(scene.kind);
                            let mut kind = current;
                            ui.selectable_value(&mut kind, SceneKind::Video, "🎞 Video");
                            ui.selectable_value(&mut kind, SceneKind::Still, "🖼 Still");
                            if kind != current {
                                actions.push(Action::SetKind(scene.id, kind));
                            }
                            if ui.button("▶ Preview").clicked() {
                                actions.push(Action::SelectScene { index: i, play: true });
                            }
                            if i + 1 < n && ui.button("Merge with next").on_hover_text("Delete the cut between this scene and the next").clicked() {
                                actions.push(Action::DeleteCut(l.scenes[i + 1].id));
                            }
                        });
                    });
                });

                if is_selected {
                    let rect = row.response.rect.expand2(vec2(4.0, 2.0));
                    let fill = ui.visuals().selection.bg_fill.gamma_multiply(0.35);
                    ui.painter().set(background, Shape::rect_filled(rect, 4.0, fill));
                }
                // The rest of the row's height, with the separator line in it.
                let rest = ROW_HEIGHT - ROW_SEPARATOR_AT - ui.spacing().item_spacing.y;
                let (line, _) = ui.allocate_exact_size(vec2(ui.available_width(), rest), Sense::hover());
                ui.painter().hline(line.x_range(), line.center().y, ui.visuals().widgets.noninteractive.bg_stroke);
            }
        });
        self.visible_rows = visible;
        actions
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
        if let Some(since) = self.dirty_since {
            let wait = SAVE_DELAY.saturating_sub(since.elapsed());
            if wait.is_zero() {
                self.save_project();
            } else {
                ctx.request_repaint_after(wait);
            }
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
            actions.extend(self.cut_editor(ui));
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
        egui::CentralPanel::default().show(ui, |ui| {
            actions.extend(self.filmstrip(ui));
            actions.extend(self.scene_list(ui));
        });

        for action in actions {
            self.apply(action);
        }
    }

    fn on_exit(&mut self) {
        if self.dirty_since.is_some() {
            self.save_project();
        }
    }
}

/// Background thread that decodes thumbnails and filmstrip frames on request. Exits when the
/// sender is dropped (i.e. when another video is opened).
fn spawn_frame_worker(
    ctx: egui::Context,
    tx: Sender<Msg>,
    generation: u64,
    path: PathBuf,
    info: &VideoInfo,
    fps: f64,
) -> Sender<FrameJob> {
    let (req_tx, req_rx) = channel::<FrameJob>();
    let thumb = (THUMB_W, scaled_height(info, THUMB_W).min(ROW_SEPARATOR_AT as u32));
    let strip = (STRIP_W, scaled_height(info, STRIP_W));
    std::thread::spawn(move || {
        let mut queue = Vec::new();
        while let Ok(first) = req_rx.recv() {
            queue.push(first);
            // Serve the most recent requests first: they're what's on screen now.
            loop {
                queue.extend(req_rx.try_iter());
                let Some(job) = queue.pop() else { break };
                let (w, h) = if job.kind == FrameKind::Thumb { thumb } else { strip };
                let Ok(frames) = ffmpeg::grab_frames_rgba(&path, job.start, job.count, fps, w, h) else { continue };
                let msg = Msg::Frames { generation, kind: job.kind, start: job.start, size: [w as usize, h as usize], frames };
                if tx.send(msg).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
        }
    });
    req_tx
}

/// Height for `width` that keeps the video's aspect ratio; even, which keeps ffmpeg's scaler happy.
fn scaled_height(info: &VideoInfo, width: u32) -> u32 {
    if info.width == 0 || info.height == 0 {
        return width * 9 / 16 / 2 * 2;
    }
    ((width as f64 * info.height as f64 / info.width as f64).round() as u32 / 2 * 2).max(2)
}

fn default_out_dir(input: &Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    input.parent().unwrap_or(Path::new(".")).join(format!("{stem}_scenes"))
}

/// Hooks for the UI tests in `ui_tests.rs`.
#[cfg(test)]
impl App {
    pub(crate) fn open_for_test(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.open(ctx, path);
    }

    pub(crate) fn scene_count(&self) -> Option<usize> {
        self.loaded.as_ref().map(|l| l.scenes.len())
    }

    pub(crate) fn cut_frames(&self) -> Vec<usize> {
        self.loaded.as_ref().map_or(Vec::new(), |l| l.cuts.cuts.iter().map(|c| c.frame).collect())
    }

    /// (start, end, locked) of each section.
    pub(crate) fn section_ranges(&self) -> Vec<(usize, usize, bool)> {
        self.edits.sections.list.iter().map(|s| (s.start, s.end, s.lock.is_some())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut_frames(app: &App) -> Vec<(usize, CutSource)> {
        app.loaded.as_ref().unwrap().cuts.cuts.iter().map(|c| (c.frame, c.source)).collect()
    }

    fn position(app: &App) -> usize {
        app.loaded.as_ref().unwrap().player.position()
    }

    /// Drives the editing actions against a real (synthetic) video: cuts detected at 75 and 150.
    #[test]
    #[ignore = "needs ffmpeg on PATH"]
    fn cut_editing_end_to_end() {
        let input = ffmpeg::make_test_video("app");
        let ctx = egui::Context::default();
        let mut app = App::empty();
        open_and_wait(&ctx, &mut app, &input);
        assert_eq!(cut_frames(&app), [(75, CutSource::Detected), (150, CutSource::Detected)]);

        // Jump to the next cut from the start, nudge it twice (one undo step), then back.
        app.apply(Action::JumpCut(1));
        assert_eq!(app.selected_cut, Some(CutId::Detected(75)));
        assert_eq!(position(&app), 75);
        app.apply(Action::Nudge(1));
        app.apply(Action::Nudge(1));
        assert_eq!(cut_frames(&app)[0], (77, CutSource::Moved));
        assert_eq!(position(&app), 77, "player previews the new first frame");
        app.apply(Action::Nudge(-2));
        assert_eq!(cut_frames(&app)[0], (75, CutSource::Detected), "back on the detected frame = not an edit");
        app.apply(Action::Undo);
        assert_eq!(cut_frames(&app)[0], (75, CutSource::Detected), "all three nudges were one undo step");

        // A drag: one checkpoint, many moves, one undo step. Can't cross the neighbouring cut.
        app.apply(Action::Checkpoint);
        for frame in [90, 120, 400] {
            app.apply(Action::MoveCut { id: CutId::Detected(75), frame });
        }
        assert_eq!(cut_frames(&app)[0].0, 149);
        app.apply(Action::Undo);
        assert_eq!(cut_frames(&app)[0], (75, CutSource::Detected));

        // Split at the playhead, refuse a duplicate, then delete the new cut.
        app.apply(Action::Seek(100));
        app.apply(Action::Split);
        assert_eq!(cut_frames(&app), [(75, CutSource::Detected), (100, CutSource::Manual), (150, CutSource::Detected)]);
        let manual = app.selected_cut.unwrap();
        app.apply(Action::SplitAt(100));
        assert_eq!(cut_frames(&app).len(), 3);
        assert!(app.notice.is_some());
        app.apply(Action::DeleteSelected);
        assert_eq!(cut_frames(&app).len(), 2);
        assert_eq!(app.selected_cut, None);
        app.apply(Action::Undo);
        assert!(app.loaded.as_ref().unwrap().cuts.get(manual).is_some());
        app.apply(Action::Redo);
        assert_eq!(cut_frames(&app).len(), 2);

        // Offset only shifts detected cuts; manual cuts stay put.
        app.apply(Action::SplitAt(100));
        app.params.cut_offset = 2;
        app.recompute();
        assert_eq!(cut_frames(&app), [(77, CutSource::Detected), (100, CutSource::Manual), (152, CutSource::Detected)]);

        // Selecting from the timeline picks the clip under the click and seeks to that frame.
        let index = app.loaded.as_ref().unwrap().scene_at(120);
        app.apply(Action::SelectSceneAt { index, frame: 120 });
        let scene = app.loaded.as_ref().unwrap().scenes[index].clone();
        assert!((scene.start..scene.end).contains(&120));
        assert_eq!(app.selected, Some(scene.id));
        assert_eq!(position(&app), 120);
        assert_eq!(app.scroll_to_scene, Some(index));

        // Edits are saved next to the video and restored on reopen.
        app.save_project();
        let saved = project::load(&input).unwrap().expect("sidecar written");
        assert_eq!(saved.params.cut_offset, 2);
        assert_eq!(saved.edits, app.edits);
        let mut reopened = App::empty();
        open_and_wait(&ctx, &mut reopened, &input);
        assert_eq!(reopened.edits, app.edits);
        assert_eq!(reopened.params, app.params);
        // The second open skips decoding and gets the same scores.
        assert!(reopened.status.contains("analysis from cache"), "{}", reopened.status);
        assert_eq!(reopened.loaded.as_ref().unwrap().analysis.diffs, app.loaded.as_ref().unwrap().analysis.diffs);
    }

    /// Sections against a real video (cuts at 75 and 150): their own settings, locking, undo.
    #[test]
    #[ignore = "needs ffmpeg on PATH"]
    fn sections_end_to_end() {
        let input = ffmpeg::make_test_video("sections");
        let _ = std::fs::remove_file(project::sidecar_path(&input));
        let ctx = egui::Context::default();
        let mut app = App::empty();
        open_and_wait(&ctx, &mut app, &input);

        app.apply(Action::NewSection);
        assert!(app.edits.sections.list.is_empty() && app.notice.is_some(), "needs selected scenes");

        // Shift+click selects scenes 1..=2; a section over them.
        app.apply(Action::SelectScene { index: 1, play: false });
        app.apply(Action::ExtendSelection(2));
        assert_eq!(app.selection(), Some(1..=2));
        app.apply(Action::NewSection);
        let id = app.selected_section.expect("new section is selected for editing");
        assert_eq!(app.section_ranges(), [(75, 225, false)]);
        app.apply(Action::NewSection);
        assert_eq!(app.edits.sections.list.len(), 1, "overlapping sections are refused");

        // Its own offset moves only its cuts; the whole-video offset still applies elsewhere.
        app.edit(None, |e| e.sections.get_mut(id).unwrap().overrides.cut_offset = Some(3));
        assert_eq!(cut_frames(&app), [(78, CutSource::Detected), (153, CutSource::Detected)]);
        app.apply(Action::SetSectionRange { id, start: 100, end: 225 });
        assert_eq!(cut_frames(&app), [(75, CutSource::Detected), (153, CutSource::Detected)]);

        // Locked, it ignores whole-video changes that remove every other cut.
        app.apply(Action::ToggleLock(id));
        app.params = Params { mode: DetectMode::Fixed, cut_threshold: 255.0, ..app.params.clone() };
        app.recompute();
        assert_eq!(cut_frames(&app), [(153, CutSource::Detected)]);
        app.apply(Action::SetSectionRange { id, start: 0, end: 225 });
        assert_eq!(app.section_ranges(), [(100, 225, true)], "locked sections can't be resized");

        // Unlocking lets the change through; undo locks it again.
        app.apply(Action::ToggleLock(id));
        assert_eq!(cut_frames(&app), []);
        app.apply(Action::Undo);
        assert_eq!(cut_frames(&app), [(153, CutSource::Detected)]);

        // Saved and restored with the project.
        app.save_project();
        let mut reopened = App::empty();
        open_and_wait(&ctx, &mut reopened, &input);
        assert_eq!(reopened.edits.sections, app.edits.sections);
        assert_eq!(cut_frames(&reopened), [(153, CutSource::Detected)]);
        let _ = std::fs::remove_file(project::sidecar_path(&input));
    }

    fn open_and_wait(ctx: &egui::Context, app: &mut App, path: &Path) {
        app.open(ctx, path.to_path_buf());
        let deadline = Instant::now() + Duration::from_secs(20);
        while app.loaded.is_none() {
            assert!(Instant::now() < deadline, "analysis timed out: {}", app.status);
            app.handle_messages(ctx);
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
