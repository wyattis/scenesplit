//! Cut editing widgets: the zoomable difference graph with draggable cut markers, the
//! editing toolbar, and the filmstrip around the selected cut.
//!
//! Widgets don't change app state directly; they return [`Action`]s that the app applies.

use std::collections::HashMap;
use std::ops::Range;

use eframe::egui::{
    self, Color32, CursorIcon, Event, FontId, Key, Modifiers, Rect, RichText, Sense, Shape, Stroke, TextureHandle,
    WidgetInfo, WidgetType, pos2, vec2,
};

use crate::project::Edits;
use crate::scenes::{Cut, CutId, CutSource, ResolvedCuts, Scene, SceneKind};

pub const DETECTED_COLOR: Color32 = Color32::from_rgb(230, 120, 40);
pub const MOVED_COLOR: Color32 = Color32::from_rgb(240, 200, 60);
pub const MANUAL_COLOR: Color32 = Color32::from_rgb(190, 110, 230);
const SELECTED_COLOR: Color32 = Color32::from_rgb(255, 255, 255);

/// Frames shown on each side of the selected cut in the filmstrip.
pub const STRIP_RADIUS: usize = 6;
/// Smallest zoom window, in frames.
const MIN_SPAN: f64 = 12.0;
/// How close (in pixels) the pointer must be to grab a cut marker.
const GRAB_PX: f32 = 6.0;
/// Snapping looks for a spike within this many pixels.
const SNAP_PX: f32 = 10.0;

pub const SHORTCUTS: &str = "Space  play / pause\n\
    ← →  step one frame (Shift: 10)\n\
    S  split at playhead\n\
    Delete  remove selected cut\n\
    , .  nudge selected cut (Shift: 10)\n\
    [ ]  jump to previous / next cut\n\
    Esc  deselect cut\n\
    Ctrl+Z / Ctrl+Shift+Z  undo / redo\n\
    \n\
    Graph: drag a marker to move it (Alt: no snapping),\n\
    double-click to add a cut, right-click for more,\n\
    Ctrl+scroll to zoom, scroll to pan.";

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Seek(usize),
    Step(i64),
    TogglePlay,
    /// Select scene by index and jump to it, optionally starting playback.
    SelectScene { index: usize, play: bool },
    /// Select scene by index and move the playhead to `frame` inside it.
    SelectSceneAt { index: usize, frame: usize },
    SelectCut(Option<CutId>),
    /// Record an undo step before an edit gesture (e.g. at the start of a drag).
    Checkpoint,
    /// Move a cut (clamped between its neighbours). Doesn't record an undo step by itself.
    MoveCut { id: CutId, frame: usize },
    SplitAt(usize),
    /// Split at the playhead.
    Split,
    DeleteCut(CutId),
    /// Delete the selected cut.
    DeleteSelected,
    ResetCut(CutId),
    ResetAllCuts,
    /// Move the selected cut (or the one nearest the playhead) by this many frames.
    Nudge(i64),
    /// Jump to the previous (-1) or next (+1) cut.
    JumpCut(i32),
    Undo,
    Redo,
    ZoomToSelectedScene,
    SetKind(CutId, SceneKind),
    SetExcluded(CutId, bool),
}

/// Actions for this frame's key presses. Nothing while a widget (e.g. a number field) has
/// keyboard focus, so typing there doesn't trigger shortcuts.
pub fn shortcut_actions(ctx: &egui::Context) -> Vec<Action> {
    if ctx.memory(|m| m.focused().is_some()) {
        return Vec::new();
    }
    ctx.input(|i| {
        i.events
            .iter()
            .filter_map(|e| match e {
                Event::Key { key, pressed: true, modifiers, .. } => shortcut(*key, *modifiers),
                _ => None,
            })
            .collect()
    })
}

fn shortcut(key: Key, m: Modifiers) -> Option<Action> {
    let step = if m.shift { 10 } else { 1 };
    Some(match key {
        Key::Z if m.command && m.shift => Action::Redo,
        Key::Z if m.command => Action::Undo,
        Key::Y if m.command => Action::Redo,
        _ if m.command || m.alt => return None,
        Key::Space => Action::TogglePlay,
        Key::ArrowLeft => Action::Step(-step),
        Key::ArrowRight => Action::Step(step),
        Key::S => Action::Split,
        Key::Delete | Key::Backspace => Action::DeleteSelected,
        Key::Comma => Action::Nudge(-step),
        Key::Period => Action::Nudge(step),
        Key::OpenBracket => Action::JumpCut(-1),
        Key::CloseBracket => Action::JumpCut(1),
        Key::Escape => Action::SelectCut(None),
        _ => return None,
    })
}

/// View state of the cut editor that persists between frames.
pub struct EditorView {
    /// Visible frame range of the graph.
    pub start: f64,
    pub end: f64,
    frame_count: f64,
    dragging: Option<CutId>,
    context: Option<ContextTarget>,
    last_playhead: usize,
}

#[derive(Clone, Copy)]
enum ContextTarget {
    Cut(CutId),
    Frame(usize),
}

impl EditorView {
    pub fn new(frame_count: usize) -> Self {
        let fc = frame_count.max(1) as f64;
        Self { start: 0.0, end: fc, frame_count: fc, dragging: None, context: None, last_playhead: 0 }
    }

    fn span(&self) -> f64 {
        self.end - self.start
    }

    pub fn is_zoomed(&self) -> bool {
        self.span() < self.frame_count - 0.5
    }

    pub fn fit(&mut self) {
        self.set(0.0, self.frame_count);
    }

    /// Show `range` with a little margin.
    pub fn show_range(&mut self, range: Range<usize>) {
        let pad = (range.len() as f64 * 0.1).max(2.0);
        self.set(range.start as f64 - pad, range.end as f64 + pad);
    }

    /// Zoom by `factor` (> 1 zooms in) keeping frame `anchor` under the same pixel.
    pub fn zoom(&mut self, factor: f64, anchor: f64) {
        let span = (self.span() / factor).clamp(MIN_SPAN.min(self.frame_count), self.frame_count);
        let t = (anchor - self.start) / self.span();
        self.set(anchor - t * span, anchor - t * span + span);
    }

    fn pan(&mut self, frames: f64) {
        self.set(self.start + frames, self.end + frames);
    }

    fn set(&mut self, start: f64, end: f64) {
        let span = (end - start).clamp(MIN_SPAN.min(self.frame_count), self.frame_count);
        let start = start.clamp(0.0, self.frame_count - span);
        self.start = start;
        self.end = start + span;
    }

    /// Keep the playhead visible when it moves off-screen (playback, jumps, seeks).
    fn follow(&mut self, playhead: usize) {
        if playhead != self.last_playhead {
            let p = playhead as f64;
            if p < self.start || p >= self.end {
                let span = self.span();
                self.set(p - span * 0.1, p - span * 0.1 + span);
            }
            self.last_playhead = playhead;
        }
    }
}

pub struct GraphInput<'a> {
    pub diffs: &'a [f32],
    pub cuts: &'a ResolvedCuts,
    pub frame_count: usize,
    pub fps: f64,
    pub playhead: usize,
    pub selected_cut: Option<CutId>,
    /// Horizontal reference line (the active detection threshold).
    pub threshold: f32,
}

pub fn cut_color(source: CutSource) -> Color32 {
    match source {
        CutSource::Detected => DETECTED_COLOR,
        CutSource::Moved => MOVED_COLOR,
        CutSource::Manual => MANUAL_COLOR,
    }
}

/// The difference graph, which is also the main cut editing surface.
pub fn graph(ui: &mut egui::Ui, view: &mut EditorView, input: &GraphInput<'_>) -> Vec<Action> {
    let mut actions = Vec::new();
    view.follow(input.playhead);

    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 90.0), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    if input.frame_count < 2 {
        return actions;
    }

    let span = view.span();
    let x_of = |f: f64| rect.left() + ((f - view.start) / span) as f32 * rect.width();
    let f_of = |x: f32| view.start + ((x - rect.left()) / rect.width()) as f64 * span;
    let px_per_frame = rect.width() as f64 / span;
    let diff_at = |boundary: usize| -> f32 {
        // The spike that suggests a cut at `boundary` compares frames boundary-1 and boundary.
        boundary.checked_sub(1).and_then(|i| input.diffs.get(i)).copied().unwrap_or(0.0)
    };

    // Difference bars: one per pixel column, showing the largest difference in that column.
    let y_max = input.diffs.iter().copied().fold(0.0f32, f32::max).clamp(10.0, 120.0);
    let to_y = |v: f32| rect.bottom() - (v / y_max).min(1.0) * (rect.height() - 10.0);
    let bar = Stroke::new(1.0, ui.visuals().text_color().gamma_multiply(0.55));
    for c in 0..rect.width().max(1.0) as usize {
        let x = rect.left() + c as f32;
        let b0 = f_of(x).round().max(0.0) as usize;
        let b1 = (f_of(x + 1.0).round().max(0.0) as usize).max(b0 + 1);
        let v = (b0..b1).map(diff_at).fold(0.0f32, f32::max);
        if v > 0.0 {
            painter.vline(x + 0.5, to_y(v)..=rect.bottom(), bar);
        }
    }

    // Per-frame ticks once zoomed in far enough to tell frames apart.
    if px_per_frame >= 6.0 {
        let tick = Stroke::new(1.0, ui.visuals().weak_text_color().gamma_multiply(0.5));
        for f in view.start.ceil() as usize..=view.end.floor() as usize {
            painter.vline(x_of(f as f64), rect.bottom() - 4.0..=rect.bottom(), tick);
        }
    }

    painter.hline(rect.x_range(), to_y(input.threshold), Stroke::new(1.0, Color32::from_rgb(70, 130, 200)));

    // Deleted detected cuts and the original positions of moved cuts, dashed.
    let dashed = |f: usize, color: Color32| {
        let x = x_of(f as f64);
        painter.extend(Shape::dashed_line(&[pos2(x, rect.top()), pos2(x, rect.bottom())], Stroke::new(1.0, color), 3.0, 3.0));
    };
    for &f in &input.cuts.removed {
        dashed(f, Color32::GRAY.gamma_multiply(0.6));
    }
    for cut in &input.cuts.cuts {
        if let Some(g) = cut.ghost.filter(|_| input.selected_cut == Some(cut.id)) {
            dashed(g, DETECTED_COLOR.gamma_multiply(0.6));
        }
    }

    // Cut markers, with a grab handle on top.
    let pointer = response.hover_pos();
    let near = |x: f32| nearest_cut(&input.cuts.cuts, |c| (x_of(c.frame as f64) - x).abs()).filter(|(_, d)| *d <= GRAB_PX);
    let hovered_cut = pointer.and_then(|p| near(p.x)).map(|(c, _)| c);
    for cut in &input.cuts.cuts {
        let x = x_of(cut.frame as f64);
        if !rect.x_range().contains(x) {
            continue;
        }
        let selected = input.selected_cut == Some(cut.id);
        let hovered = hovered_cut.is_some_and(|h| h.id == cut.id) || view.dragging == Some(cut.id);
        let color = if selected { SELECTED_COLOR } else { cut_color(cut.source) };
        let width = if selected || hovered { 2.5 } else { 1.5 };
        painter.vline(x, rect.y_range(), Stroke::new(width, color.gamma_multiply(if selected { 1.0 } else { 0.85 })));
        let h = if selected || hovered { 8.0 } else { 6.0 };
        painter.add(Shape::convex_polygon(
            vec![pos2(x - h, rect.top()), pos2(x + h, rect.top()), pos2(x, rect.top() + h * 1.2)],
            cut_color(cut.source),
            Stroke::new(1.0, if selected { SELECTED_COLOR } else { Color32::TRANSPARENT }),
        ));
    }

    // Playhead.
    painter.vline(x_of(input.playhead as f64), rect.y_range(), Stroke::new(1.5, ui.visuals().strong_text_color()));

    // Hover readout.
    if let Some(p) = pointer {
        let b = f_of(p.x).round().max(0.0) as usize;
        let text = match hovered_cut {
            Some(c) => format!("cut @ {}  #{}  ({})", fmt_time(c.frame as f64 / input.fps), c.frame, source_label(c.source)),
            None => format!("{}  #{b}  diff {:.1}", fmt_time(b as f64 / input.fps), diff_at(b)),
        };
        painter.text(pos2(rect.left() + 4.0, rect.top() + 2.0), egui::Align2::LEFT_TOP, text, FontId::monospace(11.0), ui.visuals().strong_text_color());
    }

    // ---- interaction ----
    let response = if hovered_cut.is_some() || view.dragging.is_some() {
        response.on_hover_cursor(CursorIcon::ResizeHorizontal)
    } else {
        response
    };

    if response.drag_started() {
        let origin = ui.input(|i| i.pointer.press_origin()).map(|p| p.x);
        view.dragging = origin.and_then(near).map(|(c, _)| c.id);
        if let Some(id) = view.dragging {
            actions.push(Action::Checkpoint);
            actions.push(Action::SelectCut(Some(id)));
        }
    }
    if response.dragged() {
        if let Some(x) = response.interact_pointer_pos().map(|p| p.x) {
            match view.dragging {
                Some(id) => {
                    let raw = f_of(x).round().max(1.0) as usize;
                    let alt = ui.input(|i| i.modifiers.alt);
                    let radius = (SNAP_PX as f64 / px_per_frame).floor() as usize;
                    let frame = if alt { raw } else { snap_to_spike(input.diffs, raw, radius) };
                    actions.push(Action::MoveCut { id, frame });
                }
                None => actions.push(Action::Seek(f_of(x).floor().max(0.0) as usize)),
            }
        }
    }
    if response.drag_stopped() {
        view.dragging = None;
    }

    if response.clicked() {
        match hovered_cut {
            Some(c) => {
                actions.push(Action::SelectCut(Some(c.id)));
                actions.push(Action::Seek(c.frame));
            }
            None => {
                actions.push(Action::SelectCut(None));
                if let Some(p) = pointer {
                    actions.push(Action::Seek(f_of(p.x).floor().max(0.0) as usize));
                }
            }
        }
    }
    if response.double_clicked() && hovered_cut.is_none() {
        if let Some(p) = pointer {
            actions.push(Action::SplitAt(f_of(p.x).round() as usize));
        }
    }

    if response.secondary_clicked() {
        view.context = match (hovered_cut, pointer) {
            (Some(c), _) => Some(ContextTarget::Cut(c.id)),
            (None, Some(p)) => Some(ContextTarget::Frame(f_of(p.x).round() as usize)),
            _ => None,
        };
    }
    let context = view.context;
    response.context_menu(|ui| match context {
        Some(ContextTarget::Cut(id)) => {
            let cut = input.cuts.get(id);
            if ui.button("Select and show frames").clicked() {
                actions.push(Action::SelectCut(Some(id)));
                if let Some(c) = cut {
                    actions.push(Action::Seek(c.frame));
                }
                ui.close();
            }
            if ui.add_enabled(cut.is_some_and(|c| c.source == CutSource::Moved), egui::Button::new("Reset to detected position")).clicked() {
                actions.push(Action::ResetCut(id));
                ui.close();
            }
            if ui.button("Delete cut").clicked() {
                actions.push(Action::DeleteCut(id));
                ui.close();
            }
        }
        Some(ContextTarget::Frame(f)) => {
            if ui.button(format!("Add cut at frame {f}")).clicked() {
                actions.push(Action::SplitAt(f));
                ui.close();
            }
            if ui.button("Play from here").clicked() {
                actions.push(Action::Seek(f));
                actions.push(Action::TogglePlay);
                ui.close();
            }
        }
        None => {
            ui.close();
        }
    });

    // Zoom (Ctrl+scroll / pinch) and pan (scroll) while hovering.
    if response.hovered() {
        let (zoom, scroll) = ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta()));
        if zoom != 1.0 {
            if let Some(p) = pointer {
                view.zoom(zoom as f64, f_of(p.x));
            }
        } else {
            let delta = if scroll.x != 0.0 { scroll.x } else { scroll.y };
            if delta != 0.0 {
                view.pan(-(delta as f64) / px_per_frame);
            }
        }
    }

    actions
}

pub struct ToolbarInput {
    pub selected_cut: Option<Cut>,
    pub has_selected_scene: bool,
    pub has_cut_edits: bool,
    pub can_undo: bool,
    pub can_redo: bool,
}

pub fn toolbar(ui: &mut egui::Ui, view: &mut EditorView, input: &ToolbarInput) -> Vec<Action> {
    let mut actions = Vec::new();
    ui.horizontal_wrapped(|ui| {
        if ui.button("−").on_hover_text("Zoom out (Ctrl+scroll on the graph)").clicked() {
            view.zoom(0.5, (view.start + view.end) / 2.0);
        }
        if ui.button("+").on_hover_text("Zoom in (Ctrl+scroll on the graph)").clicked() {
            view.zoom(2.0, (view.start + view.end) / 2.0);
        }
        if ui.add_enabled(view.is_zoomed(), egui::Button::new("Fit")).clicked() {
            view.fit();
        }
        if ui.add_enabled(input.has_selected_scene, egui::Button::new("Zoom to scene")).clicked() {
            actions.push(Action::ZoomToSelectedScene);
        }
        ui.separator();

        if ui.button("✂ Split").on_hover_text("Start a new scene at the playhead (S)").clicked() {
            actions.push(Action::Split);
        }
        let has_cut = input.selected_cut.is_some();
        if ui.add_enabled(has_cut, egui::Button::new("🗑 Delete cut")).on_hover_text("Delete the selected cut (Delete)").clicked() {
            actions.push(Action::DeleteSelected);
        }
        if ui.button("◀").on_hover_text("Nudge cut one frame earlier (,  Shift: 10)").clicked() {
            actions.push(Action::Nudge(-1));
        }
        if ui.button("▶").on_hover_text("Nudge cut one frame later (.  Shift: 10)").clicked() {
            actions.push(Action::Nudge(1));
        }
        let moved = input.selected_cut.filter(|c| c.source == CutSource::Moved);
        if ui.add_enabled(moved.is_some(), egui::Button::new("Reset cut")).on_hover_text("Put the cut back where it was detected").clicked() {
            actions.extend(moved.map(|c| Action::ResetCut(c.id)));
        }
        ui.separator();

        if ui.add_enabled(input.can_undo, egui::Button::new("⟲ Undo")).on_hover_text("Ctrl+Z").clicked() {
            actions.push(Action::Undo);
        }
        if ui.add_enabled(input.can_redo, egui::Button::new("⟳ Redo")).on_hover_text("Ctrl+Shift+Z / Ctrl+Y").clicked() {
            actions.push(Action::Redo);
        }
        if ui.add_enabled(input.has_cut_edits, egui::Button::new("Reset all cuts")).on_hover_text("Discard every cut you moved, added or deleted").clicked() {
            actions.push(Action::ResetAllCuts);
        }
        ui.separator();
        ui.label(RichText::new("⌨ Shortcuts").weak()).on_hover_text(SHORTCUTS);
        legend(ui);
    });
    actions
}

fn legend(ui: &mut egui::Ui) {
    for (color, label) in [(DETECTED_COLOR, "detected"), (MOVED_COLOR, "moved"), (MANUAL_COLOR, "manual")] {
        let (rect, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
        ui.painter().rect_filled(rect, 2.0, color);
        ui.label(RichText::new(label).small().weak());
    }
}

pub const VIDEO_COLOR: Color32 = Color32::from_rgb(70, 130, 200);
pub const STILL_COLOR: Color32 = Color32::from_rgb(90, 170, 110);

pub struct OverviewInput<'a> {
    pub scenes: &'a [Scene],
    pub edits: &'a Edits,
    pub selected: Option<CutId>,
    pub frame_count: usize,
    pub fps: f64,
    /// Frame on screen in the player.
    pub shown_frame: usize,
    /// Frame the player is at or heading to (see `Player::position`).
    pub position: usize,
}

pub const OVERVIEW_HEIGHT: f32 = 22.0;

/// The bar under the video: every scene's exported frames (blue = clip, green = still,
/// faded = not exported), the selected scene, the graph's zoom window, and the playhead.
/// Click a clip to select it (the playhead moves to the click); drag to scrub.
pub fn overview(ui: &mut egui::Ui, input: &OverviewInput<'_>, view: &EditorView) -> Option<Action> {
    let total = input.frame_count.max(1);
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), OVERVIEW_HEIGHT), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let to_x = |frame: f64| rect.left() + (frame / total as f64) as f32 * rect.width();
    let frame_at = |x: f32| ((((x - rect.left()) / rect.width()) * total as f32).max(0.0) as usize).min(total - 1);
    let scene_at = |frame: usize| input.scenes.partition_point(|s| s.start <= frame).saturating_sub(1);
    let hovered = response.hover_pos().map(|p| scene_at(frame_at(p.x)));

    for (i, scene) in input.scenes.iter().enumerate() {
        let full = Rect::from_x_y_ranges(to_x(scene.start as f64)..=to_x(scene.end as f64), rect.y_range());
        if hovered == Some(i) {
            painter.rect_filled(full, 1.0, ui.visuals().widgets.hovered.bg_fill.gamma_multiply(0.5));
        }
        if !scene.keep.is_empty() {
            let color = match input.edits.kinds.get(&scene.id).copied().unwrap_or(scene.kind) {
                SceneKind::Video => VIDEO_COLOR,
                SceneKind::Still => STILL_COLOR,
            };
            let alpha = if input.edits.excluded.contains(&scene.id) { 0.2 } else { 0.6 };
            let r = Rect::from_x_y_ranges(to_x(scene.keep.start as f64)..=to_x(scene.keep.end as f64), rect.y_range())
                .shrink2(vec2(0.5, 3.0));
            painter.rect_filled(r, 1.0, color.gamma_multiply(alpha));
        }
        if input.selected == Some(scene.id) {
            let r = full.shrink2(vec2(0.5, 1.0));
            painter.rect_stroke(r, 2.0, Stroke::new(2.0, DETECTED_COLOR), egui::StrokeKind::Inside);
        }
    }
    if view.is_zoomed() {
        let r = Rect::from_x_y_ranges(to_x(view.start)..=to_x(view.end), rect.y_range());
        let stroke = Stroke::new(1.0, ui.visuals().strong_text_color().gamma_multiply(0.7));
        painter.rect_stroke(r, 1.0, stroke, egui::StrokeKind::Inside);
    }
    let x = to_x(input.shown_frame as f64);
    painter.vline(x, rect.y_range(), Stroke::new(2.0, ui.visuals().strong_text_color()));

    let response = match hovered.and_then(|i| input.scenes.get(i).map(|s| (i, s))) {
        Some((i, s)) if !response.dragged() => response.on_hover_text(format!(
            "Scene #{}  {} – {}\nClick to select, drag to scrub",
            i + 1,
            fmt_time(s.start as f64 / input.fps),
            fmt_time(s.end as f64 / input.fps),
        )),
        _ => response,
    };

    let frame = frame_at(response.interact_pointer_pos()?.x);
    if response.clicked() {
        Some(Action::SelectSceneAt { index: scene_at(frame), frame })
    } else if response.dragged() && frame != input.position {
        // Only seek when the frame changes, so holding still doesn't restart the decoder.
        Some(Action::Seek(frame))
    } else {
        None
    }
}

pub struct StripInput<'a> {
    pub cut: Cut,
    pub frame_count: usize,
    pub fps: f64,
    /// Where the cut may move without crossing its neighbours.
    pub movable: Range<usize>,
    pub frames: &'a HashMap<usize, TextureHandle>,
    pub aspect: f32,
}

/// Frames around the selected cut. Clicking a frame makes it the first frame of the new scene.
/// Returns actions plus the frames that need decoding.
pub fn filmstrip(ui: &mut egui::Ui, input: &StripInput<'_>) -> (Vec<Action>, Range<usize>) {
    let mut actions = Vec::new();
    let cut = input.cut;
    let first = cut.frame.saturating_sub(STRIP_RADIUS);
    let last = (cut.frame + STRIP_RADIUS + 1).min(input.frame_count);
    let window = first..last;

    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("Cut at #{}  ({})", cut.frame, fmt_time(cut.frame as f64 / input.fps))).strong());
        let detail = match (cut.source, cut.ghost) {
            (CutSource::Moved, Some(g)) => format!("moved {:+} frames from detected", cut.frame as i64 - g as i64),
            (CutSource::Moved, None) => "moved (no longer detected)".to_owned(),
            (source, _) => source_label(source).to_owned(),
        };
        ui.label(RichText::new(detail).color(cut_color(cut.source)));
        ui.label(RichText::new("Click the first frame of the new scene").weak());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("✕").on_hover_text("Deselect cut (Esc)").clicked() {
                actions.push(Action::SelectCut(None));
            }
        });
    });

    let count = (2 * STRIP_RADIUS + 1) as f32;
    let spacing = 4.0;
    let w = ((ui.available_width() - spacing * (count - 1.0) - 8.0) / count).clamp(24.0, 160.0);
    let size = vec2(w, w / input.aspect.max(0.1));

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = spacing;
        for f in window.clone() {
            if f == cut.frame {
                // Divider between the outgoing and incoming scene.
                let (r, _) = ui.allocate_exact_size(vec2(3.0, size.y + 16.0), Sense::hover());
                ui.painter().rect_filled(r, 1.0, cut_color(cut.source));
            }
            ui.vertical(|ui| {
                let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
                resp.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Frame {f}")));
                match input.frames.get(&f) {
                    Some(tex) => {
                        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
                        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
                    }
                    None => {
                        ui.painter().rect_filled(rect, 2.0, ui.visuals().faint_bg_color);
                    }
                }
                let incoming = f >= cut.frame;
                if !incoming {
                    ui.painter().rect_filled(rect, 0.0, Color32::from_black_alpha(90));
                }
                let movable = input.movable.contains(&f) && f != cut.frame;
                if resp.hovered() && movable {
                    ui.painter().rect_stroke(rect, 2.0, Stroke::new(2.0, SELECTED_COLOR), egui::StrokeKind::Inside);
                }
                let label = RichText::new(format!("{f}")).small().monospace();
                ui.label(if f == cut.frame { label.strong().color(cut_color(cut.source)) } else { label.weak() });
                let resp = if movable { resp.on_hover_cursor(CursorIcon::PointingHand) } else { resp };
                if resp.clicked() {
                    if movable {
                        actions.push(Action::Checkpoint);
                        actions.push(Action::MoveCut { id: cut.id, frame: f });
                    } else {
                        actions.push(Action::Seek(f));
                    }
                }
            });
        }
    });
    (actions, window)
}

fn nearest_cut<'a>(cuts: &'a [Cut], dist: impl Fn(&Cut) -> f32) -> Option<(&'a Cut, f32)> {
    cuts.iter().map(|c| (c, dist(c))).min_by(|a, b| a.1.total_cmp(&b.1))
}

/// Move `frame` to the strongest nearby spike, if there's a clearly stronger one.
pub fn snap_to_spike(diffs: &[f32], frame: usize, radius: usize) -> usize {
    if radius == 0 || diffs.is_empty() {
        return frame;
    }
    let at = |b: usize| b.checked_sub(1).and_then(|i| diffs.get(i)).copied().unwrap_or(0.0);
    let lo = frame.saturating_sub(radius).max(1);
    let hi = (frame + radius).min(diffs.len());
    let best = (lo..=hi).max_by(|&a, &b| at(a).total_cmp(&at(b)).then(b.abs_diff(frame).cmp(&a.abs_diff(frame))));
    match best {
        Some(b) if at(b) >= 4.0 && at(b) >= 2.0 * at(frame) => b,
        _ => frame,
    }
}

fn source_label(source: CutSource) -> &'static str {
    match source {
        CutSource::Detected => "detected",
        CutSource::Moved => "moved",
        CutSource::Manual => "manual",
    }
}

pub fn fmt_time(secs: f64) -> String {
    let secs = secs.max(0.0);
    let m = (secs / 60.0).floor() as u64;
    format!("{m}:{:04.1}", secs - m as f64 * 60.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_to_nearby_spike_only_when_clearly_stronger() {
        let mut diffs = vec![1.0; 50];
        diffs[19] = 60.0; // spike suggesting a cut at frame 20
        assert_eq!(snap_to_spike(&diffs, 17, 5), 20);
        assert_eq!(snap_to_spike(&diffs, 23, 5), 20);
        assert_eq!(snap_to_spike(&diffs, 30, 5), 30, "out of range");
        assert_eq!(snap_to_spike(&diffs, 17, 0), 17, "no snapping when zoomed in");
        assert_eq!(snap_to_spike(&vec![1.0; 50], 17, 5), 17, "no spike");
        assert_eq!(snap_to_spike(&diffs, 1000, 5), 1000, "past the end");
    }

    #[test]
    fn view_zoom_and_pan_stay_in_bounds() {
        let mut v = EditorView::new(1000);
        v.zoom(1e9, 500.0);
        assert_eq!(v.span(), MIN_SPAN);
        assert!((v.start..=v.end).contains(&500.0));
        v.pan(-1e9);
        assert_eq!((v.start, v.end), (0.0, MIN_SPAN));
        v.pan(1e9);
        assert_eq!(v.end, 1000.0);
        v.zoom(1e-9, 0.0);
        assert_eq!((v.start, v.end), (0.0, 1000.0));

        let mut tiny = EditorView::new(3);
        tiny.zoom(10.0, 1.0);
        assert_eq!((tiny.start, tiny.end), (0.0, 3.0));
    }

    #[test]
    fn view_follows_playhead_off_screen() {
        let mut v = EditorView::new(1000);
        v.show_range(100..200);
        v.follow(150);
        let before = (v.start, v.end);
        assert_eq!((v.start, v.end), before, "on-screen playhead doesn't move the view");
        v.follow(900);
        assert!((v.start..v.end).contains(&900.0));
    }
}
