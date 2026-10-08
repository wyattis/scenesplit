//! UI tests that drive the widgets with real mouse and keyboard input via `egui_kittest`.
//!
//! Widget tests render one widget at a known size, send pointer/key events, and check the
//! [`Action`]s it returns. The whole-app test at the bottom runs the real app on a synthetic
//! video and clicks actual buttons.

use std::collections::HashMap;

use eframe::egui::{self, Event, Key, Modifiers, MouseWheelUnit, PointerButton, Pos2, TouchPhase, pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;

use crate::editor::{self, Action, EditorView, GraphInput, OverviewInput, StripInput};
use crate::project::Edits;
use crate::scenes::{self, CutEdits, CutId, Params, ResolvedCuts};

const WIDTH: f32 = 1000.0;
/// 201 frames over 1000px: 5px per frame when zoomed out.
const FRAMES: usize = 201;

// ---- helpers ------------------------------------------------------------------------------

/// Differences with clear spikes at frames 50 and 120 (detected cuts), and a weaker one at 60
/// that detection doesn't pick but snapping should.
fn diffs() -> Vec<f32> {
    let mut d = vec![1.0; FRAMES - 1];
    d[49] = 80.0;
    d[119] = 80.0;
    d[59] = 30.0;
    d
}

fn cuts(edits: &CutEdits) -> ResolvedCuts {
    scenes::resolve_cuts(&[50, 120], edits, FRAMES, 0)
}

struct GraphState {
    view: EditorView,
    diffs: Vec<f32>,
    cuts: ResolvedCuts,
    actions: Vec<Action>,
    /// Top-left and width of the graph, recorded while drawing.
    origin: Pos2,
    width: f32,
}

impl GraphState {
    /// Screen x of the boundary before `frame`, in the current zoom.
    fn x(&self, frame: f64) -> f32 {
        self.origin.x + ((frame - self.view.start) / (self.view.end - self.view.start)) as f32 * self.width
    }

    fn pos(&self, frame: f64) -> Pos2 {
        // Mid-height of the 90px graph, away from the marker handles at the top.
        pos2(self.x(frame), self.origin.y + 50.0)
    }

    fn take(&mut self) -> Vec<Action> {
        std::mem::take(&mut self.actions)
    }
}

fn graph_harness(edits: CutEdits) -> Harness<'static, GraphState> {
    let state = GraphState {
        view: EditorView::new(FRAMES),
        diffs: diffs(),
        cuts: cuts(&edits),
        actions: Vec::new(),
        origin: Pos2::ZERO,
        width: 0.0,
    };
    let mut harness = Harness::builder()
        .with_size(vec2(WIDTH, 200.0))
        .with_step_dt(0.01)
        .build_ui_state(
            |ui, s: &mut GraphState| {
                s.origin = ui.cursor().min;
                s.width = ui.available_width();
                let input = GraphInput {
                    diffs: &s.diffs,
                    cuts: &s.cuts,
                    frame_count: FRAMES,
                    fps: 25.0,
                    playhead: 0,
                    selected_cut: None,
                    threshold: 8.0,
                };
                let actions = editor::graph(ui, &mut s.view, &input);
                s.actions.extend(actions);
            },
            state,
        );
    harness.state_mut().actions.clear();
    harness
}

fn press(h: &Harness<'_, impl Sized>, pos: Pos2, button: PointerButton, pressed: bool) {
    h.event(Event::PointerButton { pos, button, pressed, modifiers: Modifiers::NONE });
}

fn click_at(h: &mut Harness<'_, impl Sized>, pos: Pos2) {
    h.hover_at(pos);
    press(h, pos, PointerButton::Primary, true);
    press(h, pos, PointerButton::Primary, false);
    h.step();
}

/// Press at `from`, move to `to` in a few steps, release there. Optional held modifiers.
fn drag(h: &mut Harness<'_, impl Sized>, from: Pos2, to: Pos2, modifiers: Modifiers) {
    h.hover_at(from);
    h.event(Event::ModifiersChanged(modifiers));
    press(h, from, PointerButton::Primary, true);
    for t in [0.25, 0.5, 0.75, 1.0] {
        h.hover_at(from + (to - from) * t);
    }
    press(h, to, PointerButton::Primary, false);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.step();
}

fn last_move(actions: &[Action]) -> Option<(CutId, usize)> {
    actions.iter().rev().find_map(|a| match a {
        Action::MoveCut { id, frame } => Some((*id, *frame)),
        _ => None,
    })
}

// ---- difference graph -----------------------------------------------------------------------

#[test]
fn dragging_a_marker_moves_its_cut_with_snapping() {
    let mut h = graph_harness(CutEdits::default());
    let (from, to) = { (h.state().pos(50.0), h.state().pos(58.6)) };
    drag(&mut h, from, to, Modifiers::NONE);
    let actions = h.state_mut().take();

    assert_eq!(actions.first(), Some(&Action::Checkpoint), "one undo step per drag");
    assert!(actions.contains(&Action::SelectCut(Some(CutId::Detected(50)))));
    // Dropped near frame 59, next to the spike suggesting a cut at 60: snaps there.
    assert_eq!(last_move(&actions), Some((CutId::Detected(50), 60)));
    assert!(!actions.iter().any(|a| matches!(a, Action::Seek(_))), "dragging a cut doesn't scrub");
}

#[test]
fn alt_disables_snapping() {
    let mut h = graph_harness(CutEdits::default());
    let (from, to) = (h.state().pos(50.0), h.state().pos(58.6));
    drag(&mut h, from, to, Modifiers::ALT);
    assert_eq!(last_move(&h.state_mut().take()), Some((CutId::Detected(50), 59)));
}

#[test]
fn dragging_empty_space_scrubs() {
    let mut h = graph_harness(CutEdits::default());
    let (from, to) = (h.state().pos(80.0), h.state().pos(100.0));
    drag(&mut h, from, to, Modifiers::NONE);
    let actions = h.state_mut().take();
    assert_eq!(last_move(&actions), None);
    assert!(actions.iter().any(|a| matches!(a, Action::Seek(f) if (99..=100).contains(f))), "{actions:?}");
}

#[test]
fn grabs_the_nearest_of_two_close_markers() {
    // Manual cuts at 100 and 101: markers 5px apart.
    let mut edits = CutEdits::default();
    edits.add(100);
    let second = edits.add(101);
    let mut h = graph_harness(edits);
    let from = h.state().pos(101.0) + vec2(-1.5, 0.0);
    let to = h.state().pos(110.0);
    drag(&mut h, from, to, Modifiers::ALT);
    assert_eq!(last_move(&h.state_mut().take()).map(|m| m.0), Some(second));
}

#[test]
fn clicking_selects_a_marker_or_seeks() {
    let mut h = graph_harness(CutEdits::default());
    let on_marker = h.state().pos(120.0) + vec2(2.0, 0.0);
    click_at(&mut h, on_marker);
    assert_eq!(h.state_mut().take(), [Action::SelectCut(Some(CutId::Detected(120))), Action::Seek(120)]);

    let empty = h.state().pos(90.4);
    click_at(&mut h, empty);
    assert_eq!(h.state_mut().take(), [Action::SelectCut(None), Action::Seek(90)]);
}

#[test]
fn double_click_adds_a_cut() {
    let mut h = graph_harness(CutEdits::default());
    let pos = h.state().pos(150.0);
    h.hover_at(pos);
    for _ in 0..2 {
        press(&h, pos, PointerButton::Primary, true);
        press(&h, pos, PointerButton::Primary, false);
    }
    h.step();
    assert!(h.state_mut().take().contains(&Action::SplitAt(150)));
}

#[test]
fn right_click_menu_deletes_a_cut() {
    let mut h = graph_harness(CutEdits::default());
    let pos = h.state().pos(120.0);
    h.hover_at(pos);
    press(&h, pos, PointerButton::Secondary, true);
    press(&h, pos, PointerButton::Secondary, false);
    h.run_steps(3);
    h.get_by_label("Delete cut").click();
    h.run_steps(2);
    assert!(h.state_mut().take().contains(&Action::DeleteCut(CutId::Detected(120))));
}

#[test]
fn right_click_on_empty_space_offers_to_add_a_cut() {
    let mut h = graph_harness(CutEdits::default());
    let pos = h.state().pos(170.0);
    h.hover_at(pos);
    press(&h, pos, PointerButton::Secondary, true);
    press(&h, pos, PointerButton::Secondary, false);
    h.run_steps(3);
    h.get_by_label("Add cut at frame 170").click();
    h.run_steps(2);
    assert!(h.state_mut().take().contains(&Action::SplitAt(170)));
}

fn wheel(h: &mut Harness<'_, GraphState>, at: Pos2, delta: egui::Vec2, modifiers: Modifiers) {
    h.hover_at(at);
    h.event(Event::ModifiersChanged(modifiers));
    h.event(Event::MouseWheel { unit: MouseWheelUnit::Point, delta, phase: TouchPhase::Move, modifiers });
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    // Smooth scrolling spreads the delta over several frames.
    h.run_steps(30);
}

#[test]
fn ctrl_scroll_zooms_around_the_pointer_and_scroll_pans() {
    let mut h = graph_harness(CutEdits::default());
    let at = h.state().pos(120.0);
    wheel(&mut h, at, vec2(0.0, 200.0), Modifiers::COMMAND);
    let view = &h.state().view;
    assert!(view.end - view.start < FRAMES as f64 * 0.75, "zoomed in: {}..{}", view.start, view.end);
    // The frame under the pointer stays (about) under the pointer.
    assert!((h.state().x(120.0) - at.x).abs() < 3.0);

    let start = h.state().view.start;
    wheel(&mut h, at, vec2(0.0, -100.0), Modifiers::NONE);
    assert!(h.state().view.start > start, "scrolling pans the zoomed view");
}

// ---- overview bar under the video -----------------------------------------------------------

struct OverviewState {
    scenes: Vec<scenes::Scene>,
    edits: Edits,
    view: EditorView,
    actions: Vec<Action>,
    origin: Pos2,
    width: f32,
}

fn overview_harness() -> Harness<'static, OverviewState> {
    let resolved = cuts(&CutEdits::default());
    let state = OverviewState {
        scenes: scenes::build_scenes(&diffs(), FRAMES, &resolved.cuts, &Params::default()),
        edits: Edits::default(),
        view: EditorView::new(FRAMES),
        actions: Vec::new(),
        origin: Pos2::ZERO,
        width: 0.0,
    };
    Harness::builder().with_size(vec2(WIDTH, 100.0)).with_step_dt(0.01).build_ui_state(
        |ui, s: &mut OverviewState| {
            s.origin = ui.cursor().min;
            s.width = ui.available_width();
            let input = OverviewInput {
                scenes: &s.scenes,
                edits: &s.edits,
                selected: None,
                frame_count: FRAMES,
                fps: 25.0,
                shown_frame: 0,
                position: 0,
            };
            s.actions.extend(editor::overview(ui, &input, &s.view));
        },
        state,
    )
}

fn overview_pos(s: &OverviewState, frame: f32) -> Pos2 {
    pos2(s.origin.x + (frame + 0.5) / FRAMES as f32 * s.width, s.origin.y + editor::OVERVIEW_HEIGHT / 2.0)
}

#[test]
fn clicking_the_overview_selects_the_clip_under_the_pointer() {
    let mut h = overview_harness();
    h.state_mut().actions.clear();
    let pos = overview_pos(h.state(), 130.0);
    click_at(&mut h, pos);
    // Scenes: 0..50, 50..120, 120..201.
    assert_eq!(h.state().actions, [Action::SelectSceneAt { index: 2, frame: 130 }]);
}

#[test]
fn dragging_the_overview_scrubs_without_selecting() {
    let mut h = overview_harness();
    h.state_mut().actions.clear();
    let (from, to) = (overview_pos(h.state(), 20.0), overview_pos(h.state(), 60.0));
    drag(&mut h, from, to, Modifiers::NONE);
    let actions = &h.state().actions;
    assert!(actions.iter().all(|a| matches!(a, Action::Seek(_))), "{actions:?}");
    assert!(actions.contains(&Action::Seek(60)));
}

// ---- filmstrip ---------------------------------------------------------------------------

fn strip_harness(edits: CutEdits) -> Harness<'static, Vec<Action>> {
    let resolved = cuts(&edits);
    let cut = *resolved.get(CutId::Detected(50)).unwrap();
    let movable = resolved.movable_range(cut.id, FRAMES);
    let frames = HashMap::new();
    Harness::builder().with_size(vec2(1400.0, 300.0)).build_ui_state(
        move |ui, actions: &mut Vec<Action>| {
            let input = StripInput { cut, frame_count: FRAMES, fps: 25.0, movable: movable.clone(), frames: &frames, aspect: 16.0 / 9.0 };
            actions.extend(editor::filmstrip(ui, &input).0);
        },
        Vec::new(),
    )
}

#[test]
fn clicking_a_filmstrip_frame_moves_the_cut_there() {
    let mut h = strip_harness(CutEdits::default());
    h.get_by_label("Frame 53").click();
    h.run_steps(2);
    assert_eq!(*h.state(), [Action::Checkpoint, Action::MoveCut { id: CutId::Detected(50), frame: 53 }]);
}

#[test]
fn filmstrip_frames_the_cut_cant_reach_just_seek() {
    // A manual cut at 52 blocks cut 50 from moving to 53 or later.
    let mut edits = CutEdits::default();
    edits.add(52);
    let mut h = strip_harness(edits);
    h.get_by_label("Frame 53").click();
    h.get_by_label("Frame 50").click();
    h.run_steps(2);
    assert_eq!(*h.state(), [Action::Seek(53), Action::Seek(50)]);
}

// ---- keyboard ------------------------------------------------------------------------------

fn keyboard_harness() -> Harness<'static, (String, Vec<Action>)> {
    Harness::new_ui_state(
        |ui, (text, actions): &mut (String, Vec<Action>)| {
            ui.add(egui::TextEdit::singleline(text).hint_text("field"));
            actions.extend(editor::shortcut_actions(ui.ctx()));
        },
        (String::new(), Vec::new()),
    )
}

#[test]
fn keys_map_to_actions() {
    let cases = [
        (Modifiers::NONE, Key::S, Some(Action::Split)),
        (Modifiers::NONE, Key::Space, Some(Action::TogglePlay)),
        (Modifiers::NONE, Key::ArrowLeft, Some(Action::Step(-1))),
        (Modifiers::SHIFT, Key::ArrowRight, Some(Action::Step(10))),
        (Modifiers::NONE, Key::Comma, Some(Action::Nudge(-1))),
        (Modifiers::SHIFT, Key::Period, Some(Action::Nudge(10))),
        (Modifiers::NONE, Key::OpenBracket, Some(Action::JumpCut(-1))),
        (Modifiers::NONE, Key::CloseBracket, Some(Action::JumpCut(1))),
        (Modifiers::NONE, Key::Delete, Some(Action::DeleteSelected)),
        (Modifiers::NONE, Key::Backspace, Some(Action::DeleteSelected)),
        (Modifiers::NONE, Key::Escape, Some(Action::SelectCut(None))),
        (Modifiers::COMMAND, Key::Z, Some(Action::Undo)),
        (Modifiers::COMMAND | Modifiers::SHIFT, Key::Z, Some(Action::Redo)),
        (Modifiers::COMMAND, Key::Y, Some(Action::Redo)),
        (Modifiers::COMMAND, Key::S, None),
        (Modifiers::ALT, Key::S, None),
        (Modifiers::NONE, Key::Q, None),
    ];
    for (modifiers, key, expected) in cases {
        let mut h = keyboard_harness();
        h.key_press_modifiers(modifiers, key);
        h.step();
        assert_eq!(h.state().1, expected.into_iter().collect::<Vec<_>>(), "{modifiers:?} {key:?}");
    }
}

#[test]
fn shortcuts_are_ignored_while_typing_in_a_field() {
    let mut h = keyboard_harness();
    h.get_by_role(egui::accesskit::Role::TextInput).focus();
    h.run_steps(2);
    h.key_press(Key::S);
    h.key_press(Key::Delete);
    h.run_steps(2);
    assert!(h.state().1.is_empty(), "{:?}", h.state().1);
}

// ---- whole app --------------------------------------------------------------------------------

/// Runs the real app on the synthetic video (cuts at 75 and 150) and clicks real buttons.
#[test]
#[ignore = "needs ffmpeg on PATH"]
fn app_buttons_and_keys_edit_the_cuts() {
    use crate::app::App;
    use std::time::{Duration, Instant};

    let input = crate::ffmpeg::make_test_video("ui");
    let _ = std::fs::remove_file(crate::project::sidecar_path(&input));
    let mut h = Harness::builder().with_size(vec2(1500.0, 1100.0)).build_eframe(|cc| App::new(cc));
    let ctx = h.ctx.clone();
    h.state_mut().open_for_test(&ctx, input);
    let deadline = Instant::now() + Duration::from_secs(20);
    while h.state().scene_count().is_none() {
        assert!(Instant::now() < deadline, "analysis timed out");
        std::thread::sleep(Duration::from_millis(10));
        h.step();
    }
    h.run_steps(3);
    assert_eq!(h.state().scene_count(), Some(3));
    h.get_by_label("Export 3 scenes");

    h.get_all_by_label("Merge with next").next().unwrap().click();
    h.run_steps(2);
    assert_eq!(h.state().scene_count(), Some(2));
    h.get_by_label("⟲ Undo").click();
    h.run_steps(2);
    assert_eq!(h.state().scene_count(), Some(3));

    // ] selects the next cut and opens its filmstrip; clicking a frame moves the cut.
    h.key_press(Key::CloseBracket);
    h.run_steps(3);
    assert_eq!(h.state().cut_frames(), [75, 150]);
    h.get_by_label("Frame 78").click();
    h.run_steps(2);
    assert_eq!(h.state().cut_frames(), [78, 150]);
    // . nudges it, Ctrl+Z undoes the nudge and then the move.
    h.key_press(Key::Period);
    h.run_steps(2);
    assert_eq!(h.state().cut_frames(), [79, 150]);
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.run_steps(2);
    assert_eq!(h.state().cut_frames(), [75, 150]);
}
