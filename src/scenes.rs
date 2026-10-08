//! Turning per-frame difference scores into scenes. Pure and cheap to re-run.
//!
//! The pipeline is: [`sections::plan`] (which settings apply where) → [`detect`] →
//! detected cuts + the user's [`CutEdits`] → [`resolve`] → [`build`]. The `*_cuts`/`build_scenes`
//! functions are the same with one set of settings for the whole video.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::sections::{self, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectMode {
    /// Cut wherever the difference exceeds a fixed threshold.
    Fixed,
    /// Cut where the difference is much larger than its neighbours. Copes
    /// better with fast camera motion, where every frame differs a lot.
    Adaptive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    pub mode: DetectMode,
    /// Fixed mode: minimum difference (0-255) for a cut.
    pub cut_threshold: f32,
    /// Adaptive mode: how many times larger than the neighbourhood average a difference must be.
    pub adaptive_ratio: f32,
    /// Adaptive mode: frames on each side used for the neighbourhood average.
    pub adaptive_window: usize,
    /// Adaptive mode: ignore differences below this, so noise in static shots can't trigger cuts.
    pub adaptive_floor: f32,
    /// Detected cuts closer together than this are ignored.
    pub min_scene_secs: f32,
    /// Scenes whose median frame-to-frame difference is below this are treated as stills.
    pub still_threshold: f32,
    /// Moves every *detected* cut by this many frames. 0 starts the new scene on the first
    /// frame after the spike; negative values start it earlier, positive values later.
    /// Cuts the user placed by hand are never offset.
    pub cut_offset: i32,
    /// Frames dropped from the end of the scene before each cut.
    pub drop_before_cut: usize,
    /// Frames dropped from the start of the scene after each cut.
    pub drop_after_cut: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mode: DetectMode::Adaptive,
            cut_threshold: 30.0,
            adaptive_ratio: 3.0,
            adaptive_window: 4,
            adaptive_floor: 8.0,
            min_scene_secs: 0.5,
            still_threshold: 0.5,
            cut_offset: 0,
            drop_before_cut: 0,
            drop_after_cut: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneKind {
    Video,
    Still,
}

/// Stable identity of a cut (and of the scene that starts at it), independent of where the
/// cut currently sits. Used to key user choices so they survive setting changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub enum CutId {
    /// Not a real cut: the start of the video, which begins the first scene.
    Start,
    /// A detected cut, by the frame detection originally placed it at (before any offset).
    Detected(usize),
    /// A cut the user added, by a counter.
    Manual(u32),
}

// String form ("start", "d123", "m4") so `CutId` can be a JSON map key.
impl fmt::Display for CutId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CutId::Start => write!(f, "start"),
            CutId::Detected(d) => write!(f, "d{d}"),
            CutId::Manual(m) => write!(f, "m{m}"),
        }
    }
}

impl From<CutId> for String {
    fn from(id: CutId) -> Self {
        id.to_string()
    }
}

impl TryFrom<String> for CutId {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        let bad = || format!("invalid cut id {s:?}");
        match s.split_at_checked(1) {
            _ if s == "start" => Ok(CutId::Start),
            Some(("d", n)) => n.parse().map(CutId::Detected).map_err(|_| bad()),
            Some(("m", n)) => n.parse().map(CutId::Manual).map_err(|_| bad()),
            _ => Err(bad()),
        }
    }
}

/// The user's changes to the detected cuts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CutEdits {
    /// Detected cut → exact frame it was moved to.
    pub moved: BTreeMap<usize, usize>,
    /// Detected cuts the user deleted.
    pub removed: BTreeSet<usize>,
    /// Manual cut id → frame.
    pub added: BTreeMap<u32, usize>,
    pub next_manual_id: u32,
}

impl CutEdits {
    pub fn is_empty(&self) -> bool {
        self.moved.is_empty() && self.removed.is_empty() && self.added.is_empty()
    }

    pub fn add(&mut self, frame: usize) -> CutId {
        let id = self.next_manual_id;
        self.next_manual_id += 1;
        self.added.insert(id, frame);
        CutId::Manual(id)
    }

    pub fn set_frame(&mut self, id: CutId, frame: usize) {
        match id {
            CutId::Start => {}
            CutId::Detected(d) => {
                self.moved.insert(d, frame);
            }
            CutId::Manual(m) => {
                self.added.insert(m, frame);
            }
        }
    }

    pub fn delete(&mut self, id: CutId) {
        match id {
            CutId::Start => {}
            CutId::Detected(d) => {
                self.moved.remove(&d);
                self.removed.insert(d);
            }
            CutId::Manual(m) => {
                self.added.remove(&m);
            }
        }
    }

    /// Put a moved detected cut back where detection placed it.
    pub fn reset(&mut self, id: CutId) {
        if let CutId::Detected(d) = id {
            self.moved.remove(&d);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutSource {
    Detected,
    /// A detected cut the user moved. If detection no longer finds it (e.g. after a threshold
    /// change), it's kept at the user's position anyway.
    Moved,
    Manual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    pub id: CutId,
    /// First frame of the scene this cut starts.
    pub frame: usize,
    pub source: CutSource,
    /// For moved cuts that detection still finds: where detection (plus offset) would put it.
    pub ghost: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct ResolvedCuts {
    /// Sorted by frame, unique frames, all within `1..frame_count`.
    pub cuts: Vec<Cut>,
    /// Positions of detected cuts the user deleted, for display.
    pub removed: Vec<usize>,
}

impl ResolvedCuts {
    pub fn get(&self, id: CutId) -> Option<&Cut> {
        self.cuts.iter().find(|c| c.id == id)
    }

    /// Index of the cut with this id.
    pub fn index_of(&self, id: CutId) -> Option<usize> {
        self.cuts.iter().position(|c| c.id == id)
    }

    /// Frames a cut can move to without reaching its neighbours.
    pub fn movable_range(&self, id: CutId, frame_count: usize) -> Range<usize> {
        let Some(i) = self.index_of(id) else { return 1..frame_count.max(1) };
        let lo = if i == 0 { 1 } else { self.cuts[i - 1].frame + 1 };
        let hi = self.cuts.get(i + 1).map_or(frame_count, |c| c.frame);
        lo..hi.max(lo + 1)
    }
}

#[derive(Debug, Clone)]
pub struct Scene {
    /// The cut that starts this scene ([`CutId::Start`] for the first scene).
    pub id: CutId,
    /// Scene boundaries: frames `start..end`.
    pub start: usize,
    pub end: usize,
    /// Frames that will be exported, after dropping frames around cuts. May be empty.
    pub keep: Range<usize>,
    /// Median frame-to-frame difference over the kept frames.
    pub motion: f32,
    pub kind: SceneKind,
}

impl Scene {
    /// Frame used for thumbnails and still images.
    pub fn middle(&self) -> usize {
        if self.keep.is_empty() {
            self.start + (self.end - self.start) / 2
        } else {
            self.keep.start + self.keep.len() / 2
        }
    }
}

/// A cut found by detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Detected {
    /// The frame detection placed it at; its identity ([`CutId::Detected`]).
    pub id: usize,
    /// Where it goes after the cut offset (not yet clamped to the video).
    pub frame: i64,
}

/// Frame indices where a new scene starts (never includes 0), with one set of settings.
#[cfg(test)]
pub fn detect_cuts(diffs: &[f32], fps: f64, p: &Params) -> Vec<usize> {
    let mut cuts = Vec::new();
    detect_range(diffs, fps, p, 1..diffs.len() + 1, &mut 0, &mut cuts);
    cuts
}

/// Detected cuts, each span using its own settings. Locked spans use their frozen cuts.
pub fn detect(diffs: &[f32], fps: f64, spans: &[Span]) -> Vec<Detected> {
    let mut out = Vec::new();
    let mut last_cut = 0;
    for span in spans {
        let found: Vec<(usize, i64)> = match &span.frozen {
            Some(frozen) => frozen.range(span.range.clone()).map(|(&id, &frame)| (id, frame as i64)).collect(),
            None => {
                let mut cuts = Vec::new();
                detect_range(diffs, fps, &span.params, span.range.start.max(1)..span.range.end, &mut last_cut, &mut cuts);
                cuts.into_iter().map(|id| (id, id as i64 + span.params.cut_offset as i64)).collect()
            }
        };
        if let Some(&(id, _)) = found.last() {
            last_cut = id;
        }
        out.extend(found.into_iter().map(|(id, frame)| Detected { id, frame }));
    }
    out
}

/// Detects cuts on `frames` (a cut on frame `f` comes from `diffs[f - 1]`). `last_cut` carries
/// the minimum scene length across calls.
fn detect_range(diffs: &[f32], fps: f64, p: &Params, frames: Range<usize>, last_cut: &mut usize, out: &mut Vec<usize>) {
    let min_frames = ((p.min_scene_secs as f64 * fps).round() as usize).max(1);
    for frame in frames {
        let Some(&d) = diffs.get(frame - 1) else { break };
        let is_cut = match p.mode {
            DetectMode::Fixed => d >= p.cut_threshold,
            DetectMode::Adaptive => {
                d >= p.adaptive_floor && d >= p.adaptive_ratio * neighbour_mean(diffs, frame - 1, p.adaptive_window)
            }
        };
        if is_cut && frame - *last_cut >= min_frames {
            out.push(frame);
            *last_cut = frame;
        }
    }
}

fn neighbour_mean(diffs: &[f32], i: usize, window: usize) -> f32 {
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(diffs.len());
    let (sum, n) = (lo..hi)
        .filter(|&j| j != i)
        .fold((0.0, 0), |(s, n), j| (s + diffs[j], n + 1));
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// [`resolve`] for cuts detected with one set of settings.
#[cfg(test)]
pub fn resolve_cuts(detected: &[usize], edits: &CutEdits, frame_count: usize, offset: i32) -> ResolvedCuts {
    let detected: Vec<Detected> = detected.iter().map(|&id| Detected { id, frame: id as i64 + offset as i64 }).collect();
    resolve(&detected, edits, frame_count)
}

/// Combine detected cuts with the user's edits into the final, sorted list of cuts.
pub fn resolve(detected: &[Detected], edits: &CutEdits, frame_count: usize) -> ResolvedCuts {
    if frame_count < 2 {
        return ResolvedCuts::default();
    }
    let clamp = |f: i64| f.clamp(1, frame_count as i64 - 1) as usize;
    let mut cuts = Vec::new();
    let mut removed = Vec::new();

    for &Detected { id: d, frame } in detected {
        let auto = clamp(frame);
        if edits.removed.contains(&d) {
            removed.push(auto);
            continue;
        }
        cuts.push(match edits.moved.get(&d) {
            Some(&to) => Cut { id: CutId::Detected(d), frame: clamp(to as i64), source: CutSource::Moved, ghost: Some(auto) },
            None => Cut { id: CutId::Detected(d), frame: auto, source: CutSource::Detected, ghost: None },
        });
    }

    // Moved cuts that detection no longer finds stay where the user put them.
    let detected: HashSet<usize> = detected.iter().map(|d| d.id).collect();
    for (&d, &to) in &edits.moved {
        if !detected.contains(&d) && !edits.removed.contains(&d) {
            cuts.push(Cut { id: CutId::Detected(d), frame: clamp(to as i64), source: CutSource::Moved, ghost: None });
        }
    }
    for (&m, &frame) in &edits.added {
        cuts.push(Cut { id: CutId::Manual(m), frame: clamp(frame as i64), source: CutSource::Manual, ghost: None });
    }

    // At equal frames, prefer the user's cut over a detected one.
    cuts.sort_by_key(|c| (c.frame, c.source == CutSource::Detected));
    cuts.dedup_by_key(|c| c.frame);
    ResolvedCuts { cuts, removed }
}

/// [`build`] with one set of settings.
#[cfg(test)]
pub fn build_scenes(diffs: &[f32], frame_count: usize, cuts: &[Cut], p: &Params) -> Vec<Scene> {
    build(diffs, frame_count, cuts, &sections::plan(p, &Default::default(), frame_count))
}

/// Split `0..frame_count` at `cuts`, drop frames around each cut, and classify each scene.
/// Each scene uses the settings of the span it starts in.
pub fn build(diffs: &[f32], frame_count: usize, cuts: &[Cut], spans: &[Span]) -> Vec<Scene> {
    let bounds: Vec<(CutId, usize)> = std::iter::once((CutId::Start, 0))
        .chain(cuts.iter().map(|c| (c.id, c.frame)).filter(|&(_, f)| f > 0 && f < frame_count))
        .chain(std::iter::once((CutId::Start, frame_count)))
        .collect();

    bounds
        .windows(2)
        .map(|w| {
            let ((id, start), (_, end)) = (w[0], w[1]);
            let p = &sections::span_at(spans, start).params;
            // Only trim at real cuts, not at the very start or end of the video. Dropping more
            // frames than the scene has leaves an empty range, clamped to stay inside the scene.
            let keep_start = if start == 0 { start } else { start.saturating_add(p.drop_after_cut).min(end) };
            let keep_end = if end == frame_count { end } else { end.saturating_sub(p.drop_before_cut).max(start) };
            let keep = keep_start..keep_end.max(keep_start);

            // Differences strictly inside the kept frames: diffs[i] compares frames i and i + 1.
            let inner_end = keep.end.saturating_sub(1).min(diffs.len());
            let motion = median(diffs.get(keep.start..inner_end).unwrap_or(&[]));
            Scene {
                id,
                start,
                end,
                keep,
                motion,
                kind: if motion < p.still_threshold { SceneKind::Still } else { SceneKind::Video },
            }
        })
        .collect()
}

fn median(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, f32::total_cmp).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sections::{Lock, Sections};

    /// 10 frames of motion, a hard cut, 10 static frames, a hard cut, 10 frames of motion.
    fn sample() -> Vec<f32> {
        let mut d = vec![3.0; 9];
        d.push(80.0); // cut before frame 10
        d.extend([0.1; 9]);
        d.push(60.0); // cut before frame 20
        d.extend([3.0; 9]);
        d
    }

    fn params(mode: DetectMode) -> Params {
        Params { mode, min_scene_secs: 0.0, ..Params::default() }
    }

    fn build(p: &Params, edits: &CutEdits) -> Vec<Scene> {
        let d = sample();
        let resolved = resolve_cuts(&[10, 20], edits, d.len() + 1, p.cut_offset);
        build_scenes(&d, d.len() + 1, &resolved.cuts, p)
    }

    fn bounds(scenes: &[Scene]) -> Vec<(usize, usize)> {
        scenes.iter().map(|s| (s.start, s.end)).collect()
    }

    #[test]
    fn finds_hard_cuts_in_both_modes() {
        for mode in [DetectMode::Fixed, DetectMode::Adaptive] {
            assert_eq!(detect_cuts(&sample(), 10.0, &params(mode)), vec![10, 20], "{mode:?}");
        }
    }

    #[test]
    fn min_scene_length_suppresses_close_cuts() {
        let p = Params { min_scene_secs: 1.5, ..params(DetectMode::Fixed) };
        // At 10 fps, 1.5s = 15 frames, so only the second cut survives.
        assert_eq!(detect_cuts(&sample(), 10.0, &p), vec![20]);
    }

    #[test]
    fn classifies_static_scene_as_still() {
        let scenes = build(&Params::default(), &CutEdits::default());
        let kinds: Vec<_> = scenes.iter().map(|s| (s.start, s.end, s.kind)).collect();
        assert_eq!(
            kinds,
            vec![(0, 10, SceneKind::Video), (10, 20, SceneKind::Still), (20, 30, SceneKind::Video)]
        );
    }

    #[test]
    fn deleted_cuts_are_skipped() {
        let mut edits = CutEdits::default();
        edits.delete(CutId::Detected(20));
        let scenes = build(&Params::default(), &edits);
        assert_eq!(bounds(&scenes), vec![(0, 10), (10, 30)]);
    }

    #[test]
    fn offset_moves_detected_boundaries_but_not_ids() {
        let p = Params { cut_offset: -2, ..Params::default() };
        let scenes = build(&p, &CutEdits::default());
        let got: Vec<_> = scenes.iter().map(|s| (s.id, s.start, s.end)).collect();
        assert_eq!(
            got,
            vec![(CutId::Start, 0, 8), (CutId::Detected(10), 8, 18), (CutId::Detected(20), 18, 30)]
        );
    }

    #[test]
    fn user_cuts_ignore_offset() {
        let mut edits = CutEdits::default();
        edits.set_frame(CutId::Detected(10), 12);
        edits.add(25);
        let p = Params { cut_offset: 3, ..Params::default() };
        assert_eq!(bounds(&build(&p, &edits)), vec![(0, 12), (12, 23), (23, 25), (25, 30)]);
    }

    #[test]
    fn moved_cut_survives_losing_detection() {
        let mut edits = CutEdits::default();
        edits.set_frame(CutId::Detected(10), 12);
        // Detection now only finds the second cut.
        let resolved = resolve_cuts(&[20], &edits, 31, 0);
        let got: Vec<_> = resolved.cuts.iter().map(|c| (c.id, c.frame, c.source)).collect();
        assert_eq!(
            got,
            vec![(CutId::Detected(10), 12, CutSource::Moved), (CutId::Detected(20), 20, CutSource::Detected)]
        );
    }

    #[test]
    fn moved_cut_remembers_detected_position() {
        let mut edits = CutEdits::default();
        edits.set_frame(CutId::Detected(10), 12);
        let resolved = resolve_cuts(&[10, 20], &edits, 31, 1);
        assert_eq!(resolved.get(CutId::Detected(10)).unwrap().ghost, Some(11));
        edits.reset(CutId::Detected(10));
        let resolved = resolve_cuts(&[10, 20], &edits, 31, 1);
        assert_eq!(resolved.get(CutId::Detected(10)).unwrap().frame, 11);
    }

    #[test]
    fn user_cut_wins_over_detected_cut_on_same_frame() {
        let mut edits = CutEdits::default();
        let id = edits.add(20);
        let resolved = resolve_cuts(&[10, 20], &edits, 31, 0);
        let ids: Vec<_> = resolved.cuts.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![CutId::Detected(10), id]);
    }

    #[test]
    fn movable_range_stops_at_neighbours() {
        let resolved = resolve_cuts(&[10, 20], &CutEdits::default(), 31, 0);
        assert_eq!(resolved.movable_range(CutId::Detected(10), 31), 1..20);
        assert_eq!(resolved.movable_range(CutId::Detected(20), 31), 11..31);
    }

    #[test]
    fn dropped_frames_only_trim_at_cuts() {
        let p = Params { drop_before_cut: 1, drop_after_cut: 2, ..Params::default() };
        let keeps: Vec<_> = build(&p, &CutEdits::default()).into_iter().map(|s| s.keep).collect();
        assert_eq!(keeps, vec![0..9, 12..19, 22..30]);
    }

    #[test]
    fn scene_can_be_trimmed_to_nothing() {
        let p = Params { drop_before_cut: 6, drop_after_cut: 6, ..Params::default() };
        let scenes = build(&p, &CutEdits::default());
        assert!(scenes[1].keep.is_empty());
    }

    #[test]
    fn cut_ids_round_trip_as_json_map_keys() {
        let map: BTreeMap<CutId, SceneKind> =
            [(CutId::Start, SceneKind::Still), (CutId::Detected(42), SceneKind::Video), (CutId::Manual(3), SceneKind::Still)]
                .into();
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, r#"{"start":"Still","d42":"Video","m3":"Still"}"#);
        assert_eq!(serde_json::from_str::<BTreeMap<CutId, SceneKind>>(&json).unwrap(), map);
        assert!(serde_json::from_str::<CutId>(r#""x1""#).is_err());
    }

    fn bounds_with(sections: &Sections, global: &Params) -> Vec<(CutId, usize, usize)> {
        let d = sample();
        let spans = sections::plan(global, sections, d.len() + 1);
        let detected = detect(&d, 10.0, &spans);
        let resolved = resolve(&detected, &CutEdits::default(), d.len() + 1);
        super::build(&d, d.len() + 1, &resolved.cuts, &spans).iter().map(|s| (s.id, s.start, s.end)).collect()
    }

    #[test]
    fn sections_detect_with_their_own_settings() {
        let global = Params { mode: DetectMode::Fixed, cut_threshold: 70.0, ..params(DetectMode::Fixed) };
        // Globally only the 80 spike counts; a section with a lower threshold also finds the 60 one.
        assert_eq!(bounds_with(&Sections::default(), &global), [(CutId::Start, 0, 10), (CutId::Detected(10), 10, 30)]);
        let mut sections = Sections::default();
        let id = sections.add(15..30).unwrap();
        sections.get_mut(id).unwrap().overrides.cut_threshold = Some(50.0);
        sections.get_mut(id).unwrap().overrides.cut_offset = Some(-1);
        assert_eq!(
            bounds_with(&sections, &global),
            [(CutId::Start, 0, 10), (CutId::Detected(10), 10, 19), (CutId::Detected(20), 19, 30)],
            "the offset only applies to the section's cut"
        );
    }

    #[test]
    fn scenes_use_the_settings_of_the_section_they_start_in() {
        let d = sample();
        let mut sections = Sections::default();
        let id = sections.add(10..20).unwrap();
        sections.get_mut(id).unwrap().overrides.drop_before_cut = Some(2);
        sections.get_mut(id).unwrap().overrides.drop_after_cut = Some(1);
        sections.get_mut(id).unwrap().overrides.still_threshold = Some(0.0);
        let spans = sections::plan(&Params::default(), &sections, d.len() + 1);
        let resolved = resolve(&detect(&d, 10.0, &spans), &CutEdits::default(), d.len() + 1);
        let scenes = super::build(&d, d.len() + 1, &resolved.cuts, &spans);
        let got: Vec<_> = scenes.iter().map(|s| (s.keep.clone(), s.kind)).collect();
        assert_eq!(got, [(0..10, SceneKind::Video), (11..18, SceneKind::Video), (20..30, SceneKind::Video)]);
    }

    #[test]
    fn locked_sections_ignore_setting_changes() {
        let mut sections = Sections::default();
        let id = sections.add(15..30).unwrap();
        let before = bounds_with(&sections, &Params::default());
        // Lock with the cuts and settings the section has now.
        let spans = sections::plan(&Params::default(), &sections, 30);
        let frozen = detect(&sample(), 10.0, &spans).iter().filter(|d| (15..30).contains(&d.id)).map(|d| (d.id, d.frame as usize)).collect();
        sections.get_mut(id).unwrap().lock = Some(Lock { params: spans[1].params.clone(), cuts: frozen });

        // A threshold nothing passes removes every cut, except inside the locked section.
        let strict = Params { mode: DetectMode::Fixed, cut_threshold: 255.0, cut_offset: 5, ..Params::default() };
        assert_eq!(bounds_with(&sections, &strict), [(CutId::Start, 0, 20), (CutId::Detected(20), 20, 30)]);
        assert_eq!(bounds_with(&sections, &Params::default()), before);
    }

    #[test]
    fn locked_cuts_survive_changes_next_to_the_section() {
        // Spikes at frames 10 (weak) and 13 (strong), 3 frames apart; scenes must be 5 frames.
        let mut d = vec![0.5; 29];
        d[9] = 40.0;
        d[12] = 80.0;
        let global = Params { mode: DetectMode::Fixed, cut_threshold: 50.0, ..Params::default() };
        let mut sections = Sections::default();
        let id = sections.add(12..30).unwrap();
        let cuts = |sections: &Sections, global: &Params| {
            detect(&d, 10.0, &sections::plan(global, sections, 30)).iter().map(|c| c.id).collect::<Vec<_>>()
        };
        assert_eq!(cuts(&sections, &global), [13]);
        sections.get_mut(id).unwrap().lock = Some(Lock { params: global.clone(), cuts: [(13, 13)].into() });

        // Now the whole video finds frame 10 too, which would make 13 too close. The lock keeps it.
        let looser = Params { cut_threshold: 30.0, ..global.clone() };
        assert_eq!(cuts(&Sections::default(), &looser), [10]);
        assert_eq!(cuts(&sections, &looser), [10, 13]);
    }

    /// Sweeps every setting and edit across (and beyond) its UI range on videos of awkward
    /// lengths, checking that nothing panics and the scenes stay well-formed.
    #[test]
    fn extreme_settings_never_break_scenes() {
        for frame_count in [0usize, 1, 2, 3, 30, 31] {
            let diffs: Vec<f32> = (0..frame_count.saturating_sub(1)).map(|i| if i % 7 == 6 { 90.0 } else { 0.2 }).collect();
            let cuts = detect_cuts(&diffs, 10.0, &params(DetectMode::Fixed));

            let mut edit_options = vec![CutEdits::default()];
            let mut e = CutEdits::default();
            cuts.iter().step_by(2).for_each(|&c| e.delete(CutId::Detected(c)));
            edit_options.push(e);
            let mut e = CutEdits::default();
            for (i, &c) in cuts.iter().enumerate() {
                e.set_frame(CutId::Detected(c), [0, usize::MAX, c + 3, 1][i % 4]);
            }
            for f in [0, 1, frame_count, usize::MAX] {
                e.add(f);
            }
            e.set_frame(CutId::Detected(999), 5); // orphaned move
            edit_options.push(e);

            for cut_offset in [-1000, -60, -7, -1, 0, 1, 7, 60, 1000] {
                for drop in [0usize, 1, 3, 6, 7, 30, 600, usize::MAX] {
                    for edits in &edit_options {
                        let p = Params { cut_offset, drop_before_cut: drop, drop_after_cut: drop / 2, ..Params::default() };
                        let resolved = resolve_cuts(&cuts, edits, frame_count, cut_offset);
                        let scenes = build_scenes(&diffs, frame_count, &resolved.cuts, &p);
                        let ctx = format!("frames={frame_count} offset={cut_offset} drop={drop} edits={edits:?}");

                        assert!(resolved.cuts.windows(2).all(|w| w[0].frame < w[1].frame), "{ctx}");
                        assert!(!scenes.is_empty(), "{ctx}");
                        assert_eq!(scenes[0].start, 0, "{ctx}");
                        assert_eq!(scenes.last().unwrap().end, frame_count, "{ctx}");
                        for (a, b) in scenes.iter().zip(scenes.iter().skip(1)) {
                            assert_eq!(a.end, b.start, "scenes not contiguous: {ctx}");
                        }
                        for s in &scenes {
                            assert!(s.start <= s.keep.start && s.keep.start <= s.keep.end && s.keep.end <= s.end, "{ctx}: {s:?}");
                            assert!(frame_count == 0 || s.middle() < frame_count, "{ctx}: {s:?}");
                        }
                        for c in &resolved.cuts {
                            let r = resolved.movable_range(c.id, frame_count);
                            assert!(r.start < r.end && r.contains(&c.frame), "{ctx}: {c:?} {r:?}");
                        }
                    }
                }
            }
        }
    }
}
