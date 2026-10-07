//! Turning per-frame difference scores into scenes. Pure and cheap to re-run.

use std::collections::HashSet;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectMode {
    /// Cut wherever the difference exceeds a fixed threshold.
    Fixed,
    /// Cut where the difference is much larger than its neighbours. Copes
    /// better with fast camera motion, where every frame differs a lot.
    Adaptive,
}

#[derive(Debug, Clone)]
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
    /// Cuts closer together than this are ignored.
    pub min_scene_secs: f32,
    /// Scenes whose median frame-to-frame difference is below this are treated as stills.
    pub still_threshold: f32,
    /// Moves every cut by this many frames. 0 starts the new scene on the first frame after
    /// the spike; negative values start it earlier, positive values later.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneKind {
    Video,
    Still,
}

#[derive(Debug, Clone)]
pub struct Scene {
    /// Detected (un-offset) cut that starts this scene; 0 for the first scene. Stable while
    /// the offset/drop settings change, so it's used to key user choices.
    pub id: usize,
    /// Scene boundaries after applying the cut offset: frames `start..end`.
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

/// Frame indices where a new scene starts (never includes 0).
pub fn detect_cuts(diffs: &[f32], fps: f64, p: &Params) -> Vec<usize> {
    let min_frames = ((p.min_scene_secs as f64 * fps).round() as usize).max(1);
    let mut cuts = Vec::new();
    let mut last_cut = 0usize;

    for (i, &d) in diffs.iter().enumerate() {
        let is_cut = match p.mode {
            DetectMode::Fixed => d >= p.cut_threshold,
            DetectMode::Adaptive => {
                d >= p.adaptive_floor && d >= p.adaptive_ratio * neighbour_mean(diffs, i, p.adaptive_window)
            }
        };
        let frame = i + 1;
        if is_cut && frame - last_cut >= min_frames {
            cuts.push(frame);
            last_cut = frame;
        }
    }
    cuts
}

fn neighbour_mean(diffs: &[f32], i: usize, window: usize) -> f32 {
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(diffs.len());
    let (sum, n) = (lo..hi)
        .filter(|&j| j != i)
        .fold((0.0, 0), |(s, n), j| (s + diffs[j], n + 1));
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Split `0..frame_count` at `cuts` (skipping any in `merged`), apply the offset and
/// dropped frames from `p`, and classify each scene.
pub fn build_scenes(diffs: &[f32], frame_count: usize, cuts: &[usize], merged: &HashSet<usize>, p: &Params) -> Vec<Scene> {
    // (id, shifted position) for each surviving cut, kept strictly increasing and inside the video.
    let mut bounds: Vec<(usize, usize)> = vec![(0, 0)];
    for &cut in cuts.iter().filter(|c| !merged.contains(c) && frame_count >= 2) {
        let shifted = (cut as i64 + p.cut_offset as i64).clamp(1, frame_count as i64 - 1) as usize;
        if shifted > bounds.last().unwrap().1 {
            bounds.push((cut, shifted));
        }
    }
    bounds.push((frame_count, frame_count));

    bounds
        .windows(2)
        .map(|w| {
            let ((id, start), (_, end)) = (w[0], w[1]);
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

    fn build(p: &Params, merged: &[usize]) -> Vec<Scene> {
        let d = sample();
        build_scenes(&d, d.len() + 1, &[10, 20], &merged.iter().copied().collect(), p)
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
        let scenes = build(&Params::default(), &[]);
        let kinds: Vec<_> = scenes.iter().map(|s| (s.start, s.end, s.kind)).collect();
        assert_eq!(
            kinds,
            vec![(0, 10, SceneKind::Video), (10, 20, SceneKind::Still), (20, 30, SceneKind::Video)]
        );
    }

    #[test]
    fn merged_cuts_are_skipped() {
        let scenes = build(&Params::default(), &[20]);
        assert_eq!(scenes.len(), 2);
        assert_eq!((scenes[1].start, scenes[1].end), (10, 30));
    }

    #[test]
    fn offset_moves_boundaries_but_not_ids() {
        let p = Params { cut_offset: -2, ..Params::default() };
        let scenes = build(&p, &[]);
        let got: Vec<_> = scenes.iter().map(|s| (s.id, s.start, s.end)).collect();
        assert_eq!(got, vec![(0, 0, 8), (10, 8, 18), (20, 18, 30)]);
    }

    #[test]
    fn dropped_frames_only_trim_at_cuts() {
        let p = Params { drop_before_cut: 1, drop_after_cut: 2, ..Params::default() };
        let keeps: Vec<_> = build(&p, &[]).into_iter().map(|s| s.keep).collect();
        assert_eq!(keeps, vec![0..9, 12..19, 22..30]);
    }

    #[test]
    fn scene_can_be_trimmed_to_nothing() {
        let p = Params { drop_before_cut: 6, drop_after_cut: 6, ..Params::default() };
        let scenes = build(&p, &[]);
        assert!(scenes[1].keep.is_empty());
    }

    /// Sweeps every setting across (and beyond) its UI range on videos of awkward lengths,
    /// checking that nothing panics and the scenes stay well-formed.
    #[test]
    fn extreme_settings_never_break_scenes() {
        for frame_count in [0usize, 1, 2, 3, 30, 31] {
            let diffs: Vec<f32> = (0..frame_count.saturating_sub(1)).map(|i| if i % 7 == 6 { 90.0 } else { 0.2 }).collect();
            let cuts = detect_cuts(&diffs, 10.0, &params(DetectMode::Fixed));
            let merge_options: [Vec<usize>; 2] = [vec![], cuts.iter().copied().step_by(2).collect()];
            for cut_offset in [-1000, -60, -7, -1, 0, 1, 7, 60, 1000] {
                for drop in [0usize, 1, 3, 6, 7, 30, 600, usize::MAX] {
                    for merged in &merge_options {
                        let p = Params { cut_offset, drop_before_cut: drop, drop_after_cut: drop / 2, ..Params::default() };
                        let merged = merged.iter().copied().collect();
                        let scenes = build_scenes(&diffs, frame_count, &cuts, &merged, &p);
                        let ctx = format!("frames={frame_count} offset={cut_offset} drop={drop}");

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
                    }
                }
            }
        }
    }
}
