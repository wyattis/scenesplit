//! Turning per-frame difference scores into scenes. Pure and cheap to re-run.

use std::collections::HashSet;

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
    /// First frame (inclusive).
    pub start: usize,
    /// One past the last frame.
    pub end: usize,
    /// Median frame-to-frame difference inside the scene.
    pub motion: f32,
    pub kind: SceneKind,
}

impl Scene {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn middle(&self) -> usize {
        self.start + self.len() / 2
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

/// Split `0..frame_count` at `cuts` (skipping any in `merged`) and classify each scene.
pub fn build_scenes(
    diffs: &[f32],
    frame_count: usize,
    cuts: &[usize],
    merged: &HashSet<usize>,
    still_threshold: f32,
) -> Vec<Scene> {
    let bounds: Vec<usize> = std::iter::once(0)
        .chain(cuts.iter().copied().filter(|c| !merged.contains(c)))
        .chain(std::iter::once(frame_count))
        .collect();

    bounds
        .windows(2)
        .map(|w| {
            let (start, end) = (w[0], w[1]);
            // Differences strictly inside the scene: between frames start..end.
            let motion = median(&diffs[start..end.saturating_sub(1).max(start)]);
            Scene {
                start,
                end,
                motion,
                kind: if motion < still_threshold { SceneKind::Still } else { SceneKind::Video },
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
        let d = sample();
        let scenes = build_scenes(&d, d.len() + 1, &[10, 20], &HashSet::new(), 0.5);
        let kinds: Vec<_> = scenes.iter().map(|s| (s.start, s.end, s.kind)).collect();
        assert_eq!(
            kinds,
            vec![(0, 10, SceneKind::Video), (10, 20, SceneKind::Still), (20, 30, SceneKind::Video)]
        );
    }

    #[test]
    fn merged_cuts_are_skipped() {
        let d = sample();
        let scenes = build_scenes(&d, d.len() + 1, &[10, 20], &HashSet::from([20]), 0.5);
        assert_eq!(scenes.len(), 2);
        assert_eq!((scenes[1].start, scenes[1].end), (10, 30));
    }
}
