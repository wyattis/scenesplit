//! Gradual transitions between clips: fades through black and dissolves (crossfades).
//!
//! Neither shows up as a spike in the frame differences, so they're found from other
//! measurements: fades from brightness, dissolves from the dip in detail they cause.

use std::ops::Range;

use crate::analysis::Analysis;
use crate::sections::{Span, span_at};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    Fade,
    Dissolve,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub kind: TransitionKind,
    /// The frames of the transition, which can be dropped from the scenes on either side.
    pub frames: Range<usize>,
    /// First frame of the next scene. `None` for a fade at the very start or end of the video.
    pub cut: Option<usize>,
}

/// Brightness must change by more than this (plus 1% of itself) per frame to count as part
/// of a fade, so ordinary slow drift in a shot isn't eaten.
const RAMP_STEP: f32 = 1.0;
/// Longest fade ramp on either side of the dark frames.
const MAX_RAMP_SECS: f64 = 2.0;

/// Fades through black: runs of frames darker than the span's `black_level`, with the
/// darkening and brightening ramps around them. Spans with `detect_fades` off are skipped.
pub fn find_fades(luma: &[f32], fps: f64, spans: &[Span]) -> Vec<Transition> {
    let n = luma.len();
    let max_ramp = (MAX_RAMP_SECS * fps).round() as usize;
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let p = &span_at(spans, i).params;
        if luma[i] >= p.black_level {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < n && luma[i] < p.black_level {
            i += 1;
        }
        let run_end = i;
        if !p.detect_fades {
            continue;
        }
        let steeper = |brighter: f32, darker: f32| brighter > darker + RAMP_STEP + 0.01 * darker;
        // A frame is part of the fade-out if it's clearly darker than the one before it, and of
        // the fade-in if the one after it is clearly brighter.
        let mut start = run_start;
        while start >= 2 && run_start - start < max_ramp && steeper(luma[start - 2], luma[start - 1]) {
            start -= 1;
        }
        let mut end = run_end;
        while end + 1 < n && end - run_end < max_ramp && steeper(luma[end + 1], luma[end]) {
            end += 1;
        }
        let cut = (run_start > 0 && run_end < n).then_some(run_end);
        out.push(Transition { kind: TransitionKind::Fade, frames: start..end, cut });
    }
    out
}

/// Shortest and longest dissolves looked for.
const MIN_DISSOLVE_SECS: f64 = 0.2;
const MAX_DISSOLVE_SECS: f64 = 2.5;
/// The dip must go below this fraction of the detail at both ends.
const DIP_DEPTH: f32 = 0.8;
/// Largest mean relative difference from the model curve.
const MAX_FIT_ERROR: f32 = 0.12;
/// The two ends must look different: at least this many hash bits, or this much colour change.
const MIN_HASH_BITS: u32 = 12;
const MIN_COLOR_CHANGE: i32 = 30;

/// Dissolves anywhere in the video, regardless of settings (see [`select`]).
///
/// A frame in a dissolve is `(1 - t)·A + t·B`. Two unrelated images' detail adds roughly like
/// variances, so the detail of the blend is `(1 - t)²·a + t²·b`: a dip below both clips. This
/// looks for stretches whose sharpness follows that curve between two different-looking ends.
pub fn find_dissolves(a: &Analysis) -> Vec<Transition> {
    let n = a.sharpness.len();
    let s = median5(&a.sharpness);
    let min_len = ((MIN_DISSOLVE_SECS * a.fps).round() as usize).max(3);
    let max_len = ((MAX_DISSOLVE_SECS * a.fps).round() as usize).max(min_len + 1);
    let mut out: Vec<Transition> = Vec::new();

    for m in 1..n.saturating_sub(1) {
        // Candidates: the lowest point within `min_len` on each side, clearly below what's around it.
        let near = m.saturating_sub(min_len)..(m + min_len + 1).min(n);
        if near.clone().any(|j| s[j] < s[m] || (s[j] == s[m] && j < m)) {
            continue;
        }
        let far = m.saturating_sub(max_len)..(m + max_len + 1).min(n);
        let (left, right) = (far.start..m, m + 1..far.end);
        let peak = |r: Range<usize>| r.map(|j| s[j]).fold(0.0f32, f32::max);
        if s[m] >= DIP_DEPTH * peak(left).min(peak(right)) {
            continue;
        }
        if out.last().is_some_and(|t| t.frames.contains(&m)) {
            continue;
        }

        // Where the dissolve starts and ends: the best fit of flat, curve, flat over the whole
        // neighbourhood, so the ends land where the shape changes.
        let mut best: Option<(usize, usize, f32)> = None;
        for start in far.start..m.saturating_sub(1) {
            for end in m + 2..far.end {
                let len = end - start;
                if len < min_len || len > max_len || s[m] >= DIP_DEPTH * s[start].min(s[end]) {
                    continue;
                }
                let limit = best.map_or(f32::INFINITY, |b| b.2);
                if let Some(err) = piecewise_error(&s[far.clone()], start - far.start, end - far.start, limit) {
                    best = Some((start, end, err));
                }
            }
        }
        let Some((start, end, _)) = best else { continue };
        if fit_error(&s[start..=end], MAX_FIT_ERROR).is_none() {
            continue;
        }

        let hash_bits = (a.hash[start] ^ a.hash[end]).count_ones();
        let color_change: i32 = (0..3).map(|c| (a.color[start][c] as i32 - a.color[end][c] as i32).abs()).sum();
        if hash_bits < MIN_HASH_BITS && color_change < MIN_COLOR_CHANGE {
            continue;
        }
        // A hard cut inside means it isn't a dissolve (the spike detector handles it).
        let inner = &a.diffs[start..end.min(a.diffs.len())];
        let typical = median(inner);
        if inner.iter().any(|&d| d >= 20.0 && d >= 4.0 * typical) {
            continue;
        }
        out.push(Transition { kind: TransitionKind::Dissolve, frames: start..end + 1, cut: Some((start + end + 1) / 2) });
    }
    out
}

/// Total difference between `s` and: `s[start]` up to `start`, the dissolve curve to `end`, then
/// `s[end]`. `None` once it reaches `limit`.
fn piecewise_error(s: &[f32], start: usize, end: usize, limit: f32) -> Option<f32> {
    let (a, b) = (s[start], s[end]);
    let len = (end - start) as f32;
    let mut err = 0.0;
    for (i, &v) in s.iter().enumerate() {
        let model = match i {
            i if i <= start => a,
            i if i >= end => b,
            i => {
                let t = (i - start) as f32 / len;
                (1.0 - t).powi(2) * a + t * t * b
            }
        };
        err += (v - model).abs();
        if err >= limit {
            return None;
        }
    }
    Some(err)
}

/// Mean relative difference between `s` and the dissolve curve through its ends, or `None`
/// as soon as it's certain to exceed `limit`.
fn fit_error(s: &[f32], limit: f32) -> Option<f32> {
    let (a, b) = (s[0], s[s.len() - 1]);
    let last = (s.len() - 1) as f32;
    // Σ(1 - t)² = Σt² over the samples, so the model's total is known up front.
    let squares: f32 = (0..s.len()).map(|i| (i as f32 / last).powi(2)).sum();
    let total = squares * (a + b);
    if total <= 0.0 {
        return None;
    }
    let mut err = 0.0;
    for (i, &v) in s.iter().enumerate() {
        let t = i as f32 / last;
        err += (v - ((1.0 - t).powi(2) * a + t * t * b)).abs();
        if err > limit * total {
            return None;
        }
    }
    Some(err / total)
}

fn median5(xs: &[f32]) -> Vec<f32> {
    (0..xs.len()).map(|i| median(&xs[i.saturating_sub(2)..(i + 3).min(xs.len())])).collect()
}

fn median(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, f32::total_cmp).1
}

/// The transitions that apply with the current settings: fades, plus the dissolves in spans
/// that detect them (and that don't overlap a fade). Sorted by start.
pub fn select(a: &Analysis, dissolves: &[Transition], spans: &[Span]) -> Vec<Transition> {
    let mut out = find_fades(&a.luma, a.fps, spans);
    let fades = out.clone();
    for d in dissolves {
        let overlaps_fade = fades.iter().any(|f| f.frames.start < d.frames.end && d.frames.start < f.frames.end);
        if span_at(spans, d.cut.unwrap_or(d.frames.start)).params.detect_dissolves && !overlaps_fade {
            out.push(d.clone());
        }
    }
    out.sort_by_key(|t| t.frames.start);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::Params;
    use crate::sections::{Sections, plan};

    fn spans(p: Params, n: usize) -> Vec<Span> {
        plan(&p, &Sections::default(), n)
    }

    #[test]
    fn finds_fade_through_black_with_its_ramps() {
        //            clip      fade out      black     fade in        clip
        let luma = [120.0, 121.0, 90.0, 40.0, 2.0, 0.0, 30.0, 70.0, 100.0, 101.0, 100.5];
        let fades = find_fades(&luma, 25.0, &spans(Params::default(), luma.len()));
        assert_eq!(fades, [Transition { kind: TransitionKind::Fade, frames: 2..8, cut: Some(6) }]);
    }

    #[test]
    fn fades_at_the_ends_trim_without_cutting() {
        let luma = [0.0, 0.0, 50.0, 100.0, 100.0, 100.0, 60.0, 3.0];
        let fades = find_fades(&luma, 25.0, &spans(Params::default(), luma.len()));
        let got: Vec<_> = fades.iter().map(|f| (f.frames.clone(), f.cut)).collect();
        assert_eq!(got, [(0..3, None), (6..8, None)]);
    }

    #[test]
    fn fade_settings_apply_per_span() {
        let luma = [100.0, 100.0, 5.0, 100.0, 100.0];
        assert_eq!(find_fades(&luma, 25.0, &spans(Params::default(), 5)).len(), 1);
        assert!(find_fades(&luma, 25.0, &spans(Params { black_level: 4.0, ..Params::default() }, 5)).is_empty());
        assert!(find_fades(&luma, 25.0, &spans(Params { detect_fades: false, ..Params::default() }, 5)).is_empty());
    }

    /// Two shots (detail 1300 and 500) with a dissolve over frames 40..65, plus noise.
    fn dissolve_analysis() -> Analysis {
        let n = 120;
        let mut a = Analysis::from_diffs(vec![3.0; n - 1], 25.0);
        for i in 0..n {
            let t = ((i as f32 - 40.0) / 24.0).clamp(0.0, 1.0);
            let noise = [0.0, 15.0, -10.0, 5.0, -20.0][i % 5];
            a.sharpness[i] = (1.0 - t).powi(2) * 1300.0 + t * t * 500.0 + noise;
            a.hash[i] = if t < 0.5 { 0x1818_1818_1818_1818 } else { 0x7e7e_7e7e_3f83_9898 };
        }
        a
    }

    #[test]
    fn finds_a_dissolve_from_the_dip_in_detail() {
        let found = find_dissolves(&dissolve_analysis());
        assert_eq!(found.len(), 1, "{found:?}");
        let d = &found[0];
        assert!(d.frames.start.abs_diff(40) <= 2 && d.frames.end.abs_diff(65) <= 2, "{d:?}");
        assert!(d.cut.unwrap().abs_diff(52) <= 2, "{d:?}");
    }

    #[test]
    fn a_blurry_moment_in_one_shot_is_not_a_dissolve() {
        let mut a = dissolve_analysis();
        a.hash.fill(0x1818_1818_1818_1818);
        assert!(find_dissolves(&a).is_empty());
    }

    #[test]
    fn a_hard_cut_inside_the_dip_is_not_a_dissolve() {
        let mut a = dissolve_analysis();
        a.diffs[50] = 80.0;
        assert!(find_dissolves(&a).is_empty());
    }

    #[test]
    fn dissolve_search_is_fast_on_long_noisy_videos() {
        // An hour at 30 fps of jittery detail.
        let n = 108_000;
        let mut a = Analysis::from_diffs(vec![5.0; n - 1], 30.0);
        let mut x = 12345u32;
        for i in 0..n {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            a.sharpness[i] = 800.0 + (x >> 16) as f32 % 400.0;
            a.hash[i] = x as u64;
        }
        let started = std::time::Instant::now();
        find_dissolves(&a);
        assert!(started.elapsed().as_secs_f32() < 20.0, "{:?}", started.elapsed());
    }
}
