//! One decoding pass over the video that produces a per-frame difference score.
//!
//! This is the only slow step. Everything downstream (cut detection, still
//! detection) works on the cached scores, so tweaking thresholds is instant.

use std::io::Read;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};

use crate::ffmpeg::{self, VideoInfo};

/// Bump whenever the analysis output would change, to invalidate cached results.
pub const ANALYSIS_VERSION: u32 = 1;

/// Frames are downscaled to this size before comparing.
const W: usize = 160;
const H: usize = 90;

pub struct Analysis {
    /// `diffs[i]` is the mean absolute RGB difference (0-255) between frame `i` and `i + 1`.
    pub diffs: Vec<f32>,
    /// Frame rate the analysis was sampled at; frame `i` is at `i / fps` seconds.
    pub fps: f64,
}

impl Analysis {
    pub fn frame_count(&self) -> usize {
        self.diffs.len() + 1
    }

    pub fn frame_time(&self, frame: usize) -> f64 {
        frame as f64 / self.fps
    }
}

pub fn analyze(
    path: &Path,
    info: &VideoInfo,
    cancel: &Arc<AtomicBool>,
    mut on_progress: impl FnMut(f32),
) -> Result<Analysis> {
    // The fps filter forces a constant frame rate so frame index maps cleanly
    // to a timestamp, even for variable-frame-rate phone/screen recordings.
    let mut child = ffmpeg::command("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-an", "-vf", &format!("fps={},scale={W}:{H}", info.fps)])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("couldn't run ffmpeg: put ffmpeg and ffprobe next to scenesplit, or install ffmpeg and add it to PATH")?;
    let mut stdout = child.stdout.take().unwrap();

    let expected = info.estimated_frames();
    let mut prev = vec![0u8; W * H * 3];
    let mut cur = vec![0u8; W * H * 3];
    let mut diffs = Vec::with_capacity(expected);
    let mut frames = 0usize;

    while stdout.read_exact(&mut cur).is_ok() {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            bail!("cancelled");
        }
        if frames > 0 {
            diffs.push(mean_abs_diff(&prev, &cur));
        }
        frames += 1;
        std::mem::swap(&mut prev, &mut cur);
        if frames % 30 == 0 {
            on_progress((frames as f32 / expected as f32).min(1.0));
        }
    }
    child.wait()?;

    if frames == 0 {
        bail!("ffmpeg produced no frames");
    }
    on_progress(1.0);
    Ok(Analysis { diffs, fps: info.fps })
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    let sum: u64 = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as u64).sum();
    sum as f32 / a.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{self, CutMode, ExportItem};
    use crate::scenes::{self, CutEdits, Params, SceneKind};

    /// End-to-end check against real ffmpeg: moving clip, static frame, moving clip.
    /// Run with `cargo test -- --ignored`.
    #[test]
    #[ignore = "needs ffmpeg on PATH"]
    fn splits_synthetic_video() {
        let input = ffmpeg::make_test_video("analysis");
        let dir = input.parent().unwrap();

        let info = ffmpeg::probe(&input).unwrap();
        let analysis = analyze(&input, &info, &Arc::new(AtomicBool::new(false)), |_| {}).unwrap();
        let params = Params::default();
        let cuts = scenes::detect_cuts(&analysis.diffs, analysis.fps, &params);
        assert_eq!(cuts, vec![75, 150]);

        let resolved = scenes::resolve_cuts(&cuts, &CutEdits::default(), analysis.frame_count(), params.cut_offset);
        let found = scenes::build_scenes(&analysis.diffs, analysis.frame_count(), &resolved.cuts, &params);
        let kinds: Vec<_> = found.iter().map(|s| s.kind).collect();
        assert_eq!(kinds, vec![SceneKind::Video, SceneKind::Still, SceneKind::Video]);

        let items: Vec<_> = found
            .iter()
            .enumerate()
            .map(|(index, s)| ExportItem {
                index,
                kind: s.kind,
                start: analysis.frame_time(s.start),
                end: analysis.frame_time(s.end),
            })
            .collect();
        let out = export::export_all(&input, &dir.join("out"), &items, CutMode::Exact, &Arc::new(AtomicBool::new(false)), |_| {}).unwrap();
        let names: Vec<_> = out.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["input-001.mp4", "input-002.png", "input-003.mp4"]);
        assert!(out.iter().all(|p| p.metadata().unwrap().len() > 0));
    }
}
