//! One decoding pass over the video that measures every frame.
//!
//! This is the only slow step. Everything downstream (cut, fade and dissolve detection,
//! still classification, picking still frames, finding duplicates) works on these
//! measurements, so tweaking settings is instant.

use std::io::Read;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};

use crate::ffmpeg::{self, VideoInfo};

/// Bump whenever the analysis output would change, to invalidate cached results.
pub const ANALYSIS_VERSION: u32 = 2;

/// Frames are downscaled to this size before comparing.
const W: usize = 160;
const H: usize = 90;

#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// `diffs[i]` is the mean absolute RGB difference (0-255) between frame `i` and `i + 1`.
    pub diffs: Vec<f32>,
    /// Frame rate the analysis was sampled at; frame `i` is at `i / fps` seconds.
    pub fps: f64,
    /// Per frame: mean brightness (0-255).
    pub luma: Vec<f32>,
    /// Per frame: how much fine detail it has (variance of the Laplacian). Blurry frames and
    /// frames in the middle of a fade score lower.
    pub sharpness: Vec<f32>,
    /// Per frame: 64-bit difference hash of its layout, for finding similar frames.
    pub hash: Vec<u64>,
    /// Per frame: mean colour.
    pub color: Vec<[u8; 3]>,
}

impl Analysis {
    /// Just difference scores, everything else neutral. For tests of the diff-based logic.
    #[cfg(test)]
    pub fn from_diffs(diffs: Vec<f32>, fps: f64) -> Self {
        let n = diffs.len() + 1;
        Self {
            diffs,
            fps,
            luma: vec![128.0; n],
            sharpness: vec![0.0; n],
            hash: vec![0; n],
            color: vec![[128; 3]; n],
        }
    }

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
    let mut measure = Measure::new(expected);
    let mut frame = vec![0u8; W * H * 3];
    while stdout.read_exact(&mut frame).is_ok() {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            bail!("cancelled");
        }
        let n = measure.push(&frame);
        if n % 30 == 0 {
            on_progress((n as f32 / expected as f32).min(1.0));
        }
    }
    child.wait()?;

    if measure.a.luma.is_empty() {
        bail!("ffmpeg produced no frames");
    }
    on_progress(1.0);
    measure.a.fps = info.fps;
    Ok(measure.a)
}

/// Accumulates measurements frame by frame.
struct Measure {
    a: Analysis,
    prev: Option<Vec<u8>>,
}

impl Measure {
    fn new(capacity: usize) -> Self {
        let a = Analysis {
            diffs: Vec::with_capacity(capacity),
            fps: 0.0,
            luma: Vec::with_capacity(capacity),
            sharpness: Vec::with_capacity(capacity),
            hash: Vec::with_capacity(capacity),
            color: Vec::with_capacity(capacity),
        };
        Self { a, prev: None }
    }

    /// Adds an rgb24 frame of W×H; returns the number of frames so far.
    fn push(&mut self, rgb: &[u8]) -> usize {
        let a = &mut self.a;
        if let Some(prev) = &self.prev {
            a.diffs.push(mean_abs_diff(prev, rgb));
        }
        let gray: Vec<f32> = rgb.chunks_exact(3).map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect();
        a.luma.push(gray.iter().sum::<f32>() / gray.len() as f32);
        a.sharpness.push(laplacian_variance(&gray));
        a.hash.push(dhash(&gray));
        let mut sum = [0u64; 3];
        for p in rgb.chunks_exact(3) {
            (0..3).for_each(|c| sum[c] += p[c] as u64);
        }
        let px = (W * H) as u64;
        a.color.push([(sum[0] / px) as u8, (sum[1] / px) as u8, (sum[2] / px) as u8]);
        self.prev = Some(rgb.to_vec());
        a.luma.len()
    }
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    let sum: u64 = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as u64).sum();
    sum as f32 / a.len() as f32
}

fn laplacian_variance(gray: &[f32]) -> f32 {
    let (mut sum, mut sq, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            let i = y * W + x;
            let l = (4.0 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - W] - gray[i + W]) as f64;
            sum += l;
            sq += l * l;
            n += 1.0;
        }
    }
    (sq / n - (sum / n).powi(2)) as f32
}

/// Difference hash: shrink to 9×8 and record whether each cell is brighter than its right neighbour.
fn dhash(gray: &[f32]) -> u64 {
    let mut cells = [[0.0f32; 9]; 8];
    for y in 0..H {
        for x in 0..W {
            cells[y * 8 / H][x * 9 / W] += gray[y * W + x];
        }
    }
    let mut hash = 0u64;
    for row in &cells {
        for x in 0..8 {
            hash = hash << 1 | (row[x] > row[x + 1]) as u64;
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{self, ExportItem, ExportSettings};
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
        assert_eq!((analysis.luma.len(), analysis.sharpness.len(), analysis.hash.len()), (225, 225, 225));
        let [r, g, b] = analysis.color[100];
        assert!(r < 3 && g < 3 && b.abs_diff(128) < 4, "navy: {:?}", analysis.color[100]);
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
                frame: s.start,
                start: analysis.frame_time(s.start),
                end: analysis.frame_time(s.end),
                still: analysis.frame_time(s.still_frame),
            })
            .collect();
        let settings = ExportSettings::default();
        let jobs = export::plan(&input, &dir.join("out"), &items, &settings).unwrap();
        export::run(&input, &jobs, &settings, &AtomicBool::new(false), |_, _| {}).unwrap();
        let out: Vec<_> = jobs.iter().map(|j| &j.out).collect();
        let names: Vec<_> = out.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["input-001.mp4", "input-002.png", "input-003.mp4"]);
        assert!(out.iter().all(|p| p.metadata().unwrap().len() > 0));
    }
}
