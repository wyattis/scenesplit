//! Writing scenes out as video clips or still images.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};

use crate::ffmpeg;
use crate::scenes::SceneKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutMode {
    /// Re-encode: frame-accurate cuts, slower.
    Exact,
    /// Stream copy: fast and lossless, but cuts snap to keyframes.
    Fast,
}

#[derive(Debug, Clone)]
pub struct ExportItem {
    pub index: usize,
    pub kind: SceneKind,
    pub start: f64,
    pub end: f64,
    /// Time of the frame saved for a still.
    pub still: f64,
}

pub fn export_all(
    input: &Path,
    out_dir: &Path,
    items: &[ExportItem],
    mode: CutMode,
    cancel: &Arc<AtomicBool>,
    mut on_progress: impl FnMut(usize),
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir).context("could not create output folder")?;
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    let input_ext = input.extension().and_then(|s| s.to_str()).unwrap_or("mp4");

    let mut written = Vec::new();
    for (done, item) in items.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let n = item.index + 1;
        let out = match item.kind {
            SceneKind::Still => {
                let out = out_dir.join(format!("{stem}-{n:03}.png"));
                export_still(input, item.still, &out)?;
                out
            }
            SceneKind::Video => {
                let ext = if mode == CutMode::Fast { input_ext } else { "mp4" };
                let out = out_dir.join(format!("{stem}-{n:03}.{ext}"));
                export_clip(input, item.start, item.end, mode, &out)?;
                out
            }
        };
        written.push(out);
        on_progress(done + 1);
    }
    Ok(written)
}

fn export_still(input: &Path, time: f64, out: &Path) -> Result<()> {
    let mut cmd = ffmpeg::command("ffmpeg");
    cmd.args(["-v", "error", "-y", "-ss", &format!("{time:.3}"), "-i"])
        .arg(input)
        .args(["-frames:v", "1"])
        .arg(out);
    run(cmd)
}

fn export_clip(input: &Path, start: f64, end: f64, mode: CutMode, out: &Path) -> Result<()> {
    let mut cmd = ffmpeg::command("ffmpeg");
    cmd.args(["-v", "error", "-y", "-ss", &format!("{start:.3}"), "-i"])
        .arg(input)
        .args(["-t", &format!("{:.3}", end - start)]);
    match mode {
        CutMode::Exact => cmd.args(["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "aac"]),
        CutMode::Fast => cmd.args(["-c", "copy", "-avoid_negative_ts", "make_zero"]),
    };
    cmd.arg(out);
    run(cmd)
}

fn run(mut cmd: std::process::Command) -> Result<()> {
    let out = cmd.output().context("failed to run ffmpeg")?;
    if !out.status.success() {
        bail!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}
