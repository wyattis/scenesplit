//! Thin wrappers around the `ffmpeg` / `ffprobe` command-line tools.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Build a command that won't flash a console window on Windows.
pub fn command(program: &str) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
    pub has_audio: bool,
}

impl VideoInfo {
    pub fn estimated_frames(&self) -> usize {
        (self.duration * self.fps).round().max(1.0) as usize
    }
}

#[derive(Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
    format: ProbeFormat,
}

#[derive(Deserialize)]
struct ProbeStream {
    codec_type: String,
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
}

#[derive(Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

fn parse_rate(s: &str) -> Option<f64> {
    let (n, d) = s.split_once('/').unwrap_or((s, "1"));
    let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
    (n > 0.0 && d > 0.0).then(|| n / d)
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let out = command("ffprobe")
        .args(["-v", "error", "-show_entries"])
        .arg("stream=codec_type,width,height,avg_frame_rate,r_frame_rate:format=duration")
        .args(["-of", "json"])
        .arg(path)
        .output()
        .context("failed to run ffprobe (is it on PATH?)")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let parsed: ProbeOutput = serde_json::from_slice(&out.stdout).context("bad ffprobe output")?;

    let video = parsed
        .streams
        .iter()
        .find(|s| s.codec_type == "video")
        .context("no video stream found")?;
    let fps = video
        .avg_frame_rate
        .as_deref()
        .and_then(parse_rate)
        .or_else(|| video.r_frame_rate.as_deref().and_then(parse_rate))
        .context("could not determine frame rate")?;

    Ok(VideoInfo {
        width: video.width.unwrap_or(0),
        height: video.height.unwrap_or(0),
        fps,
        duration: parsed
            .format
            .duration
            .as_deref()
            .and_then(|d| d.parse().ok())
            .unwrap_or(0.0),
        has_audio: parsed.streams.iter().any(|s| s.codec_type == "audio"),
    })
}

/// Grab a single RGBA frame at `time` seconds, scaled to `width` x `height`.
pub fn grab_frame_rgba(path: &Path, time: f64, width: u32, height: u32) -> Result<Vec<u8>> {
    let mut child = command("ffmpeg")
        .args(["-v", "error", "-ss", &format!("{time:.3}"), "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-vf", &format!("scale={width}:{height}")])
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to run ffmpeg")?;
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    child.stdout.take().unwrap().read_to_end(&mut buf)?;
    child.wait()?;
    if buf.len() != (width * height * 4) as usize {
        bail!("could not decode frame at {time:.2}s");
    }
    Ok(buf)
}

/// Generates a 9 second, 25 fps test video: 3s moving pattern, 3s solid colour, 3s moving fractal.
#[cfg(test)]
pub fn make_test_video(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("scenesplit-test").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let status = command("ffmpeg")
        .args(["-v", "error", "-y"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=3"])
        .args(["-f", "lavfi", "-i", "color=c=navy:size=320x180:rate=25:duration=3"])
        .args(["-f", "lavfi", "-i", "mandelbrot=size=320x180:rate=25"])
        .args(["-filter_complex", "[2]trim=duration=3,setpts=PTS-STARTPTS[m];[0][1][m]concat=n=3:v=1[v]"])
        .args(["-map", "[v]", "-pix_fmt", "yuv420p"])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());
    input
}
