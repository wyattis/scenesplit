//! Thin wrappers around the `ffmpeg` / `ffprobe` command-line tools.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Where `program` (`ffmpeg` or `ffprobe`) will be run from: next to our own executable if
/// it's there (release archives ship it that way), otherwise whatever is on PATH.
pub fn resolve(program: &str) -> (PathBuf, Source) {
    let exe = std::env::current_exe().ok();
    resolve_in(exe.as_deref().and_then(Path::parent), program)
}

fn resolve_in(app_dir: Option<&Path>, program: &str) -> (PathBuf, Source) {
    let bundled = app_dir
        .map(|dir| dir.join(format!("{program}{}", std::env::consts::EXE_SUFFIX)))
        .filter(|p| p.is_file());
    match bundled {
        Some(path) => (path, Source::Bundled),
        None => (PathBuf::from(program), Source::Path),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Bundled,
    Path,
}

/// Build a command for `ffmpeg`/`ffprobe` (see [`resolve`]) that won't flash a console
/// window on Windows.
pub fn command(program: &str) -> Command {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut cmd = Command::new(resolve(program).0);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

const MISSING: &str = "couldn't run ffprobe: put ffmpeg and ffprobe next to scenesplit, or install ffmpeg and add it to PATH";

/// Checks that both tools run. Returns e.g. "ffmpeg 9.0.2 (bundled)".
pub fn check() -> Result<String> {
    let mut version = String::new();
    for program in ["ffprobe", "ffmpeg"] {
        let out = command(program).arg("-version").output().map_err(|_| anyhow::anyhow!(
            "{program} not found: put ffmpeg and ffprobe next to scenesplit, or install ffmpeg and add it to PATH"
        ))?;
        if !out.status.success() {
            bail!("{program} -version failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        // "ffmpeg version 9.0.2 Copyright ..." -> "9.0.2"
        version = String::from_utf8_lossy(&out.stdout).split_whitespace().nth(2).unwrap_or("?").to_owned();
    }
    let source = match resolve("ffmpeg").1 {
        Source::Bundled => "bundled",
        Source::Path => "from PATH",
    };
    Ok(format!("ffmpeg {version} ({source})"))
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
        .context(MISSING)?;
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

/// Grab `count` consecutive RGBA frames starting at frame `start` (at `fps`, matching the
/// analysis frame numbering), scaled to `width` x `height`. Returns fewer frames if the
/// video ends first.
pub fn grab_frames_rgba(path: &Path, start: usize, count: usize, fps: f64, width: u32, height: u32) -> Result<Vec<Vec<u8>>> {
    let time = start as f64 / fps;
    let mut child = command("ffmpeg")
        .args(["-v", "error", "-ss", &format!("{time:.4}"), "-i"])
        .arg(path)
        .args(["-an", "-frames:v", &count.to_string()])
        .args(["-vf", &format!("fps={fps},scale={width}:{height}")])
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to run ffmpeg")?;
    let mut buf = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut buf)?;
    child.wait()?;
    let frame_len = (width * height * 4) as usize;
    let frames: Vec<Vec<u8>> = buf.chunks_exact(frame_len).map(<[u8]>::to_vec).collect();
    if frames.is_empty() {
        bail!("could not decode frame {start}");
    }
    Ok(frames)
}

/// Generates a 9 second, 25 fps test video: 3s moving pattern, 3s solid colour, 3s moving fractal.
#[cfg(test)]
/// 25 fps, 17 s: clip A (moving) crossfades into B (moving) over 3-4 s, B fades through black
/// into a still C over 6-7 s, then hard cuts to D (moving) at 10 s and to C again at 14 s.
/// So: a dissolve centred on frame 87, a fade with black at 162, cuts at 250 and 350.
pub fn make_transition_video(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("scenesplit-test").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let src = |f: &str| ["-f".to_owned(), "lavfi".to_owned(), "-i".to_owned(), format!("{f}{}size=320x180:rate=25", if f.contains('=') { ":" } else { "=" })];
    let status = command("ffmpeg")
        .args(["-v", "error", "-y"])
        .args(src("testsrc2=duration=4"))
        .args(src("mandelbrot"))
        .args(src("smptehdbars=duration=3"))
        .args(src("life=mold=10:ratio=0.5:death_color=#203040:life_color=#e0c040"))
        .args(["-filter_complex", concat!(
            "[1]trim=duration=4,setpts=PTS-STARTPTS[b];[3]trim=duration=4,setpts=PTS-STARTPTS,format=yuv420p,fps=25[d];",
            "[0]format=yuv420p,fps=25[a];[2]format=yuv420p,fps=25,split[c1][c2];[b]format=yuv420p,fps=25[b2];",
            "[a][b2]xfade=transition=fade:duration=1:offset=3[ab];",
            "[ab][c1]xfade=transition=fadeblack:duration=1:offset=6[abc];",
            "[abc][d][c2]concat=n=3:v=1[v]",
        )])
        .args(["-map", "[v]", "-pix_fmt", "yuv420p"])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());
    input
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The filmstrip relies on frame N from `grab_frames_rgba` being frame N of the analysis.
    #[test]
    #[ignore = "needs ffmpeg on PATH"]
    fn grabbed_frames_line_up_with_analysis_numbering() {
        let input = make_test_video("grab");
        let info = probe(&input).unwrap();
        // The solid-colour section starts exactly at frame 75 (3s at 25fps).
        let frames = grab_frames_rgba(&input, 70, 10, info.fps, 64, 36).unwrap();
        assert_eq!(frames.len(), 10);
        let is_solid = |rgba: &[u8]| rgba.chunks(4).all(|p| p.iter().zip(&rgba[..4]).all(|(a, b)| a.abs_diff(*b) <= 3));
        let solid: Vec<bool> = frames.iter().map(|f| is_solid(f)).collect();
        assert_eq!(solid, [false, false, false, false, false, true, true, true, true, true]);

        // Asking past the end returns what exists rather than failing.
        let tail = grab_frames_rgba(&input, 220, 13, info.fps, 64, 36).unwrap();
        assert_eq!(tail.len(), 5);
    }
}

#[cfg(test)]
mod lookup_tests {
    use super::*;

    #[test]
    fn falls_back_to_path_when_nothing_is_bundled() {
        // Test binaries live in target/*/deps, where no ffmpeg is bundled.
        assert_eq!(resolve("ffmpeg"), (PathBuf::from("ffmpeg"), Source::Path));
    }

    #[test]
    fn prefers_a_copy_next_to_the_app() {
        let dir = std::env::temp_dir().join("scenesplit-test").join("bundled");
        std::fs::create_dir_all(&dir).unwrap();
        let bundled = dir.join(format!("ffprobe{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&bundled, b"").unwrap();
        assert_eq!(resolve_in(Some(&dir), "ffprobe"), (bundled, Source::Bundled));
        assert_eq!(resolve_in(Some(&dir), "ffmpeg"), (PathBuf::from("ffmpeg"), Source::Path));
    }

    #[test]
    #[ignore = "needs ffmpeg on PATH"]
    fn check_reports_version_and_source() {
        let version = check().unwrap();
        assert!(version.starts_with("ffmpeg ") && version.ends_with("(from PATH)"), "{version}");
    }
}
