//! Writing scenes out as video clips or still images, several at a time.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::ffmpeg;
use crate::scenes::SceneKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClipFormat {
    /// Stream copy: fast and lossless, but cuts snap to keyframes.
    Original,
    Mp4,
    WebM,
    Gif,
}

impl ClipFormat {
    pub const ALL: [Self; 4] = [Self::Mp4, Self::WebM, Self::Gif, Self::Original];

    pub fn label(self) -> &'static str {
        match self {
            Self::Original => "Original (copy, fast)",
            Self::Mp4 => "MP4",
            Self::WebM => "WebM",
            Self::Gif => "GIF",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Original => "Copies the video without re-encoding: fast and lossless, but cuts snap to the nearest keyframe, and size and quality don't apply.",
            Self::Mp4 => "H.264 and AAC. Frame-accurate cuts; plays everywhere.",
            Self::WebM => "VP9 and Opus. Frame-accurate cuts; smaller files, slower to encode.",
            Self::Gif => "Animated GIF at 15 frames per second, without sound. Best for short clips.",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StillFormat {
    Png,
    Jpg,
    WebP,
}

impl StillFormat {
    pub const ALL: [Self; 3] = [Self::Png, Self::Jpg, Self::WebP];

    pub fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::Jpg => "JPG",
            Self::WebP => "WebP",
        }
    }

    fn ext(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpg => "jpg",
            Self::WebP => "webp",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quality {
    Best,
    Good,
    Small,
}

impl Quality {
    pub const ALL: [Self; 3] = [Self::Best, Self::Good, Self::Small];

    pub fn label(self) -> &'static str {
        match self {
            Self::Best => "Best",
            Self::Good => "Good",
            Self::Small => "Small files",
        }
    }

    /// Picks the value for this quality from `[best, good, small]`.
    fn pick<T: Copy>(self, values: [T; 3]) -> T {
        values[self as usize]
    }
}

/// Choices for the shorter side's maximum length.
pub const MAX_SIZES: [u32; 6] = [2160, 1440, 1080, 720, 480, 360];

/// Placeholders for [`ExportSettings::name_pattern`], with what they stand for.
pub const PLACEHOLDERS: [(&str, &str); 5] = [
    ("{name}", "the video's file name"),
    ("{n}", "scene number: 001, 002, …"),
    ("{time}", "start time: 00-01-23.456"),
    ("{frame}", "first frame number"),
    ("{kind}", "video or still"),
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportSettings {
    /// File name without extension; see [`PLACEHOLDERS`].
    pub name_pattern: String,
    pub clips: ClipFormat,
    pub stills: StillFormat,
    /// Scale down so the shorter side is at most this many pixels.
    pub max_size: Option<u32>,
    pub quality: Quality,
    /// Files exported at the same time.
    pub jobs: usize,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            name_pattern: "{name}-{n}".into(),
            clips: ClipFormat::Mp4,
            stills: StillFormat::Png,
            max_size: None,
            quality: Quality::Best,
            jobs: default_jobs(),
        }
    }
}

/// Half the cores, up to 4: each encode is multithreaded already, so more mostly helps with
/// many short files.
pub fn default_jobs() -> usize {
    (max_jobs() / 2).clamp(1, 4)
}

pub fn max_jobs() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn settings_path() -> Option<PathBuf> {
    // Tests mustn't read or overwrite the user's settings.
    if cfg!(test) {
        return None;
    }
    dirs::config_dir().map(|d| d.join("scenesplit").join("export.json"))
}

/// The export settings saved last time, or the defaults.
pub fn load_settings() -> ExportSettings {
    settings_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

pub fn save_settings(settings: &ExportSettings) -> Result<()> {
    let Some(path) = settings_path() else { return Ok(()) };
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, serde_json::to_string_pretty(settings)?).with_context(|| format!("could not write {}", path.display()))
}

#[derive(Debug, Clone)]
pub struct ExportItem {
    pub index: usize,
    pub kind: SceneKind,
    /// First exported frame, for `{frame}`.
    pub frame: usize,
    pub start: f64,
    pub end: f64,
    /// Time of the frame saved for a still.
    pub still: f64,
}

/// One file to write.
#[derive(Debug, Clone)]
pub struct Job {
    pub item: ExportItem,
    pub out: PathBuf,
}

/// Where each item goes, or why the name pattern can't be used.
pub fn plan(input: &Path, out_dir: &Path, items: &[ExportItem], s: &ExportSettings) -> Result<Vec<Job>, String> {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    let input_ext = input.extension().and_then(|s| s.to_str()).unwrap_or("mp4");
    let mut seen = std::collections::HashMap::new();
    let mut jobs = Vec::with_capacity(items.len());
    for item in items {
        let ext = match item.kind {
            SceneKind::Still => s.stills.ext(),
            SceneKind::Video => match s.clips {
                ClipFormat::Original => input_ext,
                ClipFormat::Mp4 => "mp4",
                ClipFormat::WebM => "webm",
                ClipFormat::Gif => "gif",
            },
        };
        let name = format!("{}.{ext}", file_stem(&s.name_pattern, stem, item)?);
        // Windows and macOS file names ignore case.
        if let Some(other) = seen.insert(name.to_lowercase(), item.index) {
            return Err(format!("Scenes {} and {} would both be saved as \"{name}\". Add {{n}} to the name.", other + 1, item.index + 1));
        }
        jobs.push(Job { item: item.clone(), out: out_dir.join(name) });
    }
    Ok(jobs)
}

/// `pattern` with its placeholders filled in, made safe to use as a file name.
fn file_stem(pattern: &str, video: &str, item: &ExportItem) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = pattern;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let close = rest[open..].find('}').ok_or("A { in the name has no matching }.")? + open;
        match &rest[open..=close] {
            "{name}" => out.push_str(video),
            "{n}" => out.push_str(&format!("{:03}", item.index + 1)),
            "{time}" => out.push_str(&time_label(item.start)),
            "{frame}" => out.push_str(&item.frame.to_string()),
            "{kind}" => out.push_str(if item.kind == SceneKind::Still { "still" } else { "video" }),
            other => return Err(format!("Unknown placeholder {other} in the name.")),
        }
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    let safe: String = out.chars().map(|c| if c.is_control() || r#"<>:"/\|?*"#.contains(c) { '_' } else { c }).collect();
    // Windows drops trailing dots and spaces.
    let safe = safe.trim().trim_end_matches('.');
    if safe.is_empty() {
        return Err("The name is empty.".into());
    }
    Ok(safe.to_owned())
}

/// `83.456` → `00-01-23.456`; sorts in time order.
fn time_label(secs: f64) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!("{:02}-{:02}-{:02}.{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

#[derive(Debug, Clone, PartialEq)]
pub enum JobState {
    Waiting,
    /// Fraction done.
    Running(f32),
    Done,
    Failed(String),
    Cancelled,
}

/// Runs `jobs`, `settings.jobs` at a time, reporting each change of a job's state. A failed
/// job doesn't stop the others; cancelling stops them all. Errors only if the output folder
/// can't be created.
pub fn run(
    input: &Path,
    jobs: &[Job],
    settings: &ExportSettings,
    cancel: &AtomicBool,
    on_update: impl Fn(usize, JobState) + Sync,
) -> Result<()> {
    for dir in jobs.iter().filter_map(|j| j.out.parent()) {
        std::fs::create_dir_all(dir).context("could not create output folder")?;
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..settings.jobs.clamp(1, jobs.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    if cancel.load(Ordering::Relaxed) {
                        on_update(i, JobState::Cancelled);
                        continue;
                    }
                    on_update(i, JobState::Running(0.0));
                    let state = match run_one(input, job, settings, cancel, |p| on_update(i, JobState::Running(p))) {
                        Ok(()) => JobState::Done,
                        Err(_) if cancel.load(Ordering::Relaxed) => JobState::Cancelled,
                        Err(e) => JobState::Failed(format!("{e:#}")),
                    };
                    if state != JobState::Done {
                        // Don't leave a half-written file behind.
                        let _ = std::fs::remove_file(&job.out);
                    }
                    on_update(i, state);
                }
            });
        }
    });
    Ok(())
}

fn run_one(input: &Path, job: &Job, s: &ExportSettings, cancel: &AtomicBool, on_progress: impl Fn(f32)) -> Result<()> {
    let mut child = ffmpeg::command("ffmpeg")
        .args(args(input, job, s))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to run ffmpeg")?;
    // Read errors on another thread so a full stderr pipe can't stall ffmpeg.
    let mut stderr = child.stderr.take().unwrap();
    let errors = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    let duration = job.item.end - job.item.start;
    // ffmpeg reports progress about twice a second.
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("cancelled");
        }
        let line = line?;
        if let Some(us) = line.strip_prefix("out_time_us=").and_then(|v| v.parse::<f64>().ok())
            && duration > 0.0
        {
            on_progress((us / 1e6 / duration).clamp(0.0, 1.0) as f32);
        }
    }
    let status = child.wait()?;
    let errors = errors.join().unwrap_or_default();
    if !status.success() {
        bail!("ffmpeg failed: {}", errors.trim());
    }
    Ok(())
}

/// ffmpeg arguments for one job.
fn args(input: &Path, job: &Job, s: &ExportSettings) -> Vec<OsString> {
    let item = &job.item;
    let start = if item.kind == SceneKind::Still { item.still } else { item.start };
    let scale = s.max_size.map(scale_filter);
    let q = s.quality;
    let mut a: Vec<OsString> = Vec::new();
    let mut push = |args: &[&str]| a.extend(args.iter().map(OsString::from));
    push(&["-v", "error", "-y", "-progress", "pipe:1", "-nostats", "-ss", &format!("{start:.3}"), "-i"]);
    a.push(input.into());
    let mut push = |args: &[&str]| a.extend(args.iter().map(OsString::from));
    match item.kind {
        SceneKind::Still => {
            push(&["-frames:v", "1", "-update", "1"]);
            if let Some(scale) = &scale {
                push(&["-vf", scale]);
            }
            match s.stills {
                StillFormat::Png => {}
                StillFormat::Jpg => push(&["-q:v", q.pick(["2", "4", "8"])]),
                StillFormat::WebP => push(&["-c:v", "libwebp", "-quality", q.pick(["95", "80", "60"])]),
            }
        }
        SceneKind::Video => {
            push(&["-t", &format!("{:.3}", item.end - item.start)]);
            if matches!(s.clips, ClipFormat::Mp4 | ClipFormat::WebM)
                && let Some(scale) = &scale
            {
                push(&["-vf", scale]);
            }
            match s.clips {
                ClipFormat::Original => push(&["-c", "copy", "-avoid_negative_ts", "make_zero"]),
                ClipFormat::Mp4 => push(&[
                    "-c:v", "libx264", "-crf", q.pick(["18", "23", "28"]), "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-movflags", "+faststart",
                ]),
                ClipFormat::WebM => push(&[
                    "-c:v", "libvpx-vp9", "-crf", q.pick(["24", "32", "40"]), "-b:v", "0", "-row-mt", "1",
                    "-deadline", "good", "-cpu-used", "4", "-pix_fmt", "yuv420p", "-c:a", "libopus",
                ]),
                ClipFormat::Gif => {
                    let colors = q.pick([256, 128, 64]);
                    let scale = scale.map(|s| format!("{s},")).unwrap_or_default();
                    // A palette made for this clip looks far better than the default one.
                    let vf = format!(
                        "fps=15,{scale}split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
                    );
                    push(&["-an", "-vf", &vf, "-loop", "0"]);
                }
            }
        }
    }
    a.push(job.out.as_os_str().into());
    a
}

/// Scales down (never up) so the shorter side is at most `max` pixels, keeping the aspect
/// ratio and even dimensions.
fn scale_filter(max: u32) -> String {
    format!("scale=w='if(lte(iw,ih),trunc(min(iw,{max})/2)*2,-2)':h='if(lte(iw,ih),-2,trunc(min(ih,{max})/2)*2)'")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn item(index: usize, kind: SceneKind, start: f64) -> ExportItem {
        ExportItem { index, kind, frame: (start * 25.0) as usize, start, end: start + 2.0, still: start + 1.0 }
    }

    fn names(pattern: &str, s: &ExportSettings, items: &[ExportItem]) -> Result<Vec<String>, String> {
        let s = ExportSettings { name_pattern: pattern.into(), ..s.clone() };
        let jobs = plan(Path::new("dir/My clip.mov"), Path::new("out"), items, &s)?;
        Ok(jobs.iter().map(|j| j.out.file_name().unwrap().to_string_lossy().into_owned()).collect())
    }

    #[test]
    fn names_fill_in_placeholders_and_extensions() {
        let items = [item(0, SceneKind::Video, 0.0), item(11, SceneKind::Still, 3723.5)];
        let d = ExportSettings::default();
        assert_eq!(names("{name}-{n}", &d, &items).unwrap(), ["My clip-001.mp4", "My clip-012.png"]);
        assert_eq!(
            names("{n} {kind} {time} f{frame}", &d, &items).unwrap(),
            ["001 video 00-00-00.000 f0.mp4", "012 still 01-02-03.500 f93087.png"]
        );
        let s = ExportSettings { clips: ClipFormat::Original, stills: StillFormat::WebP, ..d.clone() };
        assert_eq!(names("{n}", &s, &items).unwrap(), ["001.mov", "012.webp"]);
        assert_eq!(names("a/b:{n}?", &d, &items).unwrap()[0], "a_b_001_.mp4");
    }

    #[test]
    fn bad_names_are_explained() {
        let items = [item(0, SceneKind::Video, 0.0), item(1, SceneKind::Video, 5.0)];
        let d = ExportSettings::default();
        assert!(names("{name}", &d, &items).unwrap_err().contains("Scenes 1 and 2"));
        assert!(names("{nmae}-{n}", &d, &items).unwrap_err().contains("{nmae}"));
        assert!(names("{n", &d, &items).unwrap_err().contains("no matching"));
        assert!(names(" .", &d, &items).unwrap_err().contains("empty"));
        // Different kinds get different extensions, so they can share a name.
        assert!(names("x", &d, &[item(0, SceneKind::Video, 0.0), item(1, SceneKind::Still, 5.0)]).is_ok());
    }

    #[test]
    fn arguments_follow_the_settings() {
        let text = |kind, s: &ExportSettings| {
            let job = Job { item: item(0, kind, 10.0), out: "o".into() };
            args(Path::new("in.mp4"), &job, s).iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>().join(" ")
        };
        let d = ExportSettings::default();
        let mp4 = text(SceneKind::Video, &d);
        assert!(mp4.contains("-ss 10.000 -i in.mp4 -t 2.000") && mp4.contains("-crf 18") && !mp4.contains("scale"), "{mp4}");
        assert!(text(SceneKind::Still, &d).contains("-ss 11.000"), "stills seek to the still frame");

        let small = ExportSettings { quality: Quality::Small, max_size: Some(720), ..d.clone() };
        let mp4 = text(SceneKind::Video, &small);
        assert!(mp4.contains("-crf 28 ") && mp4.contains("min(ih,720)"), "{mp4}");
        let copy = ExportSettings { clips: ClipFormat::Original, ..small.clone() };
        assert!(!text(SceneKind::Video, &copy).contains("scale"), "copying can't scale");
        let gif = text(SceneKind::Video, &ExportSettings { clips: ClipFormat::Gif, ..small.clone() });
        assert!(gif.contains("fps=15,scale=") && gif.contains("max_colors=64"), "{gif}");
        let jpg = ExportSettings { stills: StillFormat::Jpg, ..small };
        assert!(text(SceneKind::Still, &jpg).contains("-q:v 8"));
    }

    /// Exports every format against real ffmpeg. Run with `cargo test -- --ignored`.
    #[test]
    #[ignore = "needs ffmpeg on PATH"]
    fn exports_every_format() {
        let input = ffmpeg::make_test_video("export");
        let input = &input;
        let out_dir = input.parent().unwrap().join("formats");
        let mut jobs = Vec::new();
        for (i, clips) in ClipFormat::ALL.into_iter().enumerate() {
            let s = ExportSettings { clips, max_size: Some(120), ..Default::default() };
            jobs.extend(plan(input, &out_dir, &[item(i, SceneKind::Video, 0.5)], &s).unwrap().into_iter().map(|j| (j, s.clone())));
        }
        for (i, stills) in StillFormat::ALL.into_iter().enumerate() {
            let s = ExportSettings { stills, max_size: Some(120), ..Default::default() };
            jobs.extend(plan(input, &out_dir, &[item(10 + i, SceneKind::Still, 3.5)], &s).unwrap().into_iter().map(|j| (j, s.clone())));
        }
        std::thread::scope(|scope| {
            for (job, s) in &jobs {
                scope.spawn(move || {
                    let updates = Mutex::new(Vec::new());
                    run(input, std::slice::from_ref(job), s, &AtomicBool::new(false), |_, state| updates.lock().unwrap().push(state)).unwrap();
                    let updates = updates.into_inner().unwrap();
                    assert_eq!(updates.last(), Some(&JobState::Done), "{}: {updates:?}", job.out.display());
                    let info = ffmpeg::probe(&job.out).unwrap();
                    // The test video is 320×180 (213.3 rounds to the nearest even width); copying can't scale.
                    let copied = s.clips == ClipFormat::Original && job.item.kind == SceneKind::Video;
                    assert_eq!((info.width, info.height), if copied { (320, 180) } else { (214, 120) }, "{}", job.out.display());
                });
            }
        });

        // A failing job is reported and doesn't stop the rest; cancelling marks them all cancelled.
        let s = ExportSettings { jobs: 1, ..Default::default() };
        let mut two = plan(input, &out_dir, &[item(0, SceneKind::Video, 0.5), item(1, SceneKind::Still, 1.0)], &s).unwrap();
        two[0].out = out_dir.join("unknown.format");
        let updates = Mutex::new(Vec::new());
        run(input, &two, &s, &AtomicBool::new(false), |i, state| updates.lock().unwrap().push((i, state))).unwrap();
        let updates = updates.into_inner().unwrap();
        assert!(updates.iter().any(|u| matches!(u, (0, JobState::Failed(e)) if e.contains("ffmpeg failed"))), "{updates:?}");
        assert_eq!(updates.last(), Some(&(1, JobState::Done)));
        assert!(!two[0].out.exists());

        let updates = Mutex::new(Vec::new());
        run(input, &two, &s, &AtomicBool::new(true), |i, state| updates.lock().unwrap().push((i, state))).unwrap();
        assert_eq!(updates.into_inner().unwrap(), [(0, JobState::Cancelled), (1, JobState::Cancelled)]);
    }
}
