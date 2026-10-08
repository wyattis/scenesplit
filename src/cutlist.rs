//! The scene list as a file for other tools: CSV or JSON for scripts and spreadsheets, or an
//! EDL (CMX 3600 edit decision list) for doing the cutting in Resolve, Premiere and the like.

use std::fmt::Write;
use std::ops::Range;

use serde::Serialize;

use crate::scenes::SceneKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Csv,
    Json,
    Edl,
}

impl Format {
    pub const ALL: [Self; 3] = [Self::Csv, Self::Json, Self::Edl];

    pub fn ext(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Json => "json",
            Self::Edl => "edl",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Csv => "CSV (spreadsheets)…",
            Self::Json => "JSON (scripts)…",
            Self::Edl => "EDL (Resolve, Premiere)…",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Csv | Self::Json => "Every scene, with its frames and times, and whether it's exported.",
            Self::Edl => "The exported scenes back to back on a timeline, to import into a video editor and cut there. Uses the exported frames, so dropped and trimmed frames are left out.",
        }
    }
}

/// One scene as it appears in the scene list.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// 1-based, as shown in the scene list and used in exported file names.
    pub number: usize,
    pub kind: SceneKind,
    pub scene: Range<usize>,
    /// Frames exported.
    pub keep: Range<usize>,
    pub still_frame: usize,
    pub export: bool,
    /// Number of an earlier scene this one looks like.
    pub looks_like: Option<usize>,
}

pub fn write(format: Format, rows: &[Row], fps: f64, video: &str, has_audio: bool) -> String {
    match format {
        Format::Csv => csv(rows, fps),
        Format::Json => json(rows, fps, video),
        Format::Edl => edl(rows, fps, video, has_audio),
    }
}

fn kind(k: SceneKind) -> &'static str {
    match k {
        SceneKind::Video => "video",
        SceneKind::Still => "still",
    }
}

fn secs(frame: usize, fps: f64) -> f64 {
    (frame as f64 / fps * 1000.0).round() / 1000.0
}

fn csv(rows: &[Row], fps: f64) -> String {
    let mut out = String::from("scene,kind,export,start_frame,end_frame,start_time,end_time,duration,scene_start_frame,scene_end_frame,still_frame,looks_like\n");
    for r in rows {
        let looks_like = r.looks_like.map(|n| n.to_string()).unwrap_or_default();
        let (start, end) = (secs(r.keep.start, fps), secs(r.keep.end, fps));
        writeln!(
            out,
            "{},{},{},{},{},{start:.3},{end:.3},{:.3},{},{},{},{looks_like}",
            r.number,
            kind(r.kind),
            r.export,
            r.keep.start,
            r.keep.end,
            end - start,
            r.scene.start,
            r.scene.end,
            r.still_frame,
        )
        .unwrap();
    }
    out
}

#[derive(Serialize)]
struct JsonList<'a> {
    video: &'a str,
    fps: f64,
    /// Frame ranges are start..end, end not included.
    scenes: Vec<JsonScene>,
}

#[derive(Serialize)]
struct JsonScene {
    scene: usize,
    kind: &'static str,
    export: bool,
    start_frame: usize,
    end_frame: usize,
    start_time: f64,
    end_time: f64,
    scene_start_frame: usize,
    scene_end_frame: usize,
    still_frame: usize,
    still_time: f64,
    looks_like: Option<usize>,
}

fn json(rows: &[Row], fps: f64, video: &str) -> String {
    let scenes = rows
        .iter()
        .map(|r| JsonScene {
            scene: r.number,
            kind: kind(r.kind),
            export: r.export,
            start_frame: r.keep.start,
            end_frame: r.keep.end,
            start_time: secs(r.keep.start, fps),
            end_time: secs(r.keep.end, fps),
            scene_start_frame: r.scene.start,
            scene_end_frame: r.scene.end,
            still_frame: r.still_frame,
            still_time: secs(r.still_frame, fps),
            looks_like: r.looks_like,
        })
        .collect();
    serde_json::to_string_pretty(&JsonList { video, fps, scenes }).unwrap() + "\n"
}

/// The exported scenes as events laid end to end on a timeline starting at 01:00:00:00, the
/// usual start for a programme. Timecode counts frames at the frame rate rounded to a whole
/// number (non-drop-frame), which is how editors number the frames of a 29.97 fps file too.
fn edl(rows: &[Row], fps: f64, video: &str, has_audio: bool) -> String {
    let base = (fps.round() as usize).max(1);
    let tc = |f: usize| format!("{:02}:{:02}:{:02}:{:02}", f / base / 3600, f / base / 60 % 60, f / base % 60, f % base);
    // Avid and Premiere write AA/V for picture with stereo sound.
    let tracks = if has_audio { "AA/V" } else { "V" };
    let title: String = video.chars().filter(|c| !c.is_control()).collect();
    let mut out = format!("TITLE: {title}\r\nFCM: NON-DROP FRAME\r\n\r\n");
    let mut record = 3600 * base;
    let events = rows.iter().filter(|r| r.export && !r.keep.is_empty());
    for (i, r) in events.enumerate() {
        let len = r.keep.len();
        write!(
            out,
            "{:03}  AX       {tracks:<4}  C        {} {} {} {}\r\n* FROM CLIP NAME: {title}\r\n* SCENE {}\r\n\r\n",
            i + 1,
            tc(r.keep.start),
            tc(r.keep.end),
            tc(record),
            tc(record + len),
            r.number,
        )
        .unwrap();
        record += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<Row> {
        vec![
            Row { number: 1, kind: SceneKind::Video, scene: 0..100, keep: 0..98, still_frame: 49, export: true, looks_like: None },
            Row { number: 2, kind: SceneKind::Still, scene: 100..130, keep: 102..130, still_frame: 120, export: false, looks_like: None },
            Row { number: 3, kind: SceneKind::Still, scene: 130..9100, keep: 132..9100, still_frame: 140, export: true, looks_like: Some(2) },
        ]
    }

    #[test]
    fn csv_lists_every_scene() {
        let text = write(Format::Csv, &rows(), 25.0, "a.mp4", true);
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[1], "1,video,true,0,98,0.000,3.920,3.920,0,100,49,");
        assert_eq!(lines[3], "3,still,true,132,9100,5.280,364.000,358.720,130,9100,140,2");
    }

    #[test]
    fn json_lists_every_scene() {
        let v: serde_json::Value = serde_json::from_str(&write(Format::Json, &rows(), 25.0, "a.mp4", true)).unwrap();
        assert_eq!(v["video"], "a.mp4");
        assert_eq!(v["scenes"].as_array().unwrap().len(), 3);
        assert_eq!(v["scenes"][1]["export"], false);
        assert_eq!(v["scenes"][2]["looks_like"], 2);
        assert_eq!(v["scenes"][2]["still_time"], 5.6);
    }

    #[test]
    fn edl_lays_exported_scenes_end_to_end() {
        let text = write(Format::Edl, &rows(), 29.97, "my video.mp4", true);
        let events: Vec<_> = text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())).collect();
        assert_eq!(
            events,
            [
                "001  AX       AA/V  C        00:00:00:00 00:00:03:08 01:00:00:00 01:00:03:08",
                "002  AX       AA/V  C        00:00:04:12 00:05:03:10 01:00:03:08 01:05:02:06",
            ]
        );
        assert!(text.starts_with("TITLE: my video.mp4\r\nFCM: NON-DROP FRAME\r\n"));
        assert!(text.contains("* FROM CLIP NAME: my video.mp4\r\n* SCENE 3\r\n"));
        assert!(write(Format::Edl, &rows(), 25.0, "x", false).contains(" V     C "));
    }
}
