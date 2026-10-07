//! Video-only preview player.
//!
//! A background ffmpeg process decodes small RGBA frames from the seek point onwards into a
//! bounded channel. While paused, the channel just fills up and ffmpeg blocks, so resuming
//! is instant. Seeking restarts the decoder.

use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};
use std::time::{Duration, Instant};

use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};

use crate::ffmpeg::{self, VideoInfo};

const PREVIEW_W: u32 = 640;
const BUFFERED_FRAMES: usize = 8;

pub struct Player {
    path: PathBuf,
    fps: f64,
    size: [usize; 2],
    frame_count: usize,
    texture: Option<TextureHandle>,
    decoder: Option<Decoder>,
    /// Frame currently on screen.
    shown: usize,
    /// While playing: when playback (re)started and from which frame. `None` = paused.
    clock: Option<(Instant, usize)>,
    /// Frame to show while paused.
    paused_at: usize,
    /// Playback loops within this range when set.
    pub loop_range: Option<Range<usize>>,
}

struct Decoder {
    rx: Receiver<(usize, Vec<u8>)>,
    last_received: Option<usize>,
    finished: bool,
}

impl Player {
    pub fn new(path: &Path, info: &VideoInfo, fps: f64, frame_count: usize) -> Self {
        let h = if info.width == 0 {
            PREVIEW_W * 9 / 16
        } else {
            ((PREVIEW_W as f64 * info.height as f64 / info.width as f64).round() as u32 / 2 * 2).max(2)
        };
        let mut player = Self {
            path: path.to_path_buf(),
            fps,
            size: [PREVIEW_W as usize, h as usize],
            frame_count,
            texture: None,
            decoder: None,
            shown: 0,
            clock: None,
            paused_at: 0,
            loop_range: None,
        };
        player.seek(0);
        player
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn current_frame(&self) -> usize {
        self.shown
    }

    pub fn is_playing(&self) -> bool {
        self.clock.is_some()
    }

    pub fn aspect(&self) -> f32 {
        self.size[0] as f32 / self.size[1] as f32
    }

    pub fn texture(&self) -> Option<&TextureHandle> {
        self.texture.as_ref()
    }

    pub fn seek(&mut self, frame: usize) {
        let frame = frame.min(self.frame_count.saturating_sub(1));
        self.decoder = Some(Decoder::start(&self.path, frame, self.fps, self.size));
        self.paused_at = frame;
        if self.clock.is_some() {
            self.clock = Some((Instant::now(), frame));
        }
    }

    pub fn step(&mut self, delta: i64) {
        self.pause();
        // `seek` clamps to the last frame.
        self.seek(self.shown.saturating_add_signed(delta as isize));
    }

    pub fn play(&mut self) {
        if self.clock.is_some() {
            return;
        }
        // `paused_at` rather than `shown`: right after a seek, the new frame may not have arrived yet.
        let pos = self.paused_at;
        let range = self.loop_range.clone().unwrap_or(0..self.frame_count);
        if pos + 1 >= range.end || pos < range.start {
            self.seek(range.start);
            self.clock = Some((Instant::now(), range.start));
        } else {
            self.clock = Some((Instant::now(), pos));
        }
    }

    pub fn pause(&mut self) {
        if self.clock.take().is_some() {
            self.paused_at = self.shown;
        }
    }

    pub fn toggle(&mut self) {
        if self.is_playing() { self.pause() } else { self.play() }
    }

    /// Advance the clock, pull decoded frames, and update the texture. Call once per UI frame.
    pub fn tick(&mut self, ctx: &egui::Context) {
        let mut target = match self.clock {
            Some((t0, from)) => from + (t0.elapsed().as_secs_f64() * self.fps) as usize,
            None => self.paused_at,
        };

        let end = self.loop_range.as_ref().map_or(self.frame_count, |r| r.end);
        if self.clock.is_some() && target >= end {
            match self.loop_range.clone() {
                Some(r) if !r.is_empty() => {
                    self.seek(r.start);
                    self.clock = Some((Instant::now(), r.start));
                    target = r.start;
                }
                _ => {
                    self.clock = None;
                    target = end.saturating_sub(1);
                    self.paused_at = target;
                }
            }
        }

        let mut latest = None;
        if let Some(dec) = &mut self.decoder {
            while !dec.finished && dec.last_received.is_none_or(|l| l < target) {
                match dec.rx.try_recv() {
                    Ok((i, rgba)) => {
                        dec.last_received = Some(i);
                        latest = Some((i, rgba));
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => dec.finished = true,
                }
            }
            if dec.finished && self.clock.is_some() && dec.last_received.is_none_or(|l| l < target) {
                // Ran out of video before the clock did.
                self.pause();
            }
        }

        if let Some((i, rgba)) = latest {
            let image = ColorImage::from_rgba_unmultiplied(self.size, &rgba);
            match &mut self.texture {
                Some(tex) => tex.set(image, TextureOptions::LINEAR),
                None => self.texture = Some(ctx.load_texture("player", image, TextureOptions::LINEAR)),
            }
            self.shown = i;
        }

        if self.is_playing() {
            ctx.request_repaint_after(Duration::from_secs_f64(0.5 / self.fps));
        } else if self.decoder.as_ref().is_some_and(|d| !d.finished && d.last_received.is_none_or(|l| l < target)) {
            // Waiting for the first frame after a seek.
            ctx.request_repaint_after(Duration::from_millis(15));
        }
    }
}

impl Decoder {
    fn start(path: &Path, from: usize, fps: f64, size: [usize; 2]) -> Self {
        let (tx, rx) = sync_channel(BUFFERED_FRAMES);
        let path = path.to_path_buf();
        std::thread::spawn(move || {
            let Ok(mut child) = ffmpeg::command("ffmpeg")
                .args(["-v", "error", "-ss", &format!("{:.4}", from as f64 / fps), "-i"])
                .arg(&path)
                .args(["-an", "-vf", &format!("fps={fps},scale={}:{}", size[0], size[1])])
                .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
            else {
                return;
            };
            let mut stdout = child.stdout.take().unwrap();
            let mut frame = from;
            loop {
                let mut buf = vec![0u8; size[0] * size[1] * 4];
                if stdout.read_exact(&mut buf).is_err() {
                    break;
                }
                // Fails once the player drops this decoder (seek / new video).
                if tx.send((frame, buf)).is_err() {
                    break;
                }
                frame += 1;
            }
            let _ = child.kill();
            let _ = child.wait();
        });
        Self { rx, last_received: None, finished: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_for(player: &mut Player, ctx: &egui::Context, cond: impl Fn(&Player) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            player.tick(ctx);
            if cond(player) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    #[test]
    #[ignore = "needs ffmpeg on PATH"]
    fn seeks_plays_and_loops() {
        let input = ffmpeg::make_test_video("player");
        let info = ffmpeg::probe(&input).unwrap();
        let ctx = egui::Context::default();
        let mut player = Player::new(&input, &info, info.fps, 225);

        player.seek(100);
        assert!(wait_for(&mut player, &ctx, |p| p.current_frame() == 100), "seek never showed frame 100");
        assert!(player.texture().is_some());

        player.play();
        assert!(wait_for(&mut player, &ctx, |p| p.current_frame() >= 110), "playback did not advance");

        player.pause();
        player.loop_range = Some(20..30);
        player.play();
        // Should jump into the range, run to its end and wrap back around.
        assert!(wait_for(&mut player, &ctx, |p| p.current_frame() == 29));
        assert!(wait_for(&mut player, &ctx, |p| (20..25).contains(&p.current_frame())));
        assert!(player.is_playing());
    }
}
