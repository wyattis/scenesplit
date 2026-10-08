//! On-disk cache of analysis results, so reopening a video skips the slow decoding pass.
//!
//! Entries live in the user's cache directory, one file per video, keyed by the video's
//! path, size and modification time plus [`analysis::ANALYSIS_VERSION`]. Any change to the
//! file (or to how analysis works) is a cache miss. Unreadable entries are ignored.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};

use crate::analysis::{ANALYSIS_VERSION, Analysis};

const MAGIC: &[u8; 8] = b"SSPLITA2";
/// Oldest entries beyond this many are deleted when saving.
const MAX_ENTRIES: usize = 200;

pub fn default_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("scenesplit").join("analysis"))
}

pub fn load(video: &Path) -> Option<Analysis> {
    load_in(&default_dir()?, video)
}

pub fn save(video: &Path, analysis: &Analysis) -> Result<()> {
    save_in(&default_dir().context("no cache directory on this system")?, video, analysis)
}

pub fn load_in(dir: &Path, video: &Path) -> Option<Analysis> {
    let key = key(video).ok()?;
    let mut bytes = Vec::new();
    fs::File::open(entry_path(dir, &key)).ok()?.read_to_end(&mut bytes).ok()?;
    decode(&bytes, &key)
}

pub fn save_in(dir: &Path, video: &Path, analysis: &Analysis) -> Result<()> {
    let key = key(video)?;
    fs::create_dir_all(dir).with_context(|| format!("could not create {}", dir.display()))?;
    let path = entry_path(dir, &key);
    let tmp = path.with_extension("tmp");
    fs::File::create(&tmp)?.write_all(&encode(analysis, &key))?;
    fs::rename(&tmp, &path)?;
    prune(dir);
    Ok(())
}

/// Identifies the video's current contents and the analysis method.
fn key(video: &Path) -> Result<String> {
    let meta = fs::metadata(video)?;
    let mtime = meta.modified()?.duration_since(UNIX_EPOCH)?.as_nanos();
    let path = fs::canonicalize(video)?;
    Ok(format!("v{ANALYSIS_VERSION}|{}|{}|{mtime}", path.display(), meta.len()))
}

fn entry_path(dir: &Path, key: &str) -> PathBuf {
    // FNV-1a: stable across Rust versions, unlike `DefaultHasher`. Collisions are harmless
    // because the full key is stored in the entry and checked on load.
    let hash = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
    dir.join(format!("{hash:016x}.bin"))
}

/// Layout: magic, key length (u32) + key, fps (f64), frame count n (u64), then per frame:
/// diffs (n - 1 f32), luma, sharpness (n f32 each), colour (n × 3 u8), hash (n u64).
/// Little-endian.
fn encode(a: &Analysis, key: &str) -> Vec<u8> {
    let n = a.luma.len();
    let mut out = Vec::with_capacity(32 + key.len() + n * 27);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&a.fps.to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    for series in [&a.diffs, &a.luma, &a.sharpness] {
        series.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    }
    a.color.iter().for_each(|c| out.extend_from_slice(c));
    a.hash.iter().for_each(|h| out.extend_from_slice(&h.to_le_bytes()));
    out
}

fn decode(bytes: &[u8], expected_key: &str) -> Option<Analysis> {
    let mut rest = bytes.strip_prefix(MAGIC)?;
    let mut take = |len: usize| -> Option<&[u8]> {
        let (head, tail) = rest.split_at_checked(len)?;
        rest = tail;
        Some(head)
    };
    let key_len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if take(key_len)? != expected_key.as_bytes() {
        return None;
    }
    let fps = f64::from_le_bytes(take(8)?.try_into().ok()?);
    let n = usize::try_from(u64::from_le_bytes(take(8)?.try_into().ok()?)).ok()?;
    if n == 0 || !(fps.is_finite() && fps > 0.0) {
        return None;
    }
    let mut floats = |count: usize| -> Option<Vec<f32>> {
        Some(take(count.checked_mul(4)?)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    };
    let diffs = floats(n - 1)?;
    let luma = floats(n)?;
    let sharpness = floats(n)?;
    let color = take(n.checked_mul(3)?)?.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let hash = take(n.checked_mul(8)?)?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    rest.is_empty().then_some(Analysis { diffs, fps, luma, sharpness, hash, color })
}

/// Keep the cache from growing without bound: delete the least recently written entries.
fn prune(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    if files.len() > MAX_ENTRIES {
        files.sort();
        for (_, path) in &files[..files.len() - MAX_ENTRIES] {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(name: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join("scenesplit-test").join("cache").join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let video = dir.join("video.mp4");
        fs::write(&video, b"pretend video").unwrap();
        (dir.join("cache"), video)
    }

    fn sample() -> Analysis {
        let mut a = Analysis::from_diffs(vec![0.5, 80.25, 1.0], 29.97);
        a.luma[2] = 3.5;
        a.sharpness[1] = 120.0;
        a.hash[3] = u64::MAX - 7;
        a.color[0] = [1, 2, 3];
        a
    }

    #[test]
    fn round_trips() {
        let (cache, video) = setup("round_trip");
        assert!(load_in(&cache, &video).is_none());
        save_in(&cache, &video, &sample()).unwrap();
        assert_eq!(load_in(&cache, &video), Some(sample()));
    }

    #[test]
    fn changed_video_is_a_miss() {
        let (cache, video) = setup("changed");
        save_in(&cache, &video, &sample()).unwrap();
        fs::write(&video, b"a different, longer video").unwrap();
        assert!(load_in(&cache, &video).is_none());
    }

    #[test]
    fn corrupt_or_foreign_entries_are_ignored() {
        let (cache, video) = setup("corrupt");
        save_in(&cache, &video, &sample()).unwrap();
        let entry = entry_path(&cache, &key(&video).unwrap());
        let good = fs::read(&entry).unwrap();

        for bad in [&good[..good.len() - 2], &good[..10], b"garbage".as_slice()] {
            fs::write(&entry, bad).unwrap();
            assert!(load_in(&cache, &video).is_none());
        }
        // An entry written for a different key (a hash collision) doesn't match.
        fs::write(&entry, encode(&sample(), "some other video")).unwrap();
        assert!(load_in(&cache, &video).is_none());
    }

    #[test]
    fn prunes_oldest_entries() {
        let (cache, _) = setup("prune");
        fs::create_dir_all(&cache).unwrap();
        for i in 0..MAX_ENTRIES + 5 {
            fs::write(cache.join(format!("{i:04}.bin")), b"x").unwrap();
        }
        prune(&cache);
        assert_eq!(fs::read_dir(&cache).unwrap().count(), MAX_ENTRIES);
    }
}
