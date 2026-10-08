//! Everything the user changed for one video, saved next to it as `<video>.scenesplit.json`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::scenes::{CutEdits, CutId, Params, SceneKind};
use crate::sections::Sections;

/// The undoable part of the user's work.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Edits {
    pub cuts: CutEdits,
    /// Video/Still choices that override the automatic classification, by scene.
    pub kinds: BTreeMap<CutId, SceneKind>,
    /// Scenes left out of the export.
    pub excluded: BTreeSet<CutId>,
    /// Parts of the video with their own settings.
    pub sections: Sections,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Project {
    pub version: u32,
    pub params: Params,
    pub edits: Edits,
}

impl Default for Project {
    fn default() -> Self {
        Self { version: 1, params: Params::default(), edits: Edits::default() }
    }
}

pub fn sidecar_path(video: &Path) -> PathBuf {
    let mut name = video.file_name().unwrap_or_default().to_os_string();
    name.push(".scenesplit.json");
    video.with_file_name(name)
}

/// Loads the project saved next to `video`, if there is one.
pub fn load(video: &Path) -> Result<Option<Project>> {
    let path = sidecar_path(video);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(serde_json::from_str(&text).with_context(|| format!("could not read {}", path.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("could not open {}", path.display())),
    }
}

/// Saves the project next to `video`. Doesn't create a file for an untouched project.
pub fn save(video: &Path, project: &Project) -> Result<()> {
    let path = sidecar_path(video);
    if *project == Project::default() && !path.exists() {
        return Ok(());
    }
    // Write then rename, so a crash mid-write can't leave a truncated file behind.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(project)?).with_context(|| format!("could not write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("could not write {}", path.display()))?;
    Ok(())
}

/// Undo/redo stack of [`Edits`] snapshots.
#[derive(Default)]
pub struct History {
    undo: Vec<Edits>,
    redo: Vec<Edits>,
    /// Consecutive checkpoints with the same key (e.g. repeated nudges of one cut) are merged
    /// into one undo step.
    last_key: Option<String>,
}

const MAX_UNDO: usize = 200;

impl History {
    /// Record `current` before changing it.
    pub fn checkpoint(&mut self, current: &Edits, coalesce_key: Option<String>) {
        if coalesce_key.is_some() && coalesce_key == self.last_key {
            return;
        }
        self.last_key = coalesce_key;
        self.undo.push(current.clone());
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn undo(&mut self, current: &mut Edits) -> bool {
        let Some(prev) = self.undo.pop() else { return false };
        self.redo.push(std::mem::replace(current, prev));
        self.last_key = None;
        true
    }

    pub fn redo(&mut self, current: &mut Edits) -> bool {
        let Some(next) = self.redo.pop() else { return false };
        self.undo.push(std::mem::replace(current, next));
        self.last_key = None;
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_round_trips_through_json() {
        let mut project = Project::default();
        project.params.cut_offset = -2;
        project.edits.cuts.set_frame(CutId::Detected(10), 12);
        let added = project.edits.cuts.add(40);
        project.edits.kinds.insert(added, SceneKind::Still);
        project.edits.excluded.insert(CutId::Start);

        let json = serde_json::to_string(&project).unwrap();
        assert_eq!(serde_json::from_str::<Project>(&json).unwrap(), project);
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let project: Project = serde_json::from_str(r#"{"params": {"cut_offset": 3}}"#).unwrap();
        assert_eq!(project.params.cut_offset, 3);
        assert_eq!(project.params.cut_threshold, Params::default().cut_threshold);
        assert_eq!(project.edits, Edits::default());
    }

    #[test]
    fn save_skips_untouched_projects_and_load_reads_back() {
        let dir = std::env::temp_dir().join("scenesplit-test").join("project");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let video = dir.join("clip.mp4");

        save(&video, &Project::default()).unwrap();
        assert!(!sidecar_path(&video).exists());
        assert_eq!(load(&video).unwrap(), None);

        let mut project = Project::default();
        project.edits.cuts.add(5);
        save(&video, &project).unwrap();
        assert_eq!(sidecar_path(&video).file_name().unwrap(), "clip.mp4.scenesplit.json");
        assert_eq!(load(&video).unwrap(), Some(project));
    }

    #[test]
    fn undo_redo_and_coalescing() {
        let mut h = History::default();
        let mut e = Edits::default();

        h.checkpoint(&e, None);
        e.cuts.add(5);
        // Three nudges of the same cut become one undo step.
        for f in 6..9 {
            h.checkpoint(&e, Some("nudge m0".into()));
            e.cuts.set_frame(CutId::Manual(0), f);
        }
        assert_eq!(e.cuts.added[&0], 8);

        assert!(h.undo(&mut e));
        assert_eq!(e.cuts.added[&0], 5);
        assert!(h.undo(&mut e));
        assert!(e.cuts.added.is_empty());
        assert!(!h.undo(&mut e));

        assert!(h.redo(&mut e));
        assert!(h.redo(&mut e));
        assert_eq!(e.cuts.added[&0], 8);
        assert!(!h.redo(&mut e));
    }
}
