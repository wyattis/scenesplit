//! Sections: frame ranges of the video with their own settings.
//!
//! A section overrides some of the whole-video [`Params`] and inherits the rest. A locked
//! section keeps the cuts and settings it had when it was locked, so later setting changes
//! don't affect it. [`plan`] splits the video into [`Span`]s with the settings that apply.

use std::collections::BTreeMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::scenes::{DetectMode, Params};

macro_rules! overrides {
    ($($field:ident: $ty:ty),* $(,)?) => {
        /// Settings a section overrides. `None` inherits the whole-video setting.
        #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
        #[serde(default)]
        pub struct Overrides {
            $(pub $field: Option<$ty>,)*
        }

        impl Overrides {
            pub fn apply(&self, base: &Params) -> Params {
                // Listing every field makes adding one to `Params` without overriding it a compile error.
                Params { $($field: self.$field.clone().unwrap_or_else(|| base.$field.clone()),)* }
            }

            /// Override every setting that differs between `before` and `after`. Returns the
            /// name of the last one changed.
            pub fn record(&mut self, before: &Params, after: &Params) -> Option<&'static str> {
                let mut changed = None;
                $(if before.$field != after.$field {
                    self.$field = Some(after.$field.clone());
                    changed = Some(stringify!($field));
                })*
                changed
            }

            pub fn is_set(&self, name: &str) -> bool {
                match name {
                    $(stringify!($field) => self.$field.is_some(),)*
                    _ => false,
                }
            }

            pub fn clear(&mut self, name: &str) {
                match name {
                    $(stringify!($field) => self.$field = None,)*
                    _ => {}
                }
            }

            pub fn count(&self) -> usize {
                0 $(+ self.$field.is_some() as usize)*
            }
        }
    };
}

overrides! {
    mode: DetectMode,
    cut_threshold: f32,
    adaptive_ratio: f32,
    adaptive_window: usize,
    adaptive_floor: f32,
    min_scene_secs: f32,
    still_threshold: f32,
    cut_offset: i32,
    drop_before_cut: usize,
    drop_after_cut: usize,
}

/// What a locked section keeps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lock {
    /// The section's settings when it was locked.
    pub params: Params,
    /// Its detected cuts when it was locked: id (see [`crate::scenes::CutId::Detected`]) → frame.
    pub cuts: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub id: u32,
    /// Frames `start..end`. Detected cuts on these frames and scenes starting on them use
    /// this section's settings.
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub overrides: Overrides,
    #[serde(default)]
    pub lock: Option<Lock>,
}

/// Non-overlapping sections, sorted by start.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sections {
    pub list: Vec<Section>,
    pub next_id: u32,
}

impl Sections {
    pub fn get(&self, id: u32) -> Option<&Section> {
        self.list.iter().find(|s| s.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Section> {
        self.list.iter_mut().find(|s| s.id == id)
    }

    /// 1-based position, for display.
    pub fn number(&self, id: u32) -> Option<usize> {
        self.list.iter().position(|s| s.id == id).map(|i| i + 1)
    }

    /// Adds a section over `range`, unless it's empty or overlaps another section.
    pub fn add(&mut self, range: Range<usize>) -> Result<u32, String> {
        if range.is_empty() {
            return Err("A section needs at least one frame.".into());
        }
        if let Some(i) = self.list.iter().position(|s| s.start < range.end && range.start < s.end) {
            return Err(format!("That overlaps section {}. Resize or remove it first.", i + 1));
        }
        let id = self.next_id;
        self.next_id += 1;
        let at = self.list.partition_point(|s| s.start < range.start);
        self.list.insert(at, Section { id, start: range.start, end: range.end, overrides: Overrides::default(), lock: None });
        Ok(id)
    }

    pub fn remove(&mut self, id: u32) {
        self.list.retain(|s| s.id != id);
    }

    /// Frames section `id` can extend over without overlapping its neighbours.
    pub fn room(&self, id: u32, frame_count: usize) -> Range<usize> {
        let Some(i) = self.list.iter().position(|s| s.id == id) else { return 0..frame_count };
        let lo = if i == 0 { 0 } else { self.list[i - 1].end };
        let hi = self.list.get(i + 1).map_or(frame_count, |s| s.start);
        lo..hi.max(lo)
    }

    /// Resize section `id`, clamped so it keeps at least one frame and doesn't overlap.
    pub fn set_range(&mut self, id: u32, range: Range<usize>, frame_count: usize) {
        let room = self.room(id, frame_count);
        let Some(s) = self.get_mut(id) else { return };
        if room.is_empty() {
            return;
        }
        let start = range.start.clamp(room.start, room.end - 1);
        s.start = start;
        s.end = range.end.clamp(start + 1, room.end);
    }
}

/// A stretch of the video and the settings that apply to it.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub range: Range<usize>,
    pub params: Params,
    /// The section this span belongs to, or `None` for the rest of the video.
    pub section: Option<u32>,
    /// Detected cuts frozen by a lock, used instead of running detection.
    pub frozen: Option<BTreeMap<usize, usize>>,
}

/// Contiguous spans covering `0..frame_count` (at least one, even for an empty video).
pub fn plan(global: &Params, sections: &Sections, frame_count: usize) -> Vec<Span> {
    let whole = |range: Range<usize>| Span { range, params: global.clone(), section: None, frozen: None };
    let mut spans = Vec::new();
    let mut at = 0;
    for s in &sections.list {
        // Ignore anything past the end of this video, or out of order.
        let (start, end) = (s.start.max(at), s.end.min(frame_count));
        if start >= end {
            continue;
        }
        if start > at {
            spans.push(whole(at..start));
        }
        let (params, frozen) = match &s.lock {
            Some(lock) => (lock.params.clone(), Some(lock.cuts.clone())),
            None => (s.overrides.apply(global), None),
        };
        spans.push(Span { range: start..end, params, section: Some(s.id), frozen });
        at = end;
    }
    if at < frame_count || spans.is_empty() {
        spans.push(whole(at..frame_count.max(at)));
    }
    spans
}

/// The span containing `frame` (the last span for frames past the end).
pub fn span_at(spans: &[Span], frame: usize) -> &Span {
    let i = spans.partition_point(|s| s.range.start <= frame).saturating_sub(1);
    &spans[i.min(spans.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_inherit_unset_fields_and_record_changes() {
        let global = Params::default();
        let mut o = Overrides::default();
        assert_eq!(o.apply(&global), global);

        let mut after = global.clone();
        after.cut_offset = 4;
        assert_eq!(o.record(&global, &after), Some("cut_offset"));
        assert!(o.is_set("cut_offset") && !o.is_set("cut_threshold"));
        assert_eq!(o.count(), 1);

        // Later whole-video changes still reach the fields that aren't overridden.
        let global2 = Params { cut_offset: -1, drop_after_cut: 3, ..global.clone() };
        let p = o.apply(&global2);
        assert_eq!((p.cut_offset, p.drop_after_cut), (4, 3));

        o.clear("cut_offset");
        assert_eq!(o, Overrides::default());
    }

    #[test]
    fn sections_refuse_overlaps_and_stay_sorted() {
        let mut s = Sections::default();
        let b = s.add(50..80).unwrap();
        let a = s.add(10..20).unwrap();
        assert!(s.add(70..90).is_err());
        assert!(s.add(15..16).is_err());
        assert!(s.add(30..30).is_err());
        assert_eq!(s.list.iter().map(|x| x.id).collect::<Vec<_>>(), [a, b]);
        assert_eq!(s.number(b), Some(2));
    }

    #[test]
    fn resizing_stops_at_neighbours() {
        let mut s = Sections::default();
        let a = s.add(10..20).unwrap();
        let b = s.add(50..80).unwrap();
        s.set_range(b, 5..500, 100);
        assert_eq!((s.get(b).unwrap().start, s.get(b).unwrap().end), (20, 100));
        s.set_range(a, 15..3, 100);
        assert_eq!((s.get(a).unwrap().start, s.get(a).unwrap().end), (15, 16), "keeps one frame");
    }

    #[test]
    fn plan_covers_the_video_with_section_settings() {
        let global = Params::default();
        let mut s = Sections::default();
        let a = s.add(10..20).unwrap();
        s.get_mut(a).unwrap().overrides.cut_offset = Some(2);
        let b = s.add(20..40).unwrap();
        s.get_mut(b).unwrap().lock = Some(Lock { params: Params { drop_after_cut: 9, ..global.clone() }, cuts: [(25, 26)].into() });
        s.add(90..200).unwrap(); // runs past the end

        let spans = plan(&global, &s, 100);
        let ranges: Vec<_> = spans.iter().map(|x| (x.range.clone(), x.section)).collect();
        assert_eq!(ranges, [(0..10, None), (10..20, Some(a)), (20..40, Some(b)), (40..90, None), (90..100, Some(2))]);
        assert_eq!(spans[1].params.cut_offset, 2);
        assert_eq!(spans[2].params.drop_after_cut, 9);
        assert_eq!(spans[2].frozen, Some([(25, 26)].into()));
        assert_eq!(span_at(&spans, 39).section, Some(b));
        assert_eq!(span_at(&spans, 40).section, None);
        assert_eq!(span_at(&spans, 1000).section, Some(2));

        assert_eq!(plan(&global, &s, 0).len(), 1, "an empty video still has a span");
    }
}
