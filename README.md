# Scene Split

Desktop app (Rust + egui) that splits a video made of multiple clips into separate
clips, exporting static scenes as single PNG images.

Uses `ffmpeg` and `ffprobe`: the `with-ffmpeg` release archives include them next to the
executable, which the app checks first; otherwise they need to be on `PATH`. The app shows a warning at
startup if it can't find them.

```
cargo run --release
cargo test                      # unit + UI tests
cargo test -- --include-ignored # plus end-to-end tests that run ffmpeg
```

UI tests (`src/ui_tests.rs`) use `egui_kittest` to drive the widgets with real mouse and
keyboard input: dragging cut markers, snapping, the right-click menu, zooming, the
overview and sections bars, the filmstrip, shortcuts, and a whole-app run that clicks real
buttons.

## How it works

1. **Analyze** (`analysis.rs`): ffmpeg decodes the video once at 160x90, at a constant
   frame rate. For each frame the app records the mean RGB difference from the previous
   frame, brightness, sharpness (variance of the Laplacian), mean colour and a 64-bit
   difference hash.
   Results are cached (`cache.rs`) in the user's cache directory, keyed by the video's
   path, size and modification time, so reopening a video is instant.
2. **Detect** (`scenes.rs`, `transitions.rs`): hard cuts are spikes in the difference,
   either above a fixed threshold or relative to neighbouring frames (adaptive). Gradual
   transitions don't spike, so they're found separately:
   - *Fades through black*: frames darker than the black level, plus the darkening and
     brightening ramps around them. The next scene starts where the picture comes back.
   - *Dissolves*: a blend of two shots has less detail than either, following
     `(1-t)²·a + t²·b`. Stretches whose sharpness fits that dip between two
     different-looking frames are dissolves; the cut goes in the middle.

   By default transition frames are left out of the scenes on both sides. A scene whose
   median motion is below the still threshold is classified as a still; its image is the
   sharpest kept frame (or one you pick with "Use playhead frame"). Scenes whose still frame
   matches a frame of an earlier scene (hash and colour) are marked ≈ so repeats can be
   skipped. Cuts can be shifted by a frame offset, and frames can be dropped before/after
   each cut. This step is instant, so the settings update live.
3. **Review** (`app.rs`, `player.rs`): thumbnails, difference graph, per-scene Video/Still
   override, include/exclude, merge with next, and a video-only preview player whose
   "loop selected scene" plays exactly the frames that will be exported.
   Shortcuts: Space play/pause, ←/→ step one frame.
4. **Edit cuts** (`editor.rs`, `project.rs`): drag cut markers on the difference graph
   (snaps to nearby spikes; hold Alt to disable), double-click to add a cut, right-click
   for more. Ctrl+scroll zooms the graph, scroll pans. Selecting a cut shows a filmstrip of
   the frames around it; click the frame that should start the new scene. Hand-placed cuts
   ignore the cut offset. Edits are undoable and auto-saved next to the video as
   `<video>.scenesplit.json`, along with the detection settings.

   | Key | Action |
   |---|---|
   | Space | play / pause |
   | ← → | step one frame (Shift: 10) |
   | S | split at playhead |
   | Delete | delete selected cut |
   | , . | nudge selected cut (Shift: 10) |
   | [ ] | previous / next cut |
   | Esc | deselect cut |
   | Ctrl+Z, Ctrl+Shift+Z / Ctrl+Y | undo, redo |
   | Shift+click a scene | select a range of scenes |
5. **Sections** (`sections.rs`): give part of the video its own settings. Select scenes
   (Shift+click for a range) and press "New section"; the settings panel then edits that
   section, and the settings it changes are highlighted (↺ goes back to the whole-video
   value). Other settings still follow the whole video. Sections show as a bar above the
   scene overview: click one to edit it, drag an edge to resize (snaps to cuts). **Lock**
   freezes a section's settings and detected cuts so later changes elsewhere can't affect
   it. Detected cuts and scenes use the settings of the section they start in.
6. **Export** (`export.rs`): clips via ffmpeg (exact re-encode or fast stream copy),
   stills as PNG of the scene's middle frame.

## Releases

`.github/workflows/release.yml` builds x86_64 binaries for Windows, Linux and macOS.

- Each platform gets two archives. `…-with-ffmpeg` includes static GPL builds of
  `ffmpeg`/`ffprobe` 9.0 (BtbN builds for Windows/Linux, evermeet.cx for macOS) plus
  `THIRD-PARTY-NOTICES.md` and the GPL text. `…-no-ffmpeg` is the app alone, for people
  who already have ffmpeg on `PATH`. CI runs the full test suite, including the ffmpeg
  end-to-end tests, against the bundled builds.
- Push a tag like `v0.1.0` to build and publish a GitHub Release with the archives attached.
- Or run the workflow manually (Actions → Build → Run workflow) to get the archives as
  run artifacts without creating a release.

Binaries are unsigned: on macOS, right-click → Open the first time (or
`xattr -d com.apple.quarantine scenesplit`); on Windows, SmartScreen → "More info" → "Run anyway".
