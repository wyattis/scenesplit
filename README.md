# Scene Split

Desktop app (Rust + egui) that splits a video made of multiple clips into separate
clips, exporting static scenes as single PNG images.

Requires `ffmpeg` and `ffprobe` on `PATH`.

```
cargo run --release
cargo test                      # unit tests
cargo test -- --include-ignored # plus an end-to-end test that runs ffmpeg
```

## How it works

1. **Analyze** (`analysis.rs`): ffmpeg decodes the video once at 160x90, at a constant
   frame rate, and the app records the mean RGB difference between consecutive frames.
2. **Detect** (`scenes.rs`): cuts are found from those cached scores, either above a fixed
   threshold or relative to neighbouring frames (adaptive). A scene whose median motion is
   below the still threshold is classified as a still. Cuts can be shifted by a frame offset,
   and frames can be dropped before/after each cut. This step is instant, so the settings
   update live.
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
5. **Export** (`export.rs`): clips via ffmpeg (exact re-encode or fast stream copy),
   stills as PNG of the scene's middle frame.

## Releases

`.github/workflows/release.yml` builds x86_64 binaries for Windows, Linux and macOS.

- Push a tag like `v0.1.0` to build and publish a GitHub Release with the archives attached.
- Or run the workflow manually (Actions → Build → Run workflow) to get the archives as
  run artifacts without creating a release.

Binaries are unsigned: on macOS, right-click → Open the first time (or
`xattr -d com.apple.quarantine scenesplit`); on Windows, SmartScreen → "More info" → "Run anyway".
