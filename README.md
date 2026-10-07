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
   below the still threshold is classified as a still. This step is instant, so the sliders
   update live.
3. **Review** (`app.rs`): thumbnails, difference graph, per-scene Video/Still override,
   include/exclude, merge with next.
4. **Export** (`export.rs`): clips via ffmpeg (exact re-encode or fast stream copy),
   stills as PNG of the scene's middle frame.
