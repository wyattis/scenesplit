## 1. Make it usable for other people (do first)
- **Bundle ffmpeg.** Every release download currently needs ffmpeg installed separately. Look for ffmpeg next to the app before PATH, and have CI download a static build into the archives. Watch the GPL point from earlier. *About half a day.*
- **The kittest UI tests we just discussed.** These should go in before the interface grows further. *About a day.*
- **Save the analysis.** Reopening a video re-decodes the whole thing, which takes minutes for long footage. Store the per-frame scores in a cache file keyed by the video's size and modified time so reopening is instant. *Small.*

## 2. Better results with less fiddling
- **Pick the best frame for stills.** Stills currently use the middle frame. Choosing the sharpest frame (least blur) avoids exporting a frame that's mid-fade or blurry. *Small.*
- **Detect fades and dissolves.** Fades to and from black are easy: look for brightness dropping close to zero. Dissolves are harder: look for a slow, steady change over several frames. These are the cuts users currently fix by hand. *About a day for fades, more for dissolves.*
- **Find duplicate scenes.** Montages often repeat the same shot or image. Flag scenes that look alike so you can skip exporting duplicates. *Small to medium.*

## 3. Export options
- **Output choices:** file-name pattern (`{name}-{n}-{time}`), JPG/WebP for stills, a maximum resolution, a quality setting, and GIF/WebM for short clips.
- **Timestamp list for other editors:** export the cut list as CSV/JSON, or as an EDL (edit decision list) to import into Resolve or Premiere and do the cutting there.
- **Run exports in parallel** with a per-file progress view. Exports currently run one at a time.

## 4. Editing speed
- **Select several scenes at once** (Shift/Ctrl-click) and set Video/Still or Export for all of them in one go.
- **Audio in the preview.** It's the most noticeable missing piece of the player, and it's the biggest job on this list: audio output plus keeping it in sync with the video.
- **Make the settings section wrap** at narrow widths (still outstanding from earlier), and add a list of recently opened files.

## 5. "Depending on the content"
Your original question mentioned splitting by what's *in* the clips. A local image model (something like CLIP, running via ONNX) could describe or group each scene's middle frame: "people / outdoors / text slide", or "these 12 scenes are from the same place". That could feed automatic grouping and output subfolders. It's the most distinctive feature here, but also the largest, and adds a sizeable model download.

**What I'd do next:** bundle ffmpeg, save the analysis, and add the kittest tests (about two days in total). Then pick the best frame for stills and detect fades, because those improve the output more than anything else for the effort.