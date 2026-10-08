# Third-party software

Release archives of Scene Split include unmodified builds of **FFmpeg** (`ffmpeg` and
`ffprobe`), which Scene Split runs as separate programs.

- FFmpeg is licensed under the GNU General Public License, version 3, in these builds
  (they include GPL components such as x264). The license text is in
  `ffmpeg-GPL-3.0.txt` next to this file.
- Source code: https://ffmpeg.org/download.html (release branch 9.0), and
  https://git.ffmpeg.org/ffmpeg.git
- Builds used:
  - Windows and Linux: https://github.com/BtbN/FFmpeg-Builds (`n9.0-latest`, GPL variant);
    build scripts and the exact versions of bundled libraries are in that repository.
  - macOS (Intel): https://evermeet.cx/ffmpeg/ (9.0.2).

FFmpeg is a trademark of Fabrice Bellard, originator of the FFmpeg project.
