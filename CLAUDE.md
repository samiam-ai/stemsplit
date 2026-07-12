# StemSplit — Claude Code Handoff

Local Windows desktop app for AI music stem separation, mixing, and vocal analysis.
Single-file Flask app (`app.py`, ~3300 lines) serving an embedded single-page HTML UI.
All state is in-memory (Python dicts). No database.

---

## Architecture

```
stemsplit/
  app.py                  ← Main Flask app (port 5000) — ALL backend + frontend in one file
  acestep_worker.py       ← Separate Flask server (port 5001) — AI remix/generation
  setup.bat               ← First-time pip install script
  setup_acestep.bat       ← Clones ACE-Step-1.5 repo and installs deps via uv
  start.bat               ← Starts StemSplit only (no AI)
  start_all.bat           ← Starts both services in 3 terminal windows:
                             1. "AceStep Worker" — acestep_worker.py on port 5001
                             2. "StemSplit"       — app.py on port 5000
                             3. The start_all.bat launcher window itself
  stop_all.bat            ← Kills all 3 windows by exact title + port kill fallback
  stemsplit_config.json   ← Created at runtime; stores genius_token, hf_token
  uploads/                ← Uploaded audio files (temp, keyed by job UUID)
  outputs/                ← Demucs stem output (outputs/{jid}/*.mp3)
  yt_downloads/           ← YouTube converted files
  projects/               ← Project saves (projects/{jid}/project.json)
```

**Adjacent directory (NOT inside stemsplit/):**
```
../ACE-Step-1.5/          ← Cloned by setup_acestep.bat; used by start_all.bat
```

---

## Running the App

```bash
# StemSplit only (no AI remix):
start.bat

# Full stack with AI remix:
start_all.bat

# Stop everything:
stop_all.bat
```

**After any change to app.py:** stop and restart — Flask runs without auto-reload.

**Test Flask starts cleanly:**
```bash
python -c "from app import app; rs=list(app.url_map.iter_rules()); print(len(rs),'routes'); assert len(rs)==len(set(str(r) for r in rs)),'DUPLICATE ROUTES'"
```
Should print `34 routes`. Duplicate routes are a fatal Flask error (returns HTML 500 on every request).

---

## Python Environment

- Python 3.13 on Windows
- Required packages (installed via setup.bat):
  ```
  flask demucs torchcodec pydub audioop-lts pedalboard noisereduce pyloudnorm
  yt-dlp resemblyzer "audio-separator[cpu]" onnxruntime
  ```
- Optional (improves auto-classify accuracy):
  ```
  lyricsgenius          # Genius lyrics-based singer detection (needs API key)
  openai-whisper        # Word-level timestamps for lyrics alignment
  pyannote.audio        # AI speaker diarization (needs HuggingFace token)
  ```

---

## Service: Main App (port 5000)

**File:** `app.py`
**Start:** `python app.py`
**Single HTML string:** The entire frontend is `HTML = """..."""` inside app.py. The
`/` route returns it. All CSS and JavaScript are embedded in this string.

**JavaScript architecture:**
- Everything is inside one IIFE: `(function() { ... })();`
- Functions defined inside the IIFE are NOT accessible from `onclick=""` HTML attributes
- All button wiring MUST use `addEventListener` from inside the IIFE
- Never use `onclick="functionName()"` for IIFE-scoped functions — they won't be found

**Key global state (Python):**
```python
jobs         = {}   # {jid: {status, message, progress, stems, filename, ...}}
yt_jobs      = {}   # YouTube download jobs
replace_jobs = {}   # AI stem replacement jobs
split_jobs   = {}   # Lead/backing vocal split jobs
auto_classify_jobs = {}  # Karaoke auto-classify jobs
```

**Key constants:**
```python
UPLOAD_DIR   = 'uploads'
OUTPUT_DIR   = 'outputs'
YT_DIR       = 'yt_downloads'
_PROJECTS_DIR = 'projects'
WORKER       = 'http://127.0.0.1:5001'   # AceStep proxy target
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stemsplit_config.json')
```

---

## Service: AceStep Worker (port 5001)

**File:** `acestep_worker.py`
**Start:** `cd ../ACE-Step-1.5 && uv run --with flask python ../stemsplit/acestep_worker.py --acestep-dir .`
(start_all.bat handles this automatically)

**Purpose:** Heavy AI music generation. app.py proxies to it via `w_get()` / `w_post()`.

**Worker routes proxied by app.py:**
| Worker endpoint | app.py proxy route |
|---|---|
| GET  `/health` | GET  `/api/ai/status` |
| POST `/generate` | POST `/api/ai/generate` |
| POST `/cover` | POST `/api/ai/cover` |
| POST `/reference` | POST `/api/ai/reference` |
| POST `/replace` | (via `/api/ai/cover` with stem_job_id) |
| GET  `/job/<jid>` | GET  `/api/ai/job/<jid>` |
| GET  `/download/<jid>` | GET  `/api/ai/download/<jid>` |

**Status:** The UI polls `/api/ai/status`. Worker returns `{ready: bool, loading: bool}`.
First launch downloads ~5GB of models. The AI Remix panel shows "Loading models..." until ready.

---

## All API Routes (34 total)

```
GET  /                              ← Serves the full HTML page
POST /api/separate                  ← Upload file + start Demucs separation
GET  /api/status/<jid>              ← Poll separation progress
GET  /api/dl/<jid>/<stem>           ← Download individual stem
GET  /api/original/<jid>            ← Download original uploaded file
POST /api/mix/<jid>                 ← Export mixed stems (MP3/WAV/FLAC)
POST /api/clean/<jid>               ← Clean & Master a stem
GET  /api/clean_status/<jid>        ← Poll clean job
POST /api/split_vocals/<jid>        ← Split vocals into lead + backing (audio-separator)
GET  /api/vocal_split_status/<sjid> ← Poll vocal split
POST /api/replace_stem/<jid>/<stem> ← AI stem replacement via AceStep
GET  /api/replace_status/<rjid>     ← Poll replacement job
GET  /api/stem_audio/<jid>/<stem>   ← Stream stem audio for karaoke waveform
POST /api/karaoke_export/<jid>      ← Export karaoke regions as labeled audio
POST /api/karaoke_autoclassify/<jid>← Start auto singer detection
GET  /api/karaoke_classify_status/<ajid> ← Poll auto-classify
POST /api/test_lyrics               ← Debug: test Genius lyrics lookup
POST /api/yt/download               ← Start YouTube download (yt-dlp)
GET  /api/yt/status/<jid>           ← Poll YT download
GET  /api/yt/file/<jid>             ← Get downloaded YT file info
POST /api/yt/separate/<yt_jid>      ← Separate a YT-downloaded track
GET  /api/projects                  ← List all saved projects
GET  /api/projects/<pid>/load       ← Load project into mixer
DELETE /api/projects/<pid>          ← Remove project (audio files kept)
POST /api/projects/<pid>/save_state ← Save mixer state to project
GET  /api/config                    ← Get config key presence flags (not values)
POST /api/config                    ← Save API keys to stemsplit_config.json
GET  /api/ai/status                 ← AceStep worker health proxy
POST /api/ai/generate               ← Generate fresh AI music
POST /api/ai/cover                  ← AI style transfer / cover
POST /api/ai/reference              ← AI reference style
GET  /api/ai/job/<jid>              ← Poll AceStep generation job
GET  /api/ai/download/<jid>         ← Download AceStep result
```

---

## Features

### Stem Separation (Demucs)
- **4-stem** (`htdemucs`): vocals, drums, bass, other
- **6-stem** (`htdemucs_6s`): vocals, drums, bass, guitar, piano, other
- **Karaoke** (`htdemucs_ft --two-stems vocals`): vocals + instrumental only
- Output: MP3 files in `outputs/{jid}/`
- Auto-saves a project to `projects/{jid}/project.json` after completion

### Mixer
- Per-stem volume slider, mute, solo
- Presets (Karaoke, Instrumental, Vocals Only, etc.)
- Export as MP3/WAV/FLAC at chosen bitrate
- `state = {}` JS object tracks {vol, muted, soloed} per stem

### Clean & Master
- Pedalboard (EQ, compression, limiting)
- noisereduce (artifact removal)
- pyloudnorm (LUFS normalization)

### AI Remix (AceStep — requires AceStep Worker)
- **Generate Fresh**: text prompt → new music
- **Style Transfer (Cover)**: replace stems with AI-generated version
- **Reference Style**: use another track as style reference
- Three-tier VRAM fallback: full → bf16 → cpu

### Vocal Split (lead vs backing)
- Uses `audio-separator` with `UVR_MDXNET_KARA_2.onnx` model
- Downloads model ~50MB on first use
- Splits the `vocals` stem into `lead_vocals` + `backing_vocals`

### Karaoke Editor
- WaveSurfer.js v6 (CDN) waveform with draggable regions
- Singer labels: Singer A, Singer B, Both
- Draw mode / Seek mode toggle
- Auto-classify singer regions (see below)
- Export labeled audio regions

### Auto-classify Singers (three-tier)
1. **Genius lyrics** (most accurate for known songs):
   - Requires: `pip install lyricsgenius` + Genius API key in settings
   - User types "Artist - Song Name" in the karaoke editor field
   - Fetches attributed lyrics (`[Verse 1: Beyoncé]` format)
   - Distributes track duration proportionally by lyric section
   - Does NOT require whisper — proportional timing only
   - Test with the 🔍 Test button in the karaoke editor

2. **pyannote.audio** (AI diarization):
   - Requires: `pip install pyannote.audio` + HuggingFace token in settings
   - Also requires accepting license at hf.co/pyannote/speaker-diarization-3.1
   - Uses `pyannote/speaker-diarization-3.1` model

3. **Voice fingerprinting** (always available, no setup):
   - Tries resemblyzer → librosa → pydub+numpy FFT (guaranteed fallback)
   - Unsupervised (no reference needed): cosine k-means clustering
   - Supervised (user marks example regions): cosine similarity to references
   - Quality: reasonable for different-gender duets, limited for same-gender

### YouTube Converter
- yt-dlp with browser cookie support (for age-restricted/private videos)
- Downloads to `yt_downloads/{jid}/`
- Can send directly to separation

### Project Panel
- Collapsible panel above the model selector
- Auto-saves every separation to `projects/{jid}/project.json`
- Project JSON stores relative paths from app.py directory
- Load restores stems into mixer with saved volume/mute state
- 💾 Save button saves current mixer state back to project

### Settings Panel (⚙ AI Settings)
- Collapsible panel, stores keys in `stemsplit_config.json`
- `genius_token`: Genius Client Access Token (free at genius.com/api-clients)
- `hf_token`: HuggingFace token (free at hf.co/settings/tokens)
- GET /api/config returns only `{has_genius_token: bool, has_hf_token: bool}` — never key values

---

## Known Issues / Quirks

### Critical: Duplicate Flask routes = silent total failure
If a route is registered twice, Flask throws AssertionError at startup. The app
serves HTML error pages for every request. Always check after edits:
```bash
python -c "from app import app; rs=list(app.url_map.iter_rules()); print(len(rs)); assert len(rs)==len(set(str(r) for r in rs))"
```

### JS function scope
All JS is inside a single IIFE. Functions are NOT global. Buttons must use
`addEventListener` — never `onclick="fnName()"` in HTML attributes.

### JS strings inside Python triple-quoted HTML
Never use `\n` inside a JavaScript string literal within `HTML = """..."""`.
Python interprets `\n` as a real newline → JS syntax error → entire script fails silently.
Use `\` + `n` escaped as `\\n` if you need a newline in a JS string, but preferably
just avoid multi-line JS strings inside Python triple-quotes altogether.

### Python 3.13 + audioop
`audioop` was removed in Python 3.13. The fix is `pip install audioop-lts`.
`setup.bat` handles this.

### AceStep VRAM
Needs ~8GB VRAM for full quality. Falls back to bf16 (4GB) or CPU automatically.
Worker must be running for AI Remix panel — it shows "Loading models..." otherwise.

### Project panel "Loading..." bug (historical)
Was caused by `_makeProjectCard` function being dropped during edits. The
`pjLoad()` JS function sets `pjScroll.innerHTML` to "Loading..." then fetches
`/api/projects`. If anything in `pjRender` throws, the DOM stays on "Loading...".
Current version has all functions inline in the project JS block.

### audio-separator import
Some systems need specific onnxruntime version. Error messages in `split_jobs`
dict include install hints. Model downloads to a cache dir on first use.

---

## File Editing Guidelines

1. **After every change**, run the Flask route check above before testing in browser
2. **Hard refresh** browser after restarting (`Ctrl+Shift+R`) — JS is heavily cached
3. **One insertion point per concern** — don't patch the same area multiple times
4. The HTML string is at the top of app.py (line ~1928 is the `/` route)
5. Python routes are at the bottom of app.py after `if __name__ == '__main__'`... 
   actually they are BEFORE it — `if __name__ == '__main__': app.run(...)` is the last thing

---

## Project JSON Format

```json
{
  "id": "abc123xyz",
  "name": "Song Title",
  "created": "2025-01-01T00:00:00+00:00",
  "modified": "2025-01-01T01:00:00+00:00",
  "source_file": "mysong.mp3",
  "source_path": "uploads/abc123xyz.mp3",
  "duration": 213.4,
  "model": "htdemucs",
  "stems": [
    {"name": "vocals", "path": "outputs/abc123xyz/vocals.mp3"},
    {"name": "drums",  "path": "outputs/abc123xyz/drums.mp3"},
    {"name": "bass",   "path": "outputs/abc123xyz/bass.mp3"},
    {"name": "other",  "path": "outputs/abc123xyz/other.mp3"}
  ],
  "cleaned": false,
  "vocals_split": false,
  "stem_replacements": {},
  "mixer_state": {
    "vocals": {"vol": 85, "muted": false, "soloed": false}
  }
}
```

All paths are relative to the `stemsplit/` directory (`_pj_rel` / `_pj_abs` helpers).

---

## UI Views (controlled by `show(v)` JS function)

| View ID | Shown when |
|---|---|
| `view-upload` | Default / after reset |
| `view-processing` | Demucs running |
| `view-results` | Separation complete, mixer active |
| `view-karaoke` | Karaoke mode result |

The project panel and settings panel are always visible (collapsible, not in `show()`).
The model selector row (`model-row`) is always visible.

---

## What Was Being Worked On

The most recent active issues (in order of priority):

1. **Project panel stuck on "Loading..."** — The collapsible `📁 Recent Projects` panel
   above the model selector fetches `/api/projects` on page load. Was failing because
   `_makeProjectCard` JS function was missing. Should be fixed in current version but
   verify by opening the app — if you see project cards (or "No projects yet"), it works.

2. **Auto-classify quality** — The Genius lyrics approach is the most accurate but
   requires exact or close song title in the "Song title" field of the karaoke editor.
   The 🔍 Test button runs a diagnostic. Audio-only approaches (resemblyzer/pydub)
   have limited accuracy for music vocals due to effects and processing.

3. **API keys flow** — Keys entered in ⚙ AI Settings are saved to `stemsplit_config.json`.
   Verify with: `python -c "import json; print(json.load(open('stemsplit_config.json')))"` 
   (should show genius_token and/or hf_token keys).
