# StemSplit - AI Audio Stem Separator

Separate any song into individual stems (vocals, drums, bass, instruments)
using Meta's open-source Demucs AI. Runs 100% locally — no subscriptions,
no uploads, no limits.

---

## Requirements

- Windows 10 or 11
- Python 3.8 or later — download free at https://www.python.org/downloads/
  - During install, check "Add Python to PATH"

---

## First-Time Setup (do this once)

1. Double-click **setup.bat**
2. Wait for it to finish installing packages
3. You're ready to go

---

## Running the App

1. Double-click **start.bat**
2. Your browser will open automatically to http://localhost:5000
3. Upload a track, choose 4 or 6 stems, and wait for processing
4. Download your stems when complete

---

## Notes

- **First run** downloads the Demucs AI model (~80MB) — requires internet
- **Subsequent runs** are fully offline
- Processing takes 2-5 minutes depending on track length and your CPU
- Stems are saved to the `outputs/` folder in this directory
- Supported formats: MP3, WAV, FLAC, M4A, OGG, AIFF

## Stem Options

| Mode    | Stems                                          |
|---------|------------------------------------------------|
| 4 Stems | Vocals, Drums, Bass, Other (everything else)   |
| 6 Stems | Vocals, Drums, Bass, Guitar, Piano, Other      |

---

Powered by Meta's Demucs: https://github.com/facebookresearch/demucs
