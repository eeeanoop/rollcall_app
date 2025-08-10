# Automated Roll Call (Face Recognition)

Cross‑platform Python app (macOS + Windows) that performs automated roll call from a webcam using modern face detection + embeddings (InsightFace), with a simple PySide6 UI.

## Features
- Start/End roll call buttons.
- Recognizes students by comparing to embeddings built from images in `student_images/`.
- Speaks each recognized student's name (`pyttsx3`) and says **"Unauthorized"** for unknown faces.
- End-of-session view: Present, Absent, and Unauthorized lists with thumbnails.
- Stores results in `results/session_YYYY-mm-dd_HHMMSS.json` and unauthorized snapshots in `results/unauthorized/`.

## Setup

1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# macOS
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

2) Put one or more photos per student in `student_images/`.
   - Filenames should be the name to announce, e.g. `Ada Lovelace.jpg`, `Ada_Lovelace_2.png`.
   - Multiple images for the same student are supported; embeddings are averaged.

3) Run the app:

```bash
python app.py
```

> On first run, the app will build `roster/embeddings.npy`, `roster/names.json` and thumbnails in `roster/thumbs/`.
> Subsequent runs will load these caches and start faster.

### macOS camera permissions
If the camera doesn't show or you never get a camera permission prompt, run from Terminal first. For distribution as a standalone app, sign/notarize the bundle and include `NSCameraUsageDescription` in the app's Info.plist so macOS prompts for access.

### Tuning
- Similarity threshold default: **0.35**. If you see false matches, raise it (e.g. 0.40). If it misses matches, lower it (e.g. 0.30).
- Lighting at the doorway matters a lot. Try to keep the camera at head height with even lighting.

### Packaging (optional)
- Windows: PyInstaller (one-folder).
- macOS: `pyside6-deploy` or PyInstaller, with codesigning/notarization for camera permissions.

## Folder Structure
```
rollcall_app/
  app.py
  README.md
  requirements.txt
  student_images/
  roster/
    embeddings.npy
    names.json
    thumbs/
  results/
    session_*.json
    unauthorized/
  assets/
```
