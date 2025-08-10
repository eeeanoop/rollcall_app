
import sys
import os
import time
import json
import math
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

from PySide6.QtCore import (Qt, QThread, Signal, Slot, QSize, QTimer)
from PySide6.QtGui import (QImage, QPixmap, QAction, QIcon)
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QFileDialog, QTabWidget, QGridLayout,
    QGroupBox, QSplitter, QMessageBox, QStyle, QFrame, QScrollArea
)

# Offline TTS
try:
    import pyttsx3
except Exception as e:
    pyttsx3 = None

# InsightFace (detector + embeddings with ONNX Runtime backend)
try:
    from insightface.app import FaceAnalysis
except Exception as e:
    FaceAnalysis = None


APP_TITLE = "Automated Roll Call"
BASE_DIR = Path(__file__).resolve().parent
ROSTER_DIR = BASE_DIR / "roster"
THUMBS_DIR = ROSTER_DIR / "thumbs"
STUDENT_IMAGES_DIR = BASE_DIR / "student_images"
RESULTS_DIR = BASE_DIR / "results"
UNAUTH_DIR = RESULTS_DIR / "unauthorized"

# =========================
# Tuning parameters
# =========================
SIM_THRESHOLD = 0.35   # cosine similarity threshold for a positive match (higher = stricter)
DETECT_SIZE = (640, 640)
FRAME_FPS = 18

# Make "Unauthorized" much stricter to cut down noise:
UNKNOWN_STRICT_SIM_MAX = 0.20     # only call unauthorized if best similarity <= this (very low confidence it's anyone in roster)
DET_CONF_MIN = 0.60               # require strong face detection confidence
MIN_FACE_AREA = 8000              # pixels; ignore tiny far-away faces
UNAUTH_CONFIRM_FRAMES = 6         # require at least N consecutive frames before calling unauthorized
UNAUTH_MATCH_DIST = 64.0          # px; how close centers must be to keep the same unknown track
UNAUTH_TRACK_MAX_GAP_SEC = 1.0    # seconds; drop stale unknown tracks

# Legacy debounce is now secondary (track confirmation is primary)
UNAUTH_DEBOUNCE_SEC = 2.0


def ensure_dirs():
    for p in [ROSTER_DIR, THUMBS_DIR, RESULTS_DIR, UNAUTH_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def speak_async(text: str):
    """Non-blocking TTS call using pyttsx3 in a short-lived thread."""
    if pyttsx3 is None:
        return

    def _run():
        try:
            engine = pyttsx3.init()  # use platform default (SAPI5/NSSpeech/eSpeak)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    th = threading.Thread(target=_run, daemon=True)
    th.start()


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert BGR (OpenCV) frame to QImage for display."""
    h, w, ch = frame_bgr.shape
    bytes_per_line = ch * w
    # Convert BGR to RGB for display
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)


def crop_with_margin(img, bbox, margin=0.15):
    """Crop bbox with margin; bbox is [x1,y1,x2,y2]."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return img[y1:y2, x1:x2]


def resize_square(img, size=128):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def name_from_filename(p: Path) -> str:
    name = p.stem
    name = name.replace("_", " ").strip()
    return name


class RosterManager:
    """
    Builds/loads the roster: names, averaged embeddings per student, and thumbnails.
    Uses InsightFace FaceAnalysis (buffalo_l) to detect + embed.
    """
    def __init__(self, app: 'Recognizer'):
        self.recognizer = app
        self.names = []               # list[str] aligned with embeddings
        self.embeddings = None        # np.ndarray (N, 512)
        self.name_to_thumb = {}       # name -> path to thumb image

    def build_or_load(self):
        ensure_dirs()
        emb_path = ROSTER_DIR / "embeddings.npy"
        names_path = ROSTER_DIR / "names.json"

        # If cache exists AND student_images hasn't changed, load cache
        if emb_path.exists() and names_path.exists():
            try:
                self.embeddings = np.load(str(emb_path))
                self.names = json.loads(names_path.read_text())
                # thumbs exist per name (best effort)
                for name in self.names:
                    t = THUMBS_DIR / f"{name}.jpg"
                    if t.exists():
                        self.name_to_thumb[name] = str(t)
                if len(self.names) == self.embeddings.shape[0] and self.embeddings.shape[1] == 512:
                    print("[Roster] Loaded cached roster.")
                    return
            except Exception as e:
                print("[Roster] Failed to load cache, rebuilding...", e)

        print("[Roster] Building roster from student_images...")
        # Map: name -> list of embeddings; also store a representative thumb
        buckets = {}
        rep_thumbs = {}

        # Iterate all images
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_files.extend(STUDENT_IMAGES_DIR.glob(ext))

        if not image_files:
            raise RuntimeError(f"No student images found in: {STUDENT_IMAGES_DIR}")

        for img_path in image_files:
            name = name_from_filename(img_path)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[Roster] Skipping unreadable image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.recognizer.analyzer.get(img_rgb)

            if not faces:
                print(f"[Roster] No face found in: {img_path}")
                continue

            # Choose largest face by area
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = best.normed_embedding  # already L2-normalized (512,)
            if emb is None or emb.shape[0] != 512:
                print(f"[Roster] No embedding for: {img_path}")
                continue

            if name not in buckets:
                buckets[name] = []
            buckets[name].append(emb)

            # Create a thumbnail once per name (first image)
            if name not in rep_thumbs:
                crop = crop_with_margin(img, best.bbox, margin=0.2)
                thumb = resize_square(crop, 128)
                thumb_path = THUMBS_DIR / f"{name}.jpg"
                cv2.imwrite(str(thumb_path), thumb)
                rep_thumbs[name] = str(thumb_path)

        # Average embeddings per name
        names = []
        embs = []
        for n, vecs in buckets.items():
            arr = np.vstack(vecs)  # (k,512)
            mean = arr.mean(axis=0)
            # Renormalize
            mean = mean / (np.linalg.norm(mean) + 1e-10)
            names.append(n)
            embs.append(mean)

        if not names:
            raise RuntimeError("No valid faces/embeddings produced from student_images.")

        # Save
        self.names = names
        self.embeddings = np.vstack(embs).astype("float32")
        self.name_to_thumb = {n: rep_thumbs.get(n, "") for n in self.names}

        np.save(str(emb_path), self.embeddings)
        names_path.write_text(json.dumps(self.names, ensure_ascii=False, indent=2))
        print(f"[Roster] Built {len(self.names)} students.")

    def get_absent(self, present_set):
        return [n for n in self.names if n not in present_set]


class Recognizer:
    """
    Wraps InsightFace FaceAnalysis (detector + embeddings).
    """
    def __init__(self, providers=None, det_size=DETECT_SIZE):
        if FaceAnalysis is None:
            raise RuntimeError("InsightFace is not installed. Please `pip install insightface onnxruntime`")
        # name='buffalo_l' includes a RetinaFace-like detector + ArcFace-like recognition model
        # Use CPU by default
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.analyzer = FaceAnalysis(name='buffalo_l', providers=providers)
        # ctx_id is unused by onnxruntime backend; keep 0
        self.analyzer.prepare(ctx_id=0, det_size=det_size)


class UnknownTracker:
    """
    Very lightweight tracker for unknown faces.
    Tracks by face center proximity and counts consecutive frames.
    """
    def __init__(self):
        self._next_id = 1
        self.tracks = {}  # id -> dict(cx, cy, count, last_ts)

    def _match(self, cx, cy):
        best_id, best_dist = None, 1e9
        for tid, t in self.tracks.items():
            dx = t["cx"] - cx
            dy = t["cy"] - cy
            d = math.hypot(dx, dy)
            if d < best_dist:
                best_dist, best_id = d, tid
        if best_dist <= UNAUTH_MATCH_DIST:
            return best_id
        return None

    def update(self, detections):
        """
        detections: list of dict(cx, cy, ts)
        Returns: list of track dicts that hit the confirmation threshold this update.
        """
        now = time.time()

        # Decay old tracks
        stale = [tid for tid, t in self.tracks.items() if (now - t["last_ts"]) > UNAUTH_TRACK_MAX_GAP_SEC]
        for tid in stale:
            self.tracks.pop(tid, None)

        confirmed = []

        for d in detections:
            cx, cy, ts = d["cx"], d["cy"], d["ts"]
            tid = self._match(cx, cy)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"cx": cx, "cy": cy, "count": 0, "last_ts": ts}

            t = self.tracks[tid]
            # update track
            t["cx"] = 0.6 * t["cx"] + 0.4 * cx
            t["cy"] = 0.6 * t["cy"] + 0.4 * cy
            t["last_ts"] = ts
            t["count"] += 1

            if t["count"] == UNAUTH_CONFIRM_FRAMES:
                confirmed.append({"id": tid, "cx": t["cx"], "cy": t["cy"]})

        # Remove confirmed tracks so we don't double-fire too quickly
        for c in confirmed:
            self.tracks.pop(c["id"], None)

        return confirmed


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)                          # BGR frame for display
    recognized_name = Signal(str, str, float)                 # name, thumb_path, score
    unauthorized_seen = Signal(str, np.ndarray)               # timestamp, face_bgr
    status_text = Signal(str)

    def __init__(self, recognizer: Recognizer, roster: RosterManager, camera_index=0, threshold=SIM_THRESHOLD, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.roster = roster
        self.camera_index = camera_index
        self.threshold = float(threshold)
        self._running = False
        self._last_unauth_ts = 0.0

        # Simple debounce for recognized names to avoid repeated announcements
        self._seen_names_recent = {}  # name -> last time announced

        # Unknown tracking
        self.unk_tracker = UnknownTracker()

    def stop(self):
        self._running = False

    def best_match(self, emb: np.ndarray):
        # cosine similarity since roster embeddings are normalized
        sims = self.roster.embeddings @ emb  # (N,)
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def _should_flag_unknown(self, face, frame_shape, best_sim):
        """
        Apply strict gating to decide if a face is a strong 'unknown' candidate.
        This does NOT emit yet; it only says whether to accumulate toward confirmation.
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = face.bbox
        area = max(0, int((x2 - x1) * (y2 - y1)))
        det_conf = getattr(face, "det_score", 1.0)

        if area < MIN_FACE_AREA:
            return False
        if det_conf < DET_CONF_MIN:
            return False
        if best_sim > UNKNOWN_STRICT_SIM_MAX:  # too similar to someone; not confident enough to call unauthorized
            return False
        return True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
        # Try a modest resolution for CPU
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        if not cap.isOpened():
            self.status_text.emit("Camera could not be opened.")
            return

        self._running = True
        frame_interval = 1.0 / FRAME_FPS
        while self._running:
            start_t = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status_text.emit("Failed to read from camera.")
                time.sleep(0.1)
                continue

            # Display early (responsive UI)
            self.frame_ready.emit(frame)

            # Process
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.recognizer.analyzer.get(rgb)

            unknown_candidates = []  # for tracker

            if faces:
                for f in faces:
                    emb = f.normed_embedding
                    if emb is None or emb.shape[0] != 512:
                        continue

                    idx, score = self.best_match(emb)
                    if score >= self.threshold:
                        name = self.roster.names[idx]
                        tlast = self._seen_names_recent.get(name, 0.0)
                        now = time.time()
                        if now - tlast > 2.0:  # announce at most once every 2 sec per name
                            self._seen_names_recent[name] = now
                            # Prepare a small face thumbnail for UI (derived from current frame)
                            crop = crop_with_margin(frame, f.bbox, margin=0.2)
                            face_thumb = resize_square(crop, 128)
                            # Use roster thumb for uniformity in the Present list
                            self.recognized_name.emit(name, self.roster.name_to_thumb.get(name, ""), score)
                    else:
                        # Strong unknown gating; only accumulate if we are VERY sure it's not someone in the roster
                        if self._should_flag_unknown(f, frame.shape, score):
                            x1, y1, x2, y2 = f.bbox
                            cx = 0.5 * (x1 + x2)
                            cy = 0.5 * (y1 + y2)
                            unknown_candidates.append({"cx": cx, "cy": cy, "ts": time.time(), "bbox": f.bbox})

                # Update unknown tracker and emit only on confirmation
                if unknown_candidates:
                    confirmed = self.unk_tracker.update(unknown_candidates)
                    for _c in confirmed:
                        # Debounce final emission so we don't spam TTS if multiple confirmations overlap
                        now = time.time()
                        if now - self._last_unauth_ts > UNAUTH_DEBOUNCE_SEC:
                            self._last_unauth_ts = now
                            # Use the first candidate bbox for a snapshot (approximate)
                            f0 = unknown_candidates[0]
                            crop = crop_with_margin(frame, f0["bbox"], margin=0.2)
                            face_thumb = resize_square(crop, 128)
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self.unauthorized_seen.emit(ts, face_thumb)

            # pacing
            elapsed = time.time() - start_t
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        cap.release()


class ImageList(QListWidget):
    """Simple list widget that shows thumbnails + names."""
    def __init__(self, icon_size=QSize(96, 96), parent=None):
        super().__init__(parent)
        self.setIconSize(icon_size)
        self.setResizeMode(QListWidget.Adjust)
        self.setViewMode(QListWidget.IconMode)
        self.setMovement(QListWidget.Static)
        self.setSpacing(8)
        self.setUniformItemSizes(False)
        self.setWrapping(True)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 720)

        ensure_dirs()

        # Recognizer + Roster
        try:
            self.recognizer = Recognizer(providers=["CPUExecutionProvider"])
        except Exception as e:
            QMessageBox.critical(self, "InsightFace Error",
                                 f"Failed to initialize InsightFace. Install deps and try again.\n\n{e}")
            raise

        self.roster = RosterManager(self.recognizer)
        try:
            self.roster.build_or_load()
        except Exception as e:
            QMessageBox.critical(self, "Roster Error", f"Failed to build roster:\n{e}")
            raise

        # State
        self.present_set = set()      # names seen this session
        self.present_first_seen = {}  # name -> timestamp str
        self.session_started = False
        self.worker = None
        self.sim_threshold = SIM_THRESHOLD

        # UI elements
        self.live_label = QLabel("Camera feed")
        self.live_label.setMinimumSize(640, 360)
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setFrameShape(QFrame.StyledPanel)

        self.btn_start = QPushButton("Start roll call")
        self.btn_end = QPushButton("End roll call")
        self.btn_end.setEnabled(False)

        self.status = QLabel("Ready.")
        self.status.setStyleSheet("color: #555; padding: 4px;")

        # Right panel tabs
        self.tabs = QTabWidget()
        self.tab_present = ImageList()
        self.tab_absent = ImageList()
        self.tab_unauth = ImageList()

        self.tabs.addTab(self.tab_present, "Present")
        self.tabs.addTab(self.tab_absent, "Absent")
        self.tabs.addTab(self.tab_unauth, "Unauthorized")

        # Layout
        left = QVBoxLayout()
        left.addWidget(self.live_label, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_end)
        left.addLayout(buttons)
        left.addWidget(self.status)

        right = QVBoxLayout()
        right.addWidget(self.tabs)

        main = QHBoxLayout(self)
        splitter = QSplitter()
        left_widget = QWidget(); left_widget.setLayout(left)
        right_widget = QWidget(); right_widget.setLayout(right)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter)

        # Signals
        self.btn_start.clicked.connect(self.start_session)
        self.btn_end.clicked.connect(self.end_session)

        # Populate initial Absent tab (everyone absent before start)
        self.refresh_absent_list()

    def refresh_absent_list(self):
        self.tab_absent.clear()
        for name in self.roster.names:
            if name in self.present_set:
                continue
            item = QListWidgetItem(name)
            thumb_path = self.roster.name_to_thumb.get(name, "")
            if thumb_path and os.path.exists(thumb_path):
                item.setIcon(QIcon(thumb_path))
            self.tab_absent.addItem(item)

    @Slot()
    def start_session(self):
        if self.session_started:
            return
        self.session_started = True
        self.present_set.clear()
        self.present_first_seen.clear()
        self.tab_present.clear()
        self.tab_unauth.clear()
        self.refresh_absent_list()

        self.btn_start.setEnabled(False)
        self.btn_end.setEnabled(True)
        self.status.setText("Roll call in progress...")

        self.worker = CameraWorker(
            recognizer=self.recognizer,
            roster=self.roster,
            camera_index=0,
            threshold=self.sim_threshold
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.recognized_name.connect(self.on_recognized)
        self.worker.unauthorized_seen.connect(self.on_unauthorized)
        self.worker.status_text.connect(self.on_status)
        self.worker.start()
        speak_async("Starting roll call.")

    @Slot()
    def end_session(self):
        if not self.session_started:
            return
        self.session_started = False
        self.btn_start.setEnabled(True)
        self.btn_end.setEnabled(False)
        self.status.setText("Ending session...")

        # Stop worker thread
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        # Finalize & show absent/present
        self.refresh_absent_list()
        self.status.setText("Session ended.")

        # Save results
        self.save_session_results()
        speak_async("Roll call ended.")

    @Slot(np.ndarray)
    def on_frame(self, frame_bgr: np.ndarray):
        qimg = bgr_to_qimage(frame_bgr)
        self.live_label.setPixmap(QPixmap.fromImage(qimg))

    @Slot(str, str, float)
    def on_recognized(self, name: str, thumb_path: str, score: float):
        if name not in self.present_set:
            self.present_set.add(name)
            self.present_first_seen[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            item = QListWidgetItem(f"{name}  ({score:.2f})")
            if thumb_path and os.path.exists(thumb_path):
                item.setIcon(QIcon(thumb_path))
            self.tab_present.addItem(item)

            # Remove from absent list
            self.refresh_absent_list()

            self.status.setText(f"Recognized: {name} (score {score:.2f})")
            speak_async(f"Welcome, {name}")

    @Slot(str, np.ndarray)
    def on_unauthorized(self, ts: str, face_bgr: np.ndarray):
        # Save the unauthorized face with timestamp
        UNAUTH_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"unauthorized_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        save_path = UNAUTH_DIR / fname
        cv2.imwrite(str(save_path), face_bgr)

        item = QListWidgetItem(ts)
        # Write temp thumb file for icon
        tmp_thumb = str(save_path)
        item.setIcon(QIcon(tmp_thumb))
        self.tab_unauth.addItem(item)

        self.status.setText("Unauthorized person confirmed.")
        speak_async("Unauthorized")

    @Slot(str)
    def on_status(self, text: str):
        self.status.setText(text)

    def save_session_results(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "started_at": None,  # can add if you want to track start time
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "present": [
                {"name": n, "first_seen": self.present_first_seen.get(n, "")}
                for n in sorted(self.present_set)
            ],
            "absent": [n for n in self.roster.get_absent(self.present_set)],
            "unauthorized_snaps": sorted([p for p in os.listdir(UNAUTH_DIR) if p.lower().endswith(".jpg")])
        }
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = RESULTS_DIR / f"session_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.status.setText(f"Saved results: {out_path.name}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
    

if __name__ == "__main__":
    main()
