
#  AI SMART SURVEILLANCE SYSTEM
#  Models: yolo11s.pt (person) | yolo11n-pose.pt (pose) | weapon.pt (gun & knife)
#  Stack : Streamlit + OpenCV + Ultralytics + pygame

import streamlit as st
import cv2
import numpy as np
import time
import os
import pygame
from collections import defaultdict, deque
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Surveillance System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family:'Rajdhani',sans-serif; background:#0a0c10; color:#c8d6e5; }
  #MainMenu, footer, header { visibility:hidden; }
  h1 { font-family:'Share Tech Mono',monospace !important; color:#00e5ff !important;
       letter-spacing:3px; font-size:1.6rem !important; margin-bottom:0 !important; }
  h2, h3 { font-family:'Share Tech Mono',monospace !important; color:#00e5ff !important;
            letter-spacing:2px; font-size:1rem !important; }
  [data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1e3a4a; }
  [data-testid="stSidebar"] * { color:#8ab4c8 !important; }
  [data-testid="stMetric"] { background:#0d1117; border:1px solid #1e3a4a;
    border-radius:6px; padding:10px 14px; margin-bottom:8px; }
  [data-testid="stMetricLabel"] { color:#5a8fa8 !important; font-size:0.72rem !important; letter-spacing:1px; }
  [data-testid="stMetricValue"] { font-family:'Share Tech Mono',monospace !important;
                                   color:#00e5ff !important; font-size:1.5rem !important; }
  .stButton > button { background:transparent; border:1px solid #00e5ff; color:#00e5ff;
    font-family:'Share Tech Mono',monospace; letter-spacing:2px; border-radius:4px;
    transition:all 0.2s; width:100%; }
  .stButton > button:hover { background:#00e5ff22; }
  .stop-btn > button { border-color:#ff4b4b !important; color:#ff4b4b !important; }
  .stop-btn > button:hover { background:#ff4b4b22 !important; }
  .lock-btn > button { border-color:#00ff99 !important; color:#00ff99 !important; }
  .lock-btn > button:hover { background:#00ff9922 !important; }
  [data-testid="stSlider"] > div > div { accent-color:#00e5ff; }
  .alert-log { background:#0d1117; border:1px solid #1e3a4a; border-radius:6px;
    padding:10px; font-family:'Share Tech Mono',monospace; font-size:0.72rem;
    color:#5a8fa8; max-height:220px; overflow-y:auto; }
  .alert-log .alert-line { color:#ff4b4b; }
  .alert-log .info-line  { color:#5a8fa8; }
  .status-badge { display:inline-block; padding:2px 10px; border-radius:20px;
    font-family:'Share Tech Mono',monospace; font-size:0.7rem; letter-spacing:2px; }
  .status-live { background:#00e5ff22; border:1px solid #00e5ff; color:#00e5ff; }
  .status-idle { background:#ffffff11; border:1px solid #445566; color:#445566; }
  .zone-locked { background:#00ff9911; border:1px solid #00ff99; border-radius:6px;
    padding:8px 12px; font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#00ff99; margin-top:8px; }
  .zone-unlocked { background:#ffaa0011; border:1px solid #ffaa00; border-radius:6px;
    padding:8px 12px; font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#ffaa00; margin-top:8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_PATH    = os.path.dirname(os.path.abspath(__file__))
ALERT_FOLDER = os.path.join(BASE_PATH, "alerts")
LOG_FILE     = os.path.join(BASE_PATH, "event_log.txt")
os.makedirs(ALERT_FOLDER, exist_ok=True)

# Weapon class IDs from the trained model (0 = gun, 1 = knife)
ACTIVE_WEAPON_IDS = {0, 1}
WEAPON_NAMES      = {0: "GUN", 1: "KNIFE"}
WEAPON_COLORS     = {0: (0, 0, 255), 1: (0, 255, 140)}

# COCO keypoint indices used for zone intrusion checks.
# We skip eyes/ears (1-4) as they are too noisy.
# 0=nose  5=L-shoulder  6=R-shoulder  7=L-elbow   8=R-elbow
# 9=L-wrist  10=R-wrist  11=L-hip  12=R-hip
# 13=L-knee  14=R-knee  15=L-ankle  16=R-ankle
BODY_KP_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
KP_CONF_THRESH  = 0.30   # minimum keypoint confidence to trust

# Detection thresholds
CONF_PERSON     = 0.50   # ignore person detections below 50 % confidence
CONF_WEAPON     = 0.40   # ignore weapon detections below 40 % confidence
RUN_THRESH_NORM = 0.18   # normalised speed threshold for running detection
LOITER_SECS     = 8      # seconds on-screen before flagging as loitering

# Alert cooldowns (seconds between repeat alerts of the same type)
CD_ZONE_BODY = 8.0
CD_WEAPON    = 10.0
CD_RUN       = 6.0

# Frames a person must be inside the zone before an alert fires (debounce)
ZONE_CONFIRM_FRAMES = 3

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────
#  ZONE HELPERS
# ─────────────────────────────────────────────────────────────
def keypoints_in_zone(zone_arr, kps):
    """Return (bool, list-of-hit-points) — True if any body keypoint is inside the zone."""
    if zone_arr is None or kps is None:
        return False, []
    hit_pts = []
    for idx in BODY_KP_INDICES:
        if kps[idx, 2] >= KP_CONF_THRESH:
            pt = (float(kps[idx, 0]), float(kps[idx, 1]))
            if cv2.pointPolygonTest(zone_arr, pt, False) >= 0:
                hit_pts.append((int(pt[0]), int(pt[1])))
    return len(hit_pts) > 0, hit_pts


def bbox_in_zone_fallback(zone_arr, x1, y1, x2, y2):
    """Fallback when pose data is unavailable — test 9 grid points of the bounding box."""
    if zone_arr is None:
        return False
    for x in [x1, (x1 + x2) // 2, x2]:
        for y in [y1, (y1 + y2) // 2, y2]:
            if cv2.pointPolygonTest(zone_arr, (float(x), float(y)), False) >= 0:
                return True
    return False


def point_in_zone(zone_arr, pt):
    if zone_arr is None or pt is None:
        return False
    return cv2.pointPolygonTest(zone_arr, (float(pt[0]), float(pt[1])), False) >= 0


# ─────────────────────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────
_defaults = {
    "run":           False,
    "zone_locked":   False,
    "zone_points":   [(100, 100), (520, 100), (520, 400), (100, 400)],
    "total_alerts":  0,
    "alert_log":     [],
    "use_droidcam":  False,
    "droidcam_ip":   "192.168.x.x",
    "droidcam_port": "4747",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────
#  LOGGING & ALERTS
# ─────────────────────────────────────────────────────────────
def log_event(msg: str):
    """Append a timestamped line to event_log.txt."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} - {msg}\n")
    except Exception:
        pass


def add_alert(msg: str, level: str = "alert"):
    """Add an alert to the on-screen log and persist it to event_log.txt."""
    ts = time.strftime("%H:%M:%S")
    st.session_state.alert_log.insert(0, (ts, msg, level))
    st.session_state.alert_log = st.session_state.alert_log[:80]
    if level == "alert":
        st.session_state.total_alerts += 1
    log_event(msg)


def save_snapshot(frame: np.ndarray, tag: str):
    """Save a JPEG snapshot — rate-limited to once every 5 seconds per tag."""
    key = f"snap_{tag}"
    now = time.time()
    if now - st.session_state.get(key, 0) < 5:
        return
    st.session_state[key] = now
    fname = os.path.join(ALERT_FOLDER, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(fname, frame)


# ─────────────────────────────────────────────────────────────
#  DETECTION UTILITIES
# ─────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter)


def nms_boxes(boxes_confs, iou_thresh=0.45):
    """Non-maximum suppression — removes duplicate overlapping detections."""
    if not boxes_confs:
        return []
    sorted_b = sorted(boxes_confs, key=lambda b: b[4], reverse=True)
    keep = []
    while sorted_b:
        best = sorted_b.pop(0)
        keep.append(best)
        sorted_b = [b for b in sorted_b if iou(best[:4], b[:4]) < iou_thresh]
    return keep


# ─────────────────────────────────────────────────────────────
#  TEMPORAL SMOOTHER
# ─────────────────────────────────────────────────────────────
class TemporalSmoother:
    """
    Reduces flickering false positives by only confirming a detection
    that appeared in at least `min_hits` of the last `window_size` frames.
    """
    def __init__(self, window_size=3, min_hits=2, iou_match=0.40):
        self.window    = deque(maxlen=window_size)
        self.min_hits  = min_hits
        self.iou_match = iou_match

    def update(self, new_detections):
        self.window.append(new_detections)
        if len(self.window) < self.min_hits:
            return new_detections
        confirmed = []
        for det in self.window[-1]:
            hits = 1
            for past in list(self.window)[:-1]:
                if any(iou(det[:4], p[:4]) >= self.iou_match for p in past):
                    hits += 1
            if hits >= self.min_hits:
                confirmed.append(det)
        return confirmed


# ─────────────────────────────────────────────────────────────
#  CENTROID TRACKER  (with re-ID across short occlusions)
# ─────────────────────────────────────────────────────────────
class CentroidTracker:
    """
    Assigns persistent IDs to people across frames using centroid distance
    matching with velocity prediction. When someone disappears briefly, the
    tracker tries to re-assign their old ID when they reappear (re-ID).
    """
    def __init__(self, max_disappeared=45, reid_ttl=20.0,
                 reid_dist=200.0, match_dist=150.0, iou_assist=0.30):
        self.next_id         = 0
        self.objects         = {}           # oid → centroid
        self.boxes           = {}           # oid → bounding box
        self.velocity        = {}           # oid → (vx, vy)
        self.disappeared     = defaultdict(int)
        self.history         = {}           # oid → deque of last 10 positions
        self.entry_time      = {}           # oid → first-seen timestamp
        self.speed_ema       = {}           # oid → smoothed speed (EMA)
        self.lost_objects    = {}           # recently lost IDs for re-ID
        self.max_disappeared = max_disappeared
        self.reid_ttl        = reid_ttl
        self.reid_dist       = reid_dist
        self.match_dist      = match_dist
        self.iou_assist      = iou_assist

    def update(self, centroids, boxes=None):
        boxes = boxes or [None] * len(centroids)
        self._cleanup_lost()
        if not centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return {}
        if not self.objects:
            for c, b in zip(centroids, boxes):
                self._register(c, b)
        else:
            ids       = list(self.objects.keys())
            predicted = []
            for oid in ids:
                cx, cy = self.objects[oid]
                vx, vy = self.velocity.get(oid, (0, 0))
                predicted.append((cx + vx, cy + vy))
            D = np.linalg.norm(
                np.array(predicted)[:, None] - np.array(centroids)[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                oid = ids[r]
                if D[r, c] <= self.match_dist:
                    self._update_object(oid, centroids[c], boxes[c])
                    used_rows.add(r); used_cols.add(c)
                elif boxes[c] is not None and self.boxes.get(oid) is not None:
                    if iou(self.boxes[oid], boxes[c]) >= self.iou_assist:
                        self._update_object(oid, centroids[c], boxes[c])
                        used_rows.add(r); used_cols.add(c)
            for r in set(range(len(ids))) - used_rows:
                self.disappeared[ids[r]] += 1
                if self.disappeared[ids[r]] > self.max_disappeared:
                    self._deregister(ids[r])
            for c in set(range(len(centroids))) - used_cols:
                self._register(centroids[c], boxes[c])
        return dict(self.objects)

    def _update_object(self, oid, centroid, box):
        prev_cx, prev_cy = self.objects[oid]
        cx, cy = centroid
        vx_p, vy_p = self.velocity.get(oid, (0, 0))
        self.velocity[oid] = (0.4 * (cx - prev_cx) + 0.6 * vx_p,
                              0.4 * (cy - prev_cy) + 0.6 * vy_p)
        self.objects[oid] = centroid
        if box is not None:
            self.boxes[oid] = box
        self.history[oid].append(centroid)
        self.disappeared[oid] = 0

    def _register(self, c, box=None):
        oid = self._recover_id(c)
        self.objects[oid]     = c
        self.boxes[oid]       = box
        self.velocity[oid]    = (0, 0)
        self.disappeared[oid] = 0
        if oid not in self.history:
            self.history[oid] = deque(maxlen=10)
        self.history[oid].append(c)
        if oid not in self.entry_time:
            self.entry_time[oid] = time.time()

    def _deregister(self, oid):
        if oid in self.objects:
            self.lost_objects[oid] = {
                "centroid":   self.objects[oid],
                "last_seen":  time.time(),
                "history":    list(self.history.get(oid, [])),
                "entry_time": self.entry_time.get(oid, time.time()),
                "box":        self.boxes.get(oid),
            }
            del self.objects[oid]
        for store in [self.disappeared, self.history, self.entry_time,
                      self.velocity, self.boxes, self.speed_ema]:
            if oid in store:
                del store[oid]

    def _recover_id(self, c):
        """Try to match a new detection to a recently lost person."""
        best_id, best_d = None, float("inf")
        for oid, data in self.lost_objects.items():
            d = float(np.linalg.norm(np.array(data["centroid"]) - np.array(c)))
            if d < best_d:
                best_d = d; best_id = oid
        if best_id is not None and best_d <= self.reid_dist:
            state = self.lost_objects.pop(best_id)
            restored = deque(state.get("history", []), maxlen=10)
            if not restored:
                restored.append(c)
            self.history[best_id]    = restored
            self.entry_time[best_id] = state.get("entry_time", time.time())
            self.boxes[best_id]      = state.get("box")
            self.velocity[best_id]   = (0, 0)
            return best_id
        oid = self.next_id
        self.next_id += 1
        return oid

    def _cleanup_lost(self):
        now   = time.time()
        stale = [oid for oid, d in self.lost_objects.items()
                 if now - d.get("last_seen", now) > self.reid_ttl]
        for oid in stale:
            del self.lost_objects[oid]

    def speed(self, oid, person_height=1.0):
        """Return normalised, EMA-smoothed movement speed for this person."""
        h = list(self.history.get(oid, []))
        if len(h) < 2:
            return 0.0
        look_back = min(4, len(h) - 1)
        raw  = float(np.linalg.norm(np.array(h[-1]) - np.array(h[-(look_back + 1)]))) / max(look_back, 1)
        norm = raw / max(person_height, 1.0)
        prev = self.speed_ema.get(oid, norm)
        s    = 0.35 * norm + 0.65 * prev
        self.speed_ema[oid] = s
        return s

    def dwell(self, oid):
        """Return how long (seconds) this person has been on screen."""
        return time.time() - self.entry_time.get(oid, time.time())


# ─────────────────────────────────────────────────────────────
#  POSE UTILITIES
# ─────────────────────────────────────────────────────────────
def extract_keypoints(pose_result, scale_x=1.0, scale_y=1.0):
    """Extract and scale keypoint arrays from a YOLO pose result."""
    people_kps = []
    if pose_result.keypoints is None:
        return people_kps
    for person_kps in pose_result.keypoints.data:
        kps = person_kps.cpu().numpy().copy()
        kps[:, 0] *= scale_x
        kps[:, 1] *= scale_y
        people_kps.append(kps)
    return people_kps


def match_pose_to_person(kps, person_box):
    """Return True if this skeleton's hip midpoint (or nose fallback) is inside person_box."""
    x1, y1, x2, y2 = person_box
    lh = kps[11]; rh = kps[12]
    if lh[2] < 0.2 and rh[2] < 0.2:
        nose = kps[0]
        if nose[2] < 0.2:
            return False
        px, py = int(nose[0]), int(nose[1])
    else:
        px = int((lh[0] + rh[0]) / 2)
        py = int((lh[1] + rh[1]) / 2)
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_pose_overlay(frame, kps, hit_pts):
    """Draw keypoint circles — red inside zone, cyan outside."""
    hit_set = set(map(tuple, hit_pts))
    for idx in BODY_KP_INDICES:
        if kps[idx, 2] >= KP_CONF_THRESH:
            pt    = (int(kps[idx, 0]), int(kps[idx, 1]))
            color = (0, 0, 255) if pt in hit_set else (0, 220, 220)
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 7, (255, 255, 255), 1)


# ─────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────
def draw_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y + 2), (10, 12, 16), -1)
    cv2.putText(frame, text, (x + 3, y - 2), FONT, 0.55, color, 1, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, color, label=""):
    """Draw corner-bracket bounding box (cleaner look than a full rectangle)."""
    L, t = 18, 2
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * L, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * L), color, t)
    if label:
        draw_label(frame, label, x1, y1, color)


def draw_zone(frame, zone_pts, color=(0, 80, 220)):
    """Draw a semi-transparent filled polygon for the restricted zone."""
    if len(zone_pts) >= 3:
        arr     = np.array(zone_pts, np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [arr], color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [arr], True, color, 1, cv2.LINE_AA)


def draw_hud(frame, counts):
    """Render recording indicator and stats bar onto the frame."""
    h  = frame.shape[0]
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"● REC  {ts}", (8, 20), FONT, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    info = (f"PEOPLE:{counts['people']}  GUN:{counts['gun']}  "
            f"KNIFE:{counts['knife']}  RUN:{counts['running']}  LOIT:{counts['loiter']}")
    cv2.putText(frame, info, (8, h - 10), FONT, 0.40, (80, 120, 160), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  ALARM
# ─────────────────────────────────────────────────────────────
_alarm_ready = False

def _init_alarm():
    global _alarm_ready
    if not _alarm_ready:
        try:
            pygame.mixer.init()
            _alarm_ready = True
        except Exception:
            pass

def trigger_alarm(force_restart=False):
    _init_alarm()
    alarm_path = os.path.join(BASE_PATH, "alarm.mp3")
    if not (_alarm_ready and os.path.exists(alarm_path)):
        return
    if force_restart and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    if force_restart or not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(alarm_path)
        pygame.mixer.music.play()

def stop_alarm():
    if _alarm_ready and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    person_m = YOLO(os.path.join(BASE_PATH, "yolo11s.pt"))
    pose_m   = YOLO(os.path.join(BASE_PATH, "yolo11n-pose.pt"))
    weapon_m = YOLO(os.path.join(BASE_PATH, "weapon.pt"))
    # Warm-up pass so the first live frame isn't slow
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    person_m(dummy, imgsz=480, verbose=False)
    pose_m(dummy,   imgsz=480, verbose=False)
    weapon_m(dummy, imgsz=480, verbose=False)
    return person_m, pose_m, weapon_m


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AI SMART SURVEILLANCE SYSTEM")
    st.markdown("---")

    if st.session_state.run:
        st.markdown('<span class="status-badge status-live">● LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">◌ IDLE</span>', unsafe_allow_html=True)
    st.markdown("")

    # ── Camera source ──────────────────────────────────────
    st.markdown("### CAMERA")
    use_dc = st.toggle(
        "Use Phone Camera (DroidCam)",
        value=st.session_state.use_droidcam,
        disabled=st.session_state.run,
    )
    if not st.session_state.run:
        st.session_state.use_droidcam = use_dc

    if use_dc:
        dc_ip = st.text_input(
            "Phone IP Address",
            value=st.session_state.droidcam_ip,
            placeholder="e.g. 192.168.0.192",
            disabled=st.session_state.run,
        )
        dc_port = st.text_input(
            "Port",
            value=st.session_state.droidcam_port,
            placeholder="e.g. 4747",
            disabled=st.session_state.run,
        )
        if not st.session_state.run:
            st.session_state.droidcam_ip   = dc_ip
            st.session_state.droidcam_port = dc_port

        droidcam_url = (
            f"http://{st.session_state.droidcam_ip}"
            f":{st.session_state.droidcam_port}/video"
        )
        st.caption(f"📡 Stream URL: {droidcam_url}")

        if not st.session_state.run:
            if st.button("🔗 Test Connection"):
                test_cap = cv2.VideoCapture(droidcam_url)
                ret, _   = test_cap.read()
                test_cap.release()
                if ret:
                    st.success("✅ DroidCam connected!")
                else:
                    st.error("❌ Could not connect. Check IP/port and ensure DroidCam is open on your phone.")

    st.markdown("---")

    # ── Start / Stop ───────────────────────────────────────
    if not st.session_state.run:
        if st.button("▶ START"):
            if st.session_state.zone_locked:
                st.session_state.run = True
                st.rerun()
            else:
                st.warning("⚠ Please lock your zone first in the Zone Setup tab!")

    if st.session_state.run:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("■ STOP"):
            st.session_state.run         = False
            st.session_state.zone_locked = False
            stop_alarm()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### LIVE METRICS")
    m_people  = st.empty()
    m_gun     = st.empty()
    m_knife   = st.empty()
    m_running = st.empty()
    m_loiter  = st.empty()
    m_alerts  = st.empty()
    st.markdown("---")
    st.markdown("### ALERT LOG")
    log_box = st.empty()


# ─────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────
st.markdown("# AI SMART SURVEILLANCE SYSTEM")
tab_live, tab_zone = st.tabs(["📺  Live Feed", "🗺  Zone Setup"])

with tab_zone:
    st.markdown("### Define Restricted Zone")
    if st.session_state.run:
        zp = st.session_state.zone_points
        st.markdown(
            f'<div class="zone-locked">🔒 ZONE LOCKED — '
            f'({zp[0][0]},{zp[0][1]}) → ({zp[2][0]},{zp[2][1]})<br>'
            f'Stop camera to change zone.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        zx1 = c1.slider("Zone X1", 0, 640, st.session_state.zone_points[0][0], key="zx1")
        zy1 = c1.slider("Zone Y1", 0, 480, st.session_state.zone_points[0][1], key="zy1")
        zx2 = c2.slider("Zone X2", 0, 640, st.session_state.zone_points[2][0], key="zx2")
        zy2 = c2.slider("Zone Y2", 0, 480, st.session_state.zone_points[2][1], key="zy2")

        col_prev, col_lock = st.columns([2, 1])
        with col_prev:
            if st.button("📷 Preview Zone"):
                tcap = cv2.VideoCapture(0)
                ret, frm = tcap.read()
                tcap.release()
                if ret:
                    frm = cv2.resize(frm, (640, 480))
                    draw_zone(frm, [(zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)])
                    st.image(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB),
                             caption="Zone Preview", use_container_width=True)
                else:
                    st.warning("Camera not accessible.")

        with col_lock:
            st.markdown(""); st.markdown("")
            st.markdown('<div class="lock-btn">', unsafe_allow_html=True)
            if st.button("🔒 Lock Zone"):
                st.session_state.zone_points = [
                    (zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)
                ]
                st.session_state.zone_locked = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.zone_locked:
            zp = st.session_state.zone_points
            st.markdown(
                f'<div class="zone-locked">🔒 ZONE LOCKED — '
                f'({zp[0][0]},{zp[0][1]}) → ({zp[2][0]},{zp[2][1]})<br>'
                f'Ready to start camera.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="zone-unlocked">⚠ ZONE NOT LOCKED — '
                'Adjust sliders, then click Lock Zone before starting.</div>',
                unsafe_allow_html=True)

with tab_live:
    feed_placeholder = st.empty()
    if not st.session_state.run:
        feed_placeholder.markdown(
            """<div style="text-align:center;padding:80px;color:#1e3a4a;
                font-family:'Share Tech Mono',monospace;font-size:0.9rem;letter-spacing:2px;">
               ◌ SYSTEM IDLE<br><br>
               Step 1 — Define zone in Zone Setup tab<br>
               Step 2 — Click 🔒 Lock Zone<br>
               Step 3 — Press ▶ START in sidebar
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MAIN CAMERA LOOP
# ─────────────────────────────────────────────────────────────
if st.session_state.run:

    person_model, pose_model, weapon_model = load_models()

    _cam_src = (
        f"http://{st.session_state.droidcam_ip}:{st.session_state.droidcam_port}/video"
        if st.session_state.use_droidcam else 0
    )

    cap = cv2.VideoCapture(_cam_src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # always grab the latest frame

    tracker         = CentroidTracker(max_disappeared=90, reid_ttl=60.0,
                                      reid_dist=300.0, match_dist=150.0, iou_assist=0.30)
    person_smoother = TemporalSmoother(window_size=3, min_hits=2, iou_match=0.40)
    weapon_smoother = TemporalSmoother(window_size=2, min_hits=2, iou_match=0.35)

    frame_count      = 0
    last_person_data = []
    last_weapon_data = []
    last_pose_kps    = []
    fps_t            = time.time()
    fps_v            = 0.0
    last_alert_time  = defaultdict(float)

    zone_frames  = defaultdict(lambda: {"body": 0})
    zone_alerted = defaultdict(lambda: {"body": 0.0})

    # Detection runs on a 480×360 sub-frame for speed; scale results back to 640×480
    SX = 640 / 480
    SY = 480 / 360

    LOCKED_ZONE = list(st.session_state.zone_points)
    zone_arr    = np.array(LOCKED_ZONE, np.int32) if len(LOCKED_ZONE) >= 3 else None

    def should_alert(key, cooldown=CD_ZONE_BODY):
        """Return True and update timer only if the cooldown has elapsed."""
        now = time.time()
        if now - last_alert_time[key] > cooldown:
            last_alert_time[key] = now
            return True
        return False

    try:
        while st.session_state.run:

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1
            small = cv2.resize(frame, (480, 360))

            # ── Person detection (every 2 frames) ─────────
            if frame_count % 2 == 0:
                res = person_model(small, imgsz=480, verbose=False)[0]
                raw_persons = []
                for box in res.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) >= CONF_PERSON:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        raw_persons.append((
                            int(bx1 * SX), int(by1 * SY),
                            int(bx2 * SX), int(by2 * SY),
                            float(box.conf[0])
                        ))
                raw_persons      = nms_boxes(raw_persons, iou_thresh=0.45)
                confirmed        = person_smoother.update(raw_persons)
                last_person_data = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in confirmed]

            # ── Pose estimation (every 2 frames) ──────────
            if frame_count % 2 == 0:
                pose_res      = pose_model(small, imgsz=480, verbose=False)[0]
                last_pose_kps = extract_keypoints(pose_res, scale_x=SX, scale_y=SY)

            # ── Weapon detection (every 3 frames) ─────────
            if frame_count % 3 == 0:
                res = weapon_model(small, imgsz=480, verbose=False)[0]
                raw_weapons = []
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in ACTIVE_WEAPON_IDS:
                        continue
                    if float(box.conf[0]) >= CONF_WEAPON:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        raw_weapons.append((
                            int(bx1 * SX), int(by1 * SY),
                            int(bx2 * SX), int(by2 * SY),
                            float(box.conf[0]), cls_id
                        ))
                raw_weapons      = nms_boxes(raw_weapons, iou_thresh=0.40)
                confirmed_w      = weapon_smoother.update(raw_weapons)
                last_weapon_data = [(x1, y1, x2, y2, cid)
                                    for x1, y1, x2, y2, _, cid in confirmed_w]

                for _, _, _, _, cls_id in last_weapon_data:
                    wname = WEAPON_NAMES.get(cls_id, "THREAT")
                    if should_alert(f"weapon_{cls_id}", cooldown=CD_WEAPON):
                        add_alert(f"⚠ {wname} DETECTED!", "alert")
                        trigger_alarm()
                        save_snapshot(frame, wname.lower())

            # ── Tracker update ─────────────────────────────
            centroids = [((x1 + x2) // 2, (y1 + y2) // 2)
                         for x1, y1, x2, y2 in last_person_data]
            tracked   = tracker.update(centroids, boxes=list(last_person_data))

            draw_zone(frame, LOCKED_ZONE)

            running_count = 0
            loiter_count  = 0
            active_ids    = set()

            for oid, (cx, cy) in tracked.items():
                active_ids.add(oid)
                if not last_person_data:
                    continue

                # Find the closest bounding box to this tracked centroid
                dists = [(abs(cx - (x1 + x2) // 2) + abs(cy - (y1 + y2) // 2), (x1, y1, x2, y2))
                         for x1, y1, x2, y2 in last_person_data]
                dists.sort()
                x1, y1, x2, y2 = dists[0][1]
                person_height   = max(y2 - y1, 1)

                spd   = tracker.speed(oid, person_height=person_height)
                dwell = tracker.dwell(oid)
                tags  = [f"#{oid}"]
                color = (60, 220, 80)  # green by default

                if spd > RUN_THRESH_NORM:
                    running_count += 1
                    tags.append("RUN")
                    color = (0, 100, 255)
                    if should_alert(f"run_{oid}", cooldown=CD_RUN):
                        add_alert(f"Person #{oid} RUNNING", "alert")
                        save_snapshot(frame, f"running_{oid}")

                if dwell > LOITER_SECS:
                    loiter_count += 1
                    tags.append(f"LOITER {dwell:.0f}s")
                    if color == (60, 220, 80):
                        color = (0, 200, 255)

                # ── Zone intrusion check ───────────────────
                if zone_arr is not None:
                    matched_kps  = None
                    for kps in last_pose_kps:
                        if match_pose_to_person(kps, (x1, y1, x2, y2)):
                            matched_kps = kps
                            break

                    body_in_zone = False
                    hit_pts      = []

                    if matched_kps is not None:
                        body_in_zone, hit_pts = keypoints_in_zone(zone_arr, matched_kps)
                        draw_pose_overlay(frame, matched_kps, hit_pts)
                    else:
                        body_in_zone = bbox_in_zone_fallback(zone_arr, x1, y1, x2, y2)

                    # Debounce: require ZONE_CONFIRM_FRAMES consecutive inside-zone frames
                    if body_in_zone:
                        zone_frames[oid]["body"] = min(
                            zone_frames[oid]["body"] + 1, ZONE_CONFIRM_FRAMES + 5)
                    else:
                        zone_frames[oid]["body"] = max(zone_frames[oid]["body"] - 1, 0)

                    body_confirmed = zone_frames[oid]["body"] >= ZONE_CONFIRM_FRAMES

                    if body_confirmed:
                        tags.append("RESTRICTED")
                        color = (0, 0, 255)
                        now   = time.time()
                        if now - zone_alerted[oid]["body"] > CD_ZONE_BODY:
                            zone_alerted[oid]["body"] = now
                            add_alert(f"Person #{oid} entered RESTRICTED ZONE!", "alert")
                            trigger_alarm(force_restart=True)
                            save_snapshot(frame, f"zone_{oid}")

                draw_box(frame, x1, y1, x2, y2, color, " | ".join(tags))

            # Clean up state for people who left the frame
            for oid in list(zone_frames.keys()):
                if oid not in active_ids:
                    del zone_frames[oid]
            for oid in list(zone_alerted.keys()):
                if oid not in active_ids:
                    del zone_alerted[oid]

            # ── Draw weapon boxes ──────────────────────────
            gun_cnt = knife_cnt = 0
            for x1, y1, x2, y2, cls_id in last_weapon_data:
                wname  = WEAPON_NAMES.get(cls_id, "THREAT")
                wcolor = WEAPON_COLORS.get(cls_id, (0, 0, 255))
                draw_box(frame, x1, y1, x2, y2, wcolor, f"! {wname}")
                if wname == "GUN":
                    gun_cnt += 1
                elif wname == "KNIFE":
                    knife_cnt += 1

            # ── FPS overlay ────────────────────────────────
            now   = time.time()
            fps_v = 0.9 * fps_v + 0.1 * (1.0 / max(now - fps_t, 1e-6))
            fps_t = now
            cv2.putText(frame, f"FPS {fps_v:.1f}", (frame.shape[1] - 80, 18),
                        FONT, 0.45, (60, 140, 200), 1, cv2.LINE_AA)

            draw_hud(frame, {
                "people":  len(last_person_data),
                "gun":     gun_cnt,
                "knife":   knife_cnt,
                "running": running_count,
                "loiter":  loiter_count,
            })

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with tab_live:
                feed_placeholder.image(frame_rgb, use_container_width=True)

            # ── Sidebar metrics ────────────────────────────
            m_people.metric("👤 People",    len(last_person_data))
            m_gun.metric("🔫 Guns",         gun_cnt)
            m_knife.metric("🔪 Knives",     knife_cnt)
            m_running.metric("🏃 Running",  running_count)
            m_loiter.metric("🕐 Loitering", loiter_count)
            m_alerts.metric("🚨 Alerts",    st.session_state.total_alerts)

            lines_html = "".join(
                f'<div class="{"alert-line" if lvl == "alert" else "info-line"}">'
                f'[{ts}] {msg}</div>'
                for ts, msg, lvl in st.session_state.alert_log[:15]
            )
            log_box.markdown(
                f'<div class="alert-log">{lines_html}</div>',
                unsafe_allow_html=True)

    finally:
        cap.release()
        stop_alarm()

else:
    # Show zeros when system is idle
    m_people.metric("👤 People",    0)
    m_gun.metric("🔫 Guns",         0)
    m_knife.metric("🔪 Knives",     0)
    m_running.metric("🏃 Running",  0)
    m_loiter.metric("🕐 Loitering", 0)
    m_alerts.metric("🚨 Alerts",    st.session_state.total_alerts)
