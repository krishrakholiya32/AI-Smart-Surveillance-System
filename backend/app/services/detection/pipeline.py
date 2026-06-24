"""
Full per-frame detection pipeline.
Handles: person, pose, weapon, zone intrusion, running, loitering, fall, crowd.
"""
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.detection.tracker import CentroidTracker, nms_boxes, iou
from app.services.detection.smoother import TemporalSmoother

# COCO keypoints: 0=nose 5=L-shldr 6=R-shldr 7=L-elbow 8=R-elbow
# 9=L-wrist 10=R-wrist 11=L-hip 12=R-hip 13=L-knee 14=R-knee 15=L-ankle 16=R-ankle
BODY_KP_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
KP_CONF_THRESH  = 0.30
FONT            = cv2.FONT_HERSHEY_SIMPLEX

ACTIVE_WEAPON_IDS = {0, 1}
WEAPON_NAMES      = {0: "GUN", 1: "KNIFE"}
WEAPON_COLORS     = {0: (0, 0, 255), 1: (0, 255, 140)}


# ─── Zone helpers ───────────────────────────────────────────────────────────

def _zone_arr(points) -> Optional[np.ndarray]:
    if points and len(points) >= 3:
        return np.array(points, dtype=np.int32)
    return None


def keypoints_in_zone(zone, kps) -> Tuple[bool, List]:
    if zone is None or kps is None:
        return False, []
    hit_pts = []
    for idx in BODY_KP_INDICES:
        if kps[idx, 2] >= KP_CONF_THRESH:
            pt = (float(kps[idx, 0]), float(kps[idx, 1]))
            if cv2.pointPolygonTest(zone, pt, False) >= 0:
                hit_pts.append((int(pt[0]), int(pt[1])))
    return len(hit_pts) > 0, hit_pts


def bbox_in_zone_fallback(zone, x1, y1, x2, y2) -> bool:
    if zone is None:
        return False
    for x in [x1, (x1 + x2) // 2, x2]:
        for y in [y1, (y1 + y2) // 2, y2]:
            if cv2.pointPolygonTest(zone, (float(x), float(y)), False) >= 0:
                return True
    return False


def match_pose_to_person(kps, box) -> bool:
    if kps is None or kps.shape[0] < 17:
        return False
    x1, y1, x2, y2 = box
    lh, rh = kps[11], kps[12]
    if lh[2] < 0.2 and rh[2] < 0.2:
        nose = kps[0]
        if nose[2] < 0.2:
            return False
        px, py = int(nose[0]), int(nose[1])
    else:
        px, py = int((lh[0] + rh[0]) / 2), int((lh[1] + rh[1]) / 2)
    return x1 <= px <= x2 and y1 <= py <= y2


# ─── Fall detection ─────────────────────────────────────────────────────────

def is_fallen(kps) -> bool:
    """Detect fall: head keypoint (nose) is at approximately the same height as hips/knees."""
    if kps is None or kps.shape[0] < 17:
        return False
    nose = kps[0]
    if nose[2] < KP_CONF_THRESH:
        return False
    hip_pts = [kps[i] for i in [11, 12] if kps[i, 2] >= KP_CONF_THRESH]
    if not hip_pts:
        return False
    avg_hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
    nose_y = nose[1]
    # Nose is within 30% of body height from hips → person is horizontal
    body_span = abs(avg_hip_y - nose_y)
    return body_span < 50 and abs(nose_y - avg_hip_y) < 80


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def draw_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y + 2), (10, 12, 16), -1)
    cv2.putText(frame, text, (x + 3, y - 2), FONT, 0.55, color, 1, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, color, label=""):
    L, t = 18, 2
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * L, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * L), color, t)
    if label:
        draw_label(frame, label, x1, y1, color)


def draw_zones(frame, zones):
    for zone in zones:
        arr = _zone_arr(zone["points"])
        if arr is None:
            continue
        color_hex = zone.get("color", "#0050dc").lstrip("#")
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        bgr = (b, g, r)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [arr], bgr)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [arr], True, bgr, 1, cv2.LINE_AA)
        # Label
        cx = int(arr[:, 0].mean())
        cy = int(arr[:, 1].mean())
        cv2.putText(frame, zone.get("name", "Zone"), (cx - 20, cy),
                    FONT, 0.4, bgr, 1, cv2.LINE_AA)


def draw_pose_overlay(frame, kps, hit_pts):
    hit_set = set(map(tuple, hit_pts))
    for idx in BODY_KP_INDICES:
        if kps[idx, 2] >= KP_CONF_THRESH:
            pt = (int(kps[idx, 0]), int(kps[idx, 1]))
            color = (0, 0, 255) if pt in hit_set else (0, 220, 220)
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 7, (255, 255, 255), 1)


def draw_hud(frame, counts):
    h = frame.shape[0]
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"● REC  {ts}", (8, 20), FONT, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    info = (f"PEOPLE:{counts['people']}  GUN:{counts.get('gun', 0)}  "
            f"KNIFE:{counts.get('knife', 0)}  RUN:{counts.get('running', 0)}  "
            f"LOIT:{counts.get('loiter', 0)}  FALL:{counts.get('fall', 0)}")
    cv2.putText(frame, info, (8, h - 10), FONT, 0.40, (80, 120, 160), 1, cv2.LINE_AA)


# ─── Main Pipeline ───────────────────────────────────────────────────────────

class DetectionPipeline:
    def __init__(self, models, zones: List[dict], settings):
        self.person_model, self.pose_model, self.weapon_model = models
        self.zones = zones          # list of {id, name, points, color}
        self.settings = settings

        self.tracker = CentroidTracker(max_disappeared=90, reid_ttl=60.0,
                                       reid_dist=300.0, match_dist=150.0, iou_assist=0.30)
        self.person_smoother = TemporalSmoother(window_size=3, min_hits=2, iou_match=0.40)
        self.weapon_smoother = TemporalSmoother(window_size=4, min_hits=3, iou_match=0.35)

        self.frame_count = 0
        self.fps_v = 0.0
        self.fps_t = time.time()

        self.last_person_data: List = []
        self.last_weapon_data: List = []
        self.last_pose_kps:    List = []

        self.last_alert_time: Dict[str, float] = defaultdict(float)
        self.zone_frames:     Dict = defaultdict(lambda: defaultdict(int))
        self.zone_alerted:    Dict = defaultdict(lambda: defaultdict(float))

        # Fall state: consecutive frames person appears fallen
        self.fall_frames: Dict[int, int] = defaultdict(int)
        FALL_CONFIRM = 3
        self.FALL_CONFIRM = FALL_CONFIRM

    def update_zones(self, zones: List[dict]):
        self.zones = zones

    def _should_alert(self, key: str, cooldown: float) -> bool:
        now = time.time()
        if now - self.last_alert_time[key] > cooldown:
            self.last_alert_time[key] = now
            return True
        return False

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict, List[dict]]:
        """
        Run the full detection pipeline on one frame.
        Returns: (annotated_frame, metrics_dict, new_alerts_list)
        """
        self.frame_count += 1
        cfg = self.settings
        new_alerts = []

        # Resize to 320×240 for faster CPU inference
        small = cv2.resize(frame, (320, 240))
        SX = frame.shape[1] / 320
        SY = frame.shape[0] / 240

        # ── Person detection (every 2 frames) ──────────────────────────────
        if self.frame_count % 2 == 0:
            res = self.person_model(small, imgsz=320, verbose=False)[0]
            raw = []
            for box in res.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= cfg.CONF_PERSON:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    raw.append((int(bx1 * SX), int(by1 * SY),
                                int(bx2 * SX), int(by2 * SY), float(box.conf[0])))
            raw = nms_boxes(raw, 0.45)
            confirmed = self.person_smoother.update(raw)
            self.last_person_data = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in confirmed]

        # ── Pose estimation (every 2 frames) ───────────────────────────────
        if self.frame_count % 2 == 0:
            pose_res = self.pose_model(small, imgsz=320, verbose=False)[0]
            kps_list = []
            if pose_res.keypoints is not None:
                for pkps in pose_res.keypoints.data:
                    kps = pkps.cpu().numpy().copy()
                    if kps.shape[0] < 17:
                        continue
                    kps[:, 0] *= SX; kps[:, 1] *= SY
                    kps_list.append(kps)
            self.last_pose_kps = kps_list

        # ── Weapon detection (every 3 frames) ──────────────────────────────
        if self.frame_count % 3 == 0:
            res = self.weapon_model(small, imgsz=320, verbose=False)[0]
            raw_w = []
            for box in res.boxes:
                cls_id = int(box.cls[0])
                if cls_id in ACTIVE_WEAPON_IDS and float(box.conf[0]) >= cfg.CONF_WEAPON:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    raw_w.append((int(bx1 * SX), int(by1 * SY),
                                  int(bx2 * SX), int(by2 * SY),
                                  float(box.conf[0]), cls_id))
            raw_w = nms_boxes(raw_w, 0.40)
            confirmed_w = self.weapon_smoother.update(raw_w)
            self.last_weapon_data = [(x1, y1, x2, y2, cid) for x1, y1, x2, y2, _, cid in confirmed_w]

            for _, _, _, _, cls_id in self.last_weapon_data:
                wname = WEAPON_NAMES.get(cls_id, "THREAT")
                if self._should_alert(f"weapon_{cls_id}", cfg.CD_WEAPON):
                    new_alerts.append({
                        "type": "weapon",
                        "message": f"{wname} DETECTED!",
                        "meta": {"weapon_type": wname.lower()}
                    })

        # ── Tracker update ──────────────────────────────────────────────────
        centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in self.last_person_data]
        tracked = self.tracker.update(centroids, boxes=list(self.last_person_data))

        # ── Draw zones ──────────────────────────────────────────────────────
        draw_zones(frame, self.zones)

        # Build zone arrays once
        zone_arrays = [(z, _zone_arr(z["points"])) for z in self.zones]

        running_count = 0
        loiter_count  = 0
        fall_count    = 0
        active_ids    = set()

        for oid, (cx, cy) in tracked.items():
            active_ids.add(oid)
            if not self.last_person_data:
                continue

            dists = [(abs(cx - (x1 + x2) // 2) + abs(cy - (y1 + y2) // 2), (x1, y1, x2, y2))
                     for x1, y1, x2, y2 in self.last_person_data]
            dists.sort()
            x1, y1, x2, y2 = dists[0][1]
            person_height = max(y2 - y1, 1)

            spd   = self.tracker.speed(oid, person_height=person_height)
            dwell = self.tracker.dwell(oid)
            tags  = [f"#{oid}"]
            color = (60, 220, 80)

            # Running
            if spd > cfg.RUN_THRESH_NORM:
                running_count += 1
                tags.append("RUN")
                color = (0, 100, 255)
                if self._should_alert(f"run_{oid}", cfg.CD_RUN):
                    new_alerts.append({
                        "type": "running",
                        "message": f"Person #{oid} is RUNNING",
                        "person_id": oid
                    })

            # Loitering
            if dwell > cfg.LOITER_SECS:
                loiter_count += 1
                tags.append(f"LOIT {dwell:.0f}s")
                if color == (60, 220, 80):
                    color = (0, 200, 255)

            # Fall detection
            matched_kps = None
            for kps in self.last_pose_kps:
                if match_pose_to_person(kps, (x1, y1, x2, y2)):
                    matched_kps = kps
                    break

            if matched_kps is not None and is_fallen(matched_kps):
                self.fall_frames[oid] = self.fall_frames.get(oid, 0) + 1
                if self.fall_frames[oid] >= self.FALL_CONFIRM:
                    fall_count += 1
                    tags.append("FALL")
                    color = (0, 50, 255)
                    if self._should_alert(f"fall_{oid}", cfg.CD_FALL):
                        new_alerts.append({
                            "type": "fall",
                            "message": f"Person #{oid} appears to have FALLEN",
                            "person_id": oid
                        })
            else:
                self.fall_frames[oid] = max(0, self.fall_frames.get(oid, 0) - 1)

            # Zone intrusion
            for zone, zone_arr in zone_arrays:
                if zone_arr is None:
                    continue
                body_in, hit_pts = False, []
                if matched_kps is not None:
                    body_in, hit_pts = keypoints_in_zone(zone_arr, matched_kps)
                    if hit_pts:
                        draw_pose_overlay(frame, matched_kps, hit_pts)
                else:
                    body_in = bbox_in_zone_fallback(zone_arr, x1, y1, x2, y2)

                zid = zone["id"]
                if body_in:
                    self.zone_frames[oid][zid] = min(self.zone_frames[oid][zid] + 1, 8)
                else:
                    self.zone_frames[oid][zid] = max(self.zone_frames[oid][zid] - 1, 0)

                if self.zone_frames[oid][zid] >= 3:
                    tags.append(f"ZONE:{zone['name']}")
                    color = (0, 0, 255)
                    now = time.time()
                    if now - self.zone_alerted[oid][zid] > cfg.CD_ZONE_BODY:
                        self.zone_alerted[oid][zid] = now
                        new_alerts.append({
                            "type": "zone_intrusion",
                            "message": f"Person #{oid} entered {zone['name']}!",
                            "person_id": oid,
                            "meta": {"zone_id": zid, "zone_name": zone["name"]}
                        })

            draw_box(frame, x1, y1, x2, y2, color, " | ".join(tags))

        # Clean stale per-person state
        for oid in list(self.zone_frames.keys()):
            if oid not in active_ids:
                del self.zone_frames[oid]
        for oid in list(self.zone_alerted.keys()):
            if oid not in active_ids:
                del self.zone_alerted[oid]
        for oid in list(self.fall_frames.keys()):
            if oid not in active_ids:
                del self.fall_frames[oid]

        # ── Crowd count ─────────────────────────────────────────────────────
        if len(self.last_person_data) >= cfg.CROWD_LIMIT:
            if self._should_alert("crowd", cfg.CD_CROWD):
                new_alerts.append({
                    "type": "crowd",
                    "message": f"CROWD ALERT: {len(self.last_person_data)} people detected"
                })

        # ── Weapon boxes ────────────────────────────────────────────────────
        gun_cnt = knife_cnt = 0
        for x1, y1, x2, y2, cls_id in self.last_weapon_data:
            wname  = WEAPON_NAMES.get(cls_id, "THREAT")
            wcolor = WEAPON_COLORS.get(cls_id, (0, 0, 255))
            draw_box(frame, x1, y1, x2, y2, wcolor, f"! {wname}")
            if wname == "GUN": gun_cnt += 1
            elif wname == "KNIFE": knife_cnt += 1

        # ── FPS ─────────────────────────────────────────────────────────────
        now = time.time()
        self.fps_v = 0.9 * self.fps_v + 0.1 * (1.0 / max(now - self.fps_t, 1e-6))
        self.fps_t = now
        cv2.putText(frame, f"FPS {self.fps_v:.1f}", (frame.shape[1] - 80, 18),
                    FONT, 0.45, (60, 140, 200), 1, cv2.LINE_AA)

        metrics = {
            "people": len(self.last_person_data),
            "gun": gun_cnt,
            "knife": knife_cnt,
            "running": running_count,
            "loiter": loiter_count,
            "fall": fall_count,
            "fps": round(self.fps_v, 1),
        }
        draw_hud(frame, metrics)

        return frame, metrics, new_alerts
