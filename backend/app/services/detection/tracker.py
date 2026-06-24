import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np


def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter)


def nms_boxes(boxes_confs, iou_thresh=0.45):
    if not boxes_confs:
        return []
    sorted_b = sorted(boxes_confs, key=lambda b: b[4], reverse=True)
    keep = []
    while sorted_b:
        best = sorted_b.pop(0)
        keep.append(best)
        sorted_b = [b for b in sorted_b if iou(best[:4], b[:4]) < iou_thresh]
    return keep


class CentroidTracker:
    def __init__(self, max_disappeared=45, reid_ttl=20.0,
                 reid_dist=200.0, match_dist=150.0, iou_assist=0.30):
        self.next_id = 0
        self.objects: Dict[int, Tuple] = {}
        self.boxes: Dict[int, Tuple] = {}
        self.velocity: Dict[int, Tuple] = {}
        self.disappeared: Dict[int, int] = defaultdict(int)
        self.history: Dict[int, deque] = {}
        self.entry_time: Dict[int, float] = {}
        self.speed_ema: Dict[int, float] = {}
        self.lost_objects: Dict[int, dict] = {}
        self.max_disappeared = max_disappeared
        self.reid_ttl = reid_ttl
        self.reid_dist = reid_dist
        self.match_dist = match_dist
        self.iou_assist = iou_assist

    def update(self, centroids: List[Tuple], boxes: Optional[List] = None) -> Dict[int, Tuple]:
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
            ids = list(self.objects.keys())
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

            # Second-chance pass: a still-live track that narrowly missed its
            # one candidate column above would otherwise sit unmatched while a
            # brand-new ID gets minted for the same detection (the old track
            # isn't deregistered yet, so the lost_objects recovery in
            # _register never triggers). Retry remaining rows/cols against
            # each other with relaxed thresholds before minting new IDs, so
            # jitter on a single subject doesn't flicker between two IDs.
            remaining_rows = [r for r in range(len(ids)) if r not in used_rows]
            remaining_cols = [c for c in range(len(centroids)) if c not in used_cols]
            if remaining_rows and remaining_cols:
                candidates = []
                for r in remaining_rows:
                    oid = ids[r]
                    for c in remaining_cols:
                        d = D[r, c]
                        box_iou = (iou(self.boxes[oid], boxes[c])
                                   if boxes[c] is not None and self.boxes.get(oid) is not None
                                   else 0.0)
                        if d <= self.match_dist * 2.0 or box_iou >= self.iou_assist * 0.5:
                            candidates.append((d, r, c))
                candidates.sort()
                for d, r, c in candidates:
                    if r in used_rows or c in used_cols:
                        continue
                    oid = ids[r]
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
        self.objects[oid] = c
        self.boxes[oid] = box
        self.velocity[oid] = (0, 0)
        self.disappeared[oid] = 0
        if oid not in self.history:
            self.history[oid] = deque(maxlen=10)
        self.history[oid].append(c)
        if oid not in self.entry_time:
            self.entry_time[oid] = time.time()

    def _deregister(self, oid):
        if oid in self.objects:
            self.lost_objects[oid] = {
                "centroid": self.objects[oid],
                "last_seen": time.time(),
                "history": list(self.history.get(oid, [])),
                "entry_time": self.entry_time.get(oid, time.time()),
                "box": self.boxes.get(oid),
            }
            del self.objects[oid]
        for store in [self.disappeared, self.history, self.entry_time,
                      self.velocity, self.boxes, self.speed_ema]:
            if oid in store:
                del store[oid]

    def _recover_id(self, c) -> int:
        best_id, best_d = None, float("inf")
        for oid, data in self.lost_objects.items():
            d = float(np.linalg.norm(np.array(data["centroid"]) - np.array(c)))
            if d < best_d:
                best_d = d; best_id = oid
        if best_id is not None and best_d <= self.reid_dist:
            state = self.lost_objects.pop(best_id)
            self.history[best_id] = deque(state.get("history", []), maxlen=10)
            if not self.history[best_id]:
                self.history[best_id].append(c)
            self.entry_time[best_id] = state.get("entry_time", time.time())
            self.boxes[best_id] = state.get("box")
            self.velocity[best_id] = (0, 0)
            return best_id
        oid = self.next_id
        self.next_id += 1
        return oid

    def _cleanup_lost(self):
        now = time.time()
        stale = [oid for oid, d in self.lost_objects.items()
                 if now - d.get("last_seen", now) > self.reid_ttl]
        for oid in stale:
            del self.lost_objects[oid]

    def speed(self, oid: int, person_height: float = 1.0) -> float:
        h = list(self.history.get(oid, []))
        if len(h) < 2:
            return 0.0
        look_back = min(4, len(h) - 1)
        raw = float(np.linalg.norm(np.array(h[-1]) - np.array(h[-(look_back + 1)]))) / max(look_back, 1)
        norm = raw / max(person_height, 1.0)
        prev = self.speed_ema.get(oid, norm)
        s = 0.35 * norm + 0.65 * prev
        self.speed_ema[oid] = s
        return s

    def dwell(self, oid: int) -> float:
        return time.time() - self.entry_time.get(oid, time.time())

    def get_box(self, oid: int) -> Optional[Tuple]:
        return self.boxes.get(oid)
