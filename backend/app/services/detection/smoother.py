from collections import deque
from typing import List

from app.services.detection.tracker import iou


class TemporalSmoother:
    def __init__(self, window_size=3, min_hits=2, iou_match=0.40):
        self.window = deque(maxlen=window_size)
        self.min_hits = min_hits
        self.iou_match = iou_match

    def update(self, new_detections: List) -> List:
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
