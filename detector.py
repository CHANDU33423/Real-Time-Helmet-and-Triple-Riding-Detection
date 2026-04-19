"""
detector.py — YOLOv8 inference wrapper
Loads the model, runs inference, and returns structured detections.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from config import (
    CLASS_HELMET,
    CLASS_MOTORCYCLE,
    CLASS_NO_HELMET,
    CLASS_RIDER,
    CONFIDENCE_THRESHOLD,
    FALLBACK_WEIGHTS,
    INPUT_SIZE,
    IOU_THRESHOLD,
    MODEL_WEIGHTS,
    USE_CUSTOM_MODEL,
)


@dataclass
class Detection:
    """Represents a single detected object in a frame."""

    class_id: int
    class_name: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2] in original frame coords
    track_id: Optional[int] = None


@dataclass
class FrameDetections:
    """All detections for a single video frame."""

    motorcycles: List[Detection] = field(default_factory=list)
    riders: List[Detection] = field(default_factory=list)
    helmets: List[Detection] = field(default_factory=list)
    no_helmets: List[Detection] = field(default_factory=list)
    raw: List[Detection] = field(default_factory=list)


class HelmetDetector:
    """
    Wraps a YOLOv8 model for helmet and rider detection.

    Behaviour:
    - If `models/best.pt` exists -> loads your custom-trained weights.
    - Otherwise -> falls back to standard yolov8n.pt (COCO) for pipeline testing.
      In COCO mode helmet detection is disabled (class not present).
    """

    def __init__(self):
        weights = MODEL_WEIGHTS if USE_CUSTOM_MODEL else FALLBACK_WEIGHTS
        print(f"[Detector] Loading model: {weights}")
        if USE_CUSTOM_MODEL:
            print("[Detector] Custom model detected -- helmet/rider classes active.")
        else:
            print("[Detector] WARNING: No custom model found. Using COCO fallback.")
            print("[Detector]          Place your trained weights at: models/best.pt")

        self.model = YOLO(weights)
        # BUG FIX #9: model.conf / model.iou are deprecated in Ultralytics >= 8.0.20.
        # Setting them on the model object has no effect. Confidence/IoU are passed
        # directly in predict() / track() calls below. Removed dead assignments.
        self._class_names = self.model.names  # e.g. {0: 'motorcycle', 1: 'rider', ...}

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: parse boxes → FrameDetections
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_boxes(self, results) -> FrameDetections:
        fd = FrameDetections()

        if not results or results[0].boxes is None:
            return fd

        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = self._class_names.get(class_id, str(class_id))

            # BUG FIX #10 & #11: box.id may not exist at all (not just be None)
            # when tracking is disabled.  Use getattr with a None default.
            raw_id = getattr(box, "id", None)
            track_id = int(raw_id[0]) if raw_id is not None else None

            det = Detection(
                class_id=class_id,
                class_name=name,
                confidence=confidence,
                box=[x1, y1, x2, y2],
                track_id=track_id,
            )
            fd.raw.append(det)

            if class_id == CLASS_MOTORCYCLE:
                fd.motorcycles.append(det)
            elif class_id == CLASS_RIDER:
                fd.riders.append(det)
            elif CLASS_HELMET is not None and class_id == CLASS_HELMET:
                fd.helmets.append(det)
            elif CLASS_NO_HELMET is not None and class_id == CLASS_NO_HELMET:
                fd.no_helmets.append(det)

        return fd

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Run YOLOv8 inference on a preprocessed frame and return categorised detections.

        Args:
            frame: BGR numpy array (already preprocessed to INPUT_SIZE x INPUT_SIZE)

        Returns:
            FrameDetections with sorted lists per category
        """
        results = self.model.predict(
            source=frame,
            imgsz=INPUT_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
        return self._parse_boxes(results)

    def track(self, frame: np.ndarray) -> FrameDetections:
        """
        Run YOLOv8 with built-in ByteTrack for persistent IDs across frames.
        Use this instead of detect() for better identity tracking.
        """
        results = self.model.track(
            source=frame,
            imgsz=INPUT_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return self._parse_boxes(results)
