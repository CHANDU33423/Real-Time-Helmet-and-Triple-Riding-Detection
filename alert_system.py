"""
alert_system.py — Evidence capture and violation alerting

Responsibilities:
- Save annotated violation frames to disk with timestamp
- Write structured log entries to a CSV file
- (Optional) Send email/webhook alerts
- Enforce cooldown between duplicate saves
"""

import csv
import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import requests

from config import (
    BOX_THICKNESS,
    COLOR_HELMET,
    COLOR_MOTORCYCLE,
    COLOR_NO_HELMET,
    COLOR_VIOLATION,
    COOLDOWN_SECONDS,
    FONT_SCALE,
    LOGS_DIR,
    VIOLATIONS_DIR,
)
from logic_engine import ViolationResult
from notification_service import NotificationService

# --- Directory Bootstrap ---
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "violations_log.csv")
_LOG_HEADERS = ["timestamp", "frame_no", "violation_type", "rider_count", "image_path"]

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_LOG_HEADERS)


class AlertSystem:
    """
    Manages violation evidence capture and logging.

    Usage:
        alert = AlertSystem()
        alert.process(frame, result, frame_number)
    """

    def __init__(self, location: str = "Main Road Camera"):
        self.location = location
        self._last_save_time: float = 0.0
        self._total_violations = 0

    # --- Public API ---

    def process(
        self,
        frame: np.ndarray,
        result: ViolationResult,
        frame_number: int,
        plate_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save evidence and log if a violation is found (subject to cooldown).

        Args:
            frame:        Original (non-preprocessed) BGR frame
            result:       ViolationResult from logic_engine
            frame_number: Current frame index in the video
            plate_text:   OCR-detected number plate (if available)

        Returns:
            Path to saved image, or None if cooldown active / no violation
        """
        if not result.has_violation:
            return None

        now = time.time()
        if now - self._last_save_time < COOLDOWN_SECONDS:
            return None  # Cooldown active — skip duplicate save

        self._last_save_time = now
        self._total_violations += 1

        annotated = self._annotate_frame(frame, result, plate_text)
        img_path = self._save_image(annotated, frame_number)
        self._log_violation(result, frame_number, img_path, plate_text)

        return img_path

    @property
    def total_violations(self) -> int:
        return self._total_violations

    # --- Internal Helpers ---

    def _annotate_frame(
        self, frame: np.ndarray, result: ViolationResult, plate_text: Optional[str]
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and violation banner on the frame."""
        annotated = frame.copy()

        # Draw motorcycle groups
        for group in result.groups:
            mx1, my1, mx2, my2 = group.motorcycle.box
            m_color = (
                COLOR_VIOLATION
                if (group.is_triple_riding or group.has_helmet_violation)
                else COLOR_MOTORCYCLE
            )
            cv2.rectangle(annotated, (mx1, my1), (mx2, my2), m_color, BOX_THICKNESS)
            cv2.putText(
                annotated,
                f"Motorcycle ({group.rider_count} riders)",
                (mx1, my1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                m_color,
                2,
            )

            # Draw helmeted riders
            for rider in group.helmeted_riders:
                r = rider.box
                cv2.rectangle(
                    annotated, (r[0], r[1]), (r[2], r[3]), COLOR_HELMET, BOX_THICKNESS
                )
                cv2.putText(
                    annotated,
                    "Helmet OK",
                    (r[0], r[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    COLOR_HELMET,
                    1,
                )

            # Draw helmetless riders
            for rider in group.helmetless_riders:
                r = rider.box
                cv2.rectangle(
                    annotated,
                    (r[0], r[1]),
                    (r[2], r[3]),
                    COLOR_NO_HELMET,
                    BOX_THICKNESS,
                )
                cv2.putText(
                    annotated,
                    "NO HELMET!",
                    (r[0], r[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    COLOR_NO_HELMET,
                    2,
                )

        # Lone helmetless riders (no motorcycle match)
        for rider in result.lone_helmetless_riders:
            r = rider.box
            cv2.rectangle(
                annotated, (r[0], r[1]), (r[2], r[3]), COLOR_NO_HELMET, BOX_THICKNESS
            )
            cv2.putText(
                annotated,
                "NO HELMET!",
                (r[0], r[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                COLOR_NO_HELMET,
                2,
            )

        banner_text = f"!! VIOLATION: {result.violation_summary()}"
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (0, 0), (tw + 20, th + 20), (0, 0, 150), -1)
        cv2.putText(
            annotated,
            banner_text,
            (10, th + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Timestamp + location
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        loc_text = f"{self.location} | {ts}"
        cv2.putText(
            annotated,
            loc_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Plate text (if OCR available)
        if plate_text:
            cv2.putText(
                annotated,
                f"Plate: {plate_text}",
                (10, frame.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

        return annotated

    def _save_image(self, frame: np.ndarray, frame_number: int) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{ts}_frame{frame_number}.jpg"
        path = os.path.join(VIOLATIONS_DIR, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[Alert] Evidence saved -> {path}")
        return path

    def _log_violation(
        self,
        result: ViolationResult,
        frame_number: int,
        img_path: str,
        plate_text: Optional[str] = None,
    ):
        """Append one row per violation type to the CSV log."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []

        for g in result.triple_riding_groups:
            v_type = "Triple Riding"
            rows.append([ts, frame_number, v_type, g.rider_count, img_path])
            self._notify_api(v_type, g.rider_count, plate_text)

        for g in result.helmet_violation_groups:
            v_type = (
                "No Helmet" if not g.is_triple_riding else "No Helmet + Triple Riding"
            )
            rows.append([ts, frame_number, v_type, len(g.helmetless_riders), img_path])
            self._notify_api(v_type, g.rider_count, plate_text)

        for _ in result.lone_helmetless_riders:
            v_type = "No Helmet"
            rows.append([ts, frame_number, v_type, 1, img_path])
            self._notify_api(v_type, 1, plate_text)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

        # Trigger Real-Time Notification Dispatch
        NotificationService.send_violation_alert(
            v_type=result.violation_summary(),
            plate=plate_text if plate_text else "UNKNOWN",
            location=self.location,
        )

    def _notify_api(
        self, violation_type: str, rider_count: int, plate_text: Optional[str]
    ):
        """Send violation data to the running Flask dashboard in the background."""

        def task():
            try:
                data = {
                    "vehicle_no": plate_text if plate_text else "UNKNOWN",
                    "violation_type": violation_type,
                    "location": self.location,
                    "rider_count": rider_count,
                }
                requests.post(
                    "http://127.0.0.1:5000/api/violations/add", json=data, timeout=2
                )
            except Exception as e:
                pass  # Fail silently if dashboard is not running

        threading.Thread(target=task, daemon=True).start()
