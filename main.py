"""
main.py — Real-Time Helmet and Triple Riding Detection System
Entry point: reads video feed, runs the full detection pipeline, displays output.

Usage:
    python main.py                         # live camera (default)
    python main.py --source path/video.mp4 # video file
    python main.py --source 0 --no-display # headless mode
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from alert_system import AlertSystem
from config import (
    BOX_THICKNESS,
    COLOR_HELMET,
    COLOR_MOTORCYCLE,
    COLOR_NO_HELMET,
    COLOR_RIDER,
    COLOR_SAFE,
    COLOR_VIOLATION,
    ENABLE_OCR,
    FONT_SCALE,
    OUTPUT_DIR,
    VIDEO_SOURCE,
    WINDOW_NAME,
)
from detector import FrameDetections, HelmetDetector
from logic_engine import analyse_frame
from preprocessing import preprocess_frame, scale_coords_back

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Optional OCR ---
ocr_reader = None
if ENABLE_OCR:
    from ocr_module import NumberPlateReader

    ocr_reader = NumberPlateReader()


# --- HUD Overlay ---


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_no: int,
    total_violations: int,
    violation_text: str,
) -> np.ndarray:
    """Draw a semi-transparent HUD panel on the live display frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top-right stats panel
    panel_w, panel_h = 280, 130
    x1, y1 = w - panel_w - 10, 10
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    lines = [
        f"FPS:        {fps:.1f}",
        f"Frame:      {frame_no}",
        f"Violations: {total_violations}",
        "",
        violation_text,
    ]
    colors = [
        (200, 200, 200),
        (200, 200, 200),
        (0, 200, 255),
        (200, 200, 200),
        COLOR_VIOLATION if violation_text.startswith("!!") else COLOR_SAFE,
    ]
    for i, (line, col) in enumerate(zip(lines, colors)):
        cv2.putText(
            frame,
            line,
            (x1 + 10, y1 + 25 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            1,
        )

    # Bottom bar — system label
    label = "Real-Time Helmet & Triple Riding Detection System  |  YOLOv8"
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(
        frame, label, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 255), 1
    )

    return frame


def draw_live_boxes(frame: np.ndarray, fd: FrameDetections) -> np.ndarray:
    """Draw all detection boxes on the live display frame (not saved)."""
    for det in fd.motorcycles:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_MOTORCYCLE, BOX_THICKNESS)
        cv2.putText(
            frame,
            f"Motorcycle {det.confidence:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            COLOR_MOTORCYCLE,
            1,
        )

    for det in fd.riders:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RIDER, BOX_THICKNESS)
        tid = f" #{det.track_id}" if det.track_id else ""
        cv2.putText(
            frame,
            f"Rider{tid}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            COLOR_RIDER,
            1,
        )

    for det in fd.helmets:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HELMET, BOX_THICKNESS)

    for det in fd.no_helmets:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NO_HELMET, BOX_THICKNESS)
        cv2.putText(
            frame,
            "NO HELMET",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            COLOR_NO_HELMET,
            2,
        )

    return frame


# --- Main Detection Loop ---


def main(source, show_display: bool, location: str):
    print("\n" + "=" * 60)
    print("  Helmet & Triple Riding Detection System — Starting")
    print("=" * 60)

    # Initialise components
    detector = HelmetDetector()
    alert = AlertSystem(location=location)

    # Open video source
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    fps_display = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[Stream] Source opened. Native FPS: {fps_display:.1f}")

    frame_no = 0
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0

    if show_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("[System] Running — Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Re-check if it's intermittent or end of stream
            time.sleep(0.5)
            ret, frame = cap.read()
            if not ret:
                print("[Stream] End of video or stream lost.")
                break

        frame_no += 1
        original_shape = frame.shape

        # ── Module 2: Preprocessing ──────────────────────────────────────────
        preprocessed = preprocess_frame(frame)

        # ── Module 3: Detection ──────────────────────────────────────────────
        fd = detector.track(preprocessed)

        # Scale boxes back to original resolution
        all_dets = fd.motorcycles + fd.riders + fd.helmets + fd.no_helmets + fd.raw
        for det in all_dets:
            det.box = scale_coords_back([det.box], original_shape)[0]

        # ── Module 4: Logic Engine ───────────────────────────────────────────
        result = analyse_frame(fd)

        # ── Module 5: Alert System ───────────────────────────────────────────
        plate_text = None
        if ENABLE_OCR and result.has_violation and ocr_reader:
            # Run OCR on motorcycle bounding boxes for number plate
            for g in result.triple_riding_groups + result.helmet_violation_groups:
                plate_text = ocr_reader.read_plate(frame, g.motorcycle.box)
                if plate_text:
                    break

        alert.process(frame, result, frame_no, plate_text)

        # ── FPS Calculation ──────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        # ── Module 7: Display ────────────────────────────────────────────────
        if show_display:
            display_frame = draw_live_boxes(frame.copy(), fd)
            v_text = (
                f"!! {result.violation_summary()}"
                if result.has_violation
                else result.safe_summary()
            )
            display_frame = draw_hud(
                display_frame, current_fps, frame_no, alert.total_violations, v_text
            )
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[System] User quit.")
                break

    cap.release()
    if show_display:
        cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  Session complete. Total violations captured: {alert.total_violations}")
    print(f"  Evidence saved to: output/violations/")
    print(f"  Log saved to:      output/logs/violations_log.csv")
    print(f"{'='*60}\n")


# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Helmet & Triple Riding Detection System"
    )
    parser.add_argument(
        "--source",
        default=str(VIDEO_SOURCE),
        help="Video source: 0 for webcam, or path to .mp4 file (default: 0)",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Run in headless mode (no GUI window)"
    )
    parser.add_argument(
        "--location",
        default="Main Road Camera",
        help="Location label for violation logs",
    )
    args = parser.parse_args()
    main(args.source, not args.no_display, args.location)
