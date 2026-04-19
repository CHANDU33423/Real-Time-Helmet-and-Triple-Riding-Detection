"""
config.py — Global configuration for Helmet & Triple Riding Detection System
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VIOLATIONS_DIR = os.path.join(OUTPUT_DIR, "violations")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Model weights — replace with your custom-trained best.pt when available
# Standard YOLOv8n is used by default for pipeline validation
MODEL_WEIGHTS = os.path.join(BASE_DIR, "models", "best.pt")
FALLBACK_WEIGHTS = os.path.join(BASE_DIR, "models", "yolov8n.pt")  # auto-downloaded if best.pt not found

# ─────────────────────────────────────────────────────────────────────────────
# YOLO Class IDs  (update after custom training)
# When using standard YOLOv8 COCO weights temporarily:
#   person=0, motorcycle=3
# When using custom trained model:
#   motorcycle=0, rider=1, helmet=2, no_helmet=3
# ─────────────────────────────────────────────────────────────────────────────
USE_CUSTOM_MODEL = os.path.exists(MODEL_WEIGHTS)

if USE_CUSTOM_MODEL:
    CLASS_MOTORCYCLE = 0
    CLASS_RIDER = 1
    CLASS_HELMET = 2
    CLASS_NO_HELMET = 3
else:
    # COCO fallback for pipeline testing
    CLASS_MOTORCYCLE = 3  # motorcycle in COCO
    CLASS_RIDER = 0  # person in COCO
    CLASS_HELMET = None  # not in COCO
    CLASS_NO_HELMET = None

# ─────────────────────────────────────────────────────────────────────────────
# Detection Parameters
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE = 640  # YOLO inference image size (px)
CONFIDENCE_THRESHOLD = 0.45  # Minimum detection confidence
IOU_THRESHOLD = 0.5  # Non-max suppression IoU threshold

# ─────────────────────────────────────────────────────────────────────────────
# Violation Logic
# ─────────────────────────────────────────────────────────────────────────────
TRIPLE_RIDING_THRESHOLD = 3  # flag if riders on ONE motorcycle >= this value
OVERLAP_IOU_THRESHOLD = 0.2  # IoU threshold to associate rider/helmet with bike
COOLDOWN_SECONDS = 3  # Min seconds between saving duplicate violations

# ─────────────────────────────────────────────────────────────────────────────
# Video Input
# ─────────────────────────────────────────────────────────────────────────────
# Set to 0 for live webcam/CCTV, or provide a path to an .mp4 file
VIDEO_SOURCE = 0

# Display
SHOW_WINDOW = True
WINDOW_NAME = "Helmet & Triple Riding Detection"
FONT_SCALE = 0.6
BOX_THICKNESS = 2

# ─────────────────────────────────────────────────────────────────────────────
# Colors (BGR format for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
COLOR_MOTORCYCLE = (255, 165, 0)  # Orange
COLOR_RIDER = (0, 255, 255)  # Cyan
COLOR_HELMET = (0, 255, 0)  # Green
COLOR_NO_HELMET = (0, 0, 255)  # Red
COLOR_VIOLATION = (0, 0, 255)  # Red
COLOR_SAFE = (0, 255, 0)  # Green
COLOR_OVERLAY = (20, 20, 20)  # Dark panel background

# ─────────────────────────────────────────────────────────────────────────────
# OCR (Number Plate — Module 6)
# ─────────────────────────────────────────────────────────────────────────────
ENABLE_OCR = True  # Set True to enable number plate recognition
OCR_LANGUAGE = "en"
