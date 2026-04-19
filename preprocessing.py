"""
preprocessing.py — Frame preprocessing pipeline
Resize, denoise, contrast enhancement before YOLO inference.
"""

import cv2
import numpy as np

from config import INPUT_SIZE


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied to each raw video frame.

    Steps:
        1. Resize to INPUT_SIZE x INPUT_SIZE
        2. Apply CLAHE for contrast enhancement (handles low-light / night)
        3. Apply slight Gaussian blur for noise reduction
        4. Return the processed frame (BGR, uint8)
    """
    resized = _resize(frame)
    enhanced = _enhance_contrast(resized)
    denoised = _denoise(enhanced)
    return denoised


def _resize(frame: np.ndarray) -> np.ndarray:
    """Resize frame to square INPUT_SIZE while preserving aspect ratio with padding."""
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox padding to reach INPUT_SIZE x INPUT_SIZE
    pad_top = (INPUT_SIZE - new_h) // 2
    pad_bottom = INPUT_SIZE - new_h - pad_top
    pad_left = (INPUT_SIZE - new_w) // 2
    pad_right = INPUT_SIZE - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded


def _enhance_contrast(frame: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel.
    Improves detection under varying light conditions / night footage.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def _denoise(frame: np.ndarray) -> np.ndarray:
    """Light Gaussian blur to suppress sensor noise."""
    return cv2.GaussianBlur(frame, (3, 3), 0)


def scale_coords_back(coords: list, original_shape: tuple) -> list:
    """
    Scale bounding box coordinates back from preprocessed 640x640 space
    to the original frame dimensions.

    Args:
        coords:  list of [x1, y1, x2, y2]
        original_shape: (orig_h, orig_w)

    Returns:
        Scaled list of [x1, y1, x2, y2]
    """
    orig_h, orig_w = original_shape[:2]
    scale = INPUT_SIZE / max(orig_h, orig_w)
    pad_left = (INPUT_SIZE - int(orig_w * scale)) // 2
    pad_top = (INPUT_SIZE - int(orig_h * scale)) // 2

    scaled = []
    for x1, y1, x2, y2 in coords:
        x1 = int((x1 - pad_left) / scale)
        y1 = int((y1 - pad_top) / scale)
        x2 = int((x2 - pad_left) / scale)
        y2 = int((y2 - pad_top) / scale)
        # Clamp to image bounds
        x1, x2 = max(0, x1), min(orig_w, x2)
        y1, y2 = max(0, y1), min(orig_h, y2)
        scaled.append([x1, y1, x2, y2])
    return scaled
