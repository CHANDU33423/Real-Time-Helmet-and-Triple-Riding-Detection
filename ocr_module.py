"""
ocr_module.py — Number Plate Recognition (Module 6)

Uses EasyOCR for licence plate text extraction.
Called by main.py only when ENABLE_OCR=True and a violation is detected.
"""

from typing import Optional

import cv2
import numpy as np


class NumberPlateReader:
    """
    Wraps EasyOCR for number plate recognition.
    Activate in config.py: ENABLE_OCR = True
    """

    def __init__(self):
        try:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[OCR] EasyOCR initialised ✅")
        except ImportError:
            self._reader = None
            print("[OCR] ⚠️  EasyOCR not installed. Run: pip install easyocr")

    def read_plate(self, frame: np.ndarray, box: list) -> Optional[str]:
        """
        Extract number plate text from a bounding box region.

        Args:
            frame: Full BGR frame
            box:   [x1, y1, x2, y2] of the motorcycle/plate region

        Returns:
            Detected plate text or None
        """
        if self._reader is None:
            return None

        x1, y1, x2, y2 = box
        # Focus on bottom third of bike box — where plate usually is
        plate_y1 = y1 + int((y2 - y1) * 0.65)
        crop = frame[max(0, plate_y1) : y2, max(0, x1) : x2]

        if crop.size == 0:
            return None

        # Preprocess crop for better OCR
        crop = self._preprocess_for_ocr(crop)

        results = self._reader.readtext(crop, detail=0, paragraph=False)
        if results:
            plate = " ".join(results).upper().strip()
            return plate if len(plate) >= 4 else None  # filter noise
        return None

    @staticmethod
    def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
        """Enhance the plate crop for maximum OCR accuracy."""
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. Denoising to remove sensor noise/grain
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 3. Increase Contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # 4. Sharpen using a kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(contrast, -1, kernel)

        # 5. Upscale for better OCR text recognition
        upscaled = cv2.resize(
            sharpened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

        # 6. Final Thresholding (Binary Inverse often works better for OCR)
        _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh
