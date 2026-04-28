"""
File for the preprocessing
"""

import cv2


def preprocess_frame(frame, width=320, height=240):
    """Resize frame, convert to gray scale and apply Gaussian blur."""
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return resized, gray, blurred
