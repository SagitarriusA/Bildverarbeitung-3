"""
File to detect the motion.
"""

from typing import List, Tuple
import numpy as np
import cv2

# Create once and reuse
BACKGROUND_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(  # pylint: disable=no-member
    history=100,
    varThreshold=25,
    detectShadows=False,
)


def detect_motion(
    current_frame: np.ndarray,
    min_area: int = 100,
    dilate_iterations: int = 2,
) -> Tuple[np.ndarray, List]:
    """
    Detect motion using a background subtractor.
    """

    # pylint: disable=no-member
    foreground_mask: np.ndarray = BACKGROUND_SUBTRACTOR.apply(current_frame)

    # Clean noise
    kernel: np.ndarray = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    processed_frame: np.ndarray = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    processed_frame = cv2.dilate(
        processed_frame,
        kernel,
        iterations=dilate_iterations,
    )

    contours, _ = cv2.findContours(
        processed_frame,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes: list = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    # pylint: enable=no-member

    return processed_frame, boxes
