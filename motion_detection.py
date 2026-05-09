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
    frame_bgr: np.ndarray,
    min_area: int = 60,
    max_area: int = 2000,
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

        if cv2.contourArea(contour) > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the roi where the box was detected:
        roi: np.ndarray = frame_bgr[y : y + h, x : x + w]

        # Convert the roi to hsv:
        hsv: np.ndarray = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define the color range for yellow and create a mask:
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([40, 255, 255])

        # Create a mask for the yellow color in the roi:
        mask: np.ndarray = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Calculate the ratio of yellow pixels in the roi:
        color_ratio = cv2.countNonZero(mask) / (w * h)

        # If the color ratio is too low, skip this box:
        if color_ratio < 0.25:
            continue

        boxes.append((x, y, w, h))
    # pylint: enable=no-member

    return processed_frame, boxes
