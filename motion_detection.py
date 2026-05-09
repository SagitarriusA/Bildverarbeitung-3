"""
File to detect the motion.
"""

from typing import Tuple, List
import numpy as np
import cv2


def detect_motion(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    min_area: int = 500,
    threshold_value: int = 25,
    dilate_iterations: int = 2,
) -> Tuple[np.ndarray, List]:
    """
    Detect motion between two preprocessed frames and return motion mask and bounding boxes.
    """

    # Check if there is a valid previous and valid current frame:
    if previous_frame is None or current_frame is None:
        raise ValueError(
            "Not enought frames, either current or previous frame missing."
        )

    # pylint: disable=no-member
    # Calculate the difference in the frames:
    diff: np.ndarray = cv2.absdiff(previous_frame, current_frame)

    # Set the threshold to dilate the frame:
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_frame = cv2.dilate(thresh, kernel, iterations=dilate_iterations)

    # Detect the contours on the frame:
    contours, _ = cv2.findContours(
        processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: list = []

    # Find the interesting contours in the frame:
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # Calculate the bounding box for the detected contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    # pylint: enable=no-member

    return thresh, boxes
