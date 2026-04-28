"""
File to detect the motion.
"""

import cv2


def detect_motion(
    previous_frame,
    current_frame,
    min_area=500,
    threshold_value=25,
    dilate_iterations=2,
):
    """Detect motion between two preprocessed frames and return motion mask and bounding boxes."""
    if previous_frame is None or current_frame is None:
        raise ValueError("Vorheriges und aktuelles Frame müssen vorhanden sein.")

    diff = cv2.absdiff(previous_frame, current_frame)
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=dilate_iterations)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))

    return thresh, boxes
