"""
File for the preprocessing of the frame.
"""

from typing import Tuple
import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray, width: int = 320, height: int = 240
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the current frame.
    """

    # pylint: disable=no-member
    # Resize the frame:
    resized: np.ndarray = cv2.resize(
        frame, (width, height), interpolation=cv2.INTER_AREA
    )

    # Convert the frame to gray scale:
    gray: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Add the gaussian blur onto the frame:
    blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)
    # pylint: enable=no-member

    return resized, gray, blurred
