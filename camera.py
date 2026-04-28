"""
File to setup the camera.
"""

import cv2


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:  # pylint: disable=no-member
    """
    Function to open and return a configured video capture device.
    """

    # Open the camera device:
    cap: cv2.VideoCapture = cv2.VideoCapture(camera_index)  # pylint: disable=no-member

    # Check if the camera has been opened:
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} couldn't be opened.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # pylint: disable=no-member
    return cap


def read_frame(cap: cv2.VideoCapture, flip: bool = True):  # pylint: disable=no-member
    """
    Read a single frame from the camera and optionally mirror it.
    """

    # Check if cap is valide:
    if cap is None or not hasattr(cap, "read"):
        raise RuntimeError("Invalide camera object.")

    # Read the frame from the camera:
    ret, frame = cap.read()

    # Check if the camera returned a valide frame:
    if not ret or frame is None:
        return None

    # Check if the frame should be fliped:
    if flip:
        frame = cv2.flip(frame, 1)  # pylint: disable=no-member

    return frame


def close_camera(cap: cv2.VideoCapture) -> None:  # pylint: disable=no-member
    """
    Release camera resources and close all open windows.
    """

    # Check if cap is valide and then close it:
    if cap is not None:
        try:
            cap.release()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    cv2.destroyAllWindows()  # pylint: disable=no-member
