"""
Main file for the wurf-coach-project for the module BV3.
"""

import time
from typing import Optional, List
import cv2
from camera import close_camera, open_camera, read_frame
from config import (
    CAMERA_INDEX,
    DILATE_ITERATIONS,
    DRAW_THROW_ZONE,
    EVALUATION_GOOD_DELTA,
    EVALUATION_HISTORY_LENGTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIN_MOTION_AREA,
    SHOW_GRAY_WINDOW,
    SHOW_MASK_WINDOW,
    THRESHOLD_VALUE,
    THROW_COOLDOWN_FRAMES,
    THROW_HISTORY_LENGTH,
    THROW_MIN_UPWARD_DELTA,
    THROW_ZONE_END_RATIO,
    THROW_ZONE_RATIO,
    THROW_ZONE_START_RATIO,
)
from motion_detection import detect_motion
from preprocessing import preprocess_frame
from throw_detection import ThrowDetector
from evaluation import ThrowEvaluator


def draw_status(
    frame: cv2.Mat,  # pylint: disable=no-member
    status_text: str,
    alert: bool = False,
    line: int = 1,
) -> None:
    """
    Draw status text on the output frame.
    """

    # Define the color for the status:
    color = (0, 255, 0) if not alert else (0, 0, 255)

    # Calculate the position:
    y = 25 + (line - 1) * 30

    cv2.putText(  # pylint: disable=no-member
        frame,
        status_text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
        0.7,
        color,
        2,
        cv2.LINE_AA,  # pylint: disable=no-member
    )


def draw_boxes(frame: cv2.Mat, boxes: List) -> None:  # pylint: disable=no-member
    """
    Draw motion boxes on the output frame.
    """

    # Draw the boxes into the frame:
    for x, y, w, h in boxes:
        cv2.rectangle(  # pylint: disable=no-member
            frame, (x, y), (x + w, y + h), (0, 255, 0), 2
        )


def main() -> None:  # pylint: disable=too-many-locals
    """
    Main function for the wurf-coach-project.
    """

    # Try to open the camera stream:
    try:
        cap: cv2.VideoCapture = open_camera(CAMERA_INDEX)  # pylint: disable=no-member

        print("Camera is running.")
        print("Press 'q' to end the script.")

        previous_blurred: Optional[cv2.Mat] = None  # pylint: disable=no-member

        # Setup the class throw detector:
        throw_detector = ThrowDetector(
            history_length=THROW_HISTORY_LENGTH,
            min_upward_delta=THROW_MIN_UPWARD_DELTA,
            cooldown_frames=THROW_COOLDOWN_FRAMES,
            zone_start_ratio=THROW_ZONE_START_RATIO,
            zone_end_ratio=THROW_ZONE_END_RATIO,
        )

        # Setup the class throw evaluator:
        throw_evaluator = ThrowEvaluator(
            history_length=EVALUATION_HISTORY_LENGTH,
            good_delta=EVALUATION_GOOD_DELTA,
            zone_end_ratio=THROW_ZONE_END_RATIO,
        )

        # Run the program until the user stops the program:
        while True:
            # Get the next frame:
            frame = read_frame(cap)

            # Check if frame is a valide frame:
            if frame is None:
                print("Error, couldn't get the next frame from the camera.")
                break

            # Preprocess the frame:
            resized, gray, blurred = preprocess_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)

            # Copy the resized frame into a new variable:
            motion_view: cv2.Mat = resized.copy()  # pylint: disable=no-member

            # Check if there is a valide blurred previous frame:
            if previous_blurred is not None:
                # Detect motion in the frame:
                motion_mask, boxes = detect_motion(
                    previous_blurred,
                    blurred,
                    min_area=MIN_MOTION_AREA,
                    threshold_value=THRESHOLD_VALUE,
                    dilate_iterations=DILATE_ITERATIONS,
                )

                # Draw the boxes for the detected motion:
                draw_boxes(motion_view, boxes)

                # Update the throw detector:
                throw_detected, status_text = throw_detector.update(boxes, FRAME_HEIGHT)
                evaluation_label, evaluation_status = throw_evaluator.update(
                    boxes,
                    FRAME_HEIGHT,
                    throw_detected=throw_detected,
                )

                # Check if the throw zone should be drawn:
                if DRAW_THROW_ZONE:
                    zone_y: int = int(FRAME_HEIGHT * THROW_ZONE_RATIO)

                    cv2.line(  # pylint: disable=no-member
                        motion_view,
                        (0, zone_y),
                        (FRAME_WIDTH, zone_y),
                        (255, 0, 0),
                        2,
                    )

                # Check if there was a throw detected:
                if throw_detected:
                    # Draw the status:
                    draw_status(
                        motion_view,
                        evaluation_label,
                        alert=(evaluation_label != "Good attempt"),
                        line=1,
                    )
                    draw_status(
                        motion_view,
                        evaluation_status,
                        alert=(evaluation_label != "Good attempt"),
                        line=2,
                    )
                else:
                    draw_status(motion_view, status_text)

                # Check if the mask should be shown:
                if SHOW_MASK_WINDOW:
                    cv2.imshow("Motion mask", motion_mask)  # pylint: disable=no-member

            # Check if the gray scaled image should be displayed:
            if SHOW_GRAY_WINDOW:
                cv2.imshow("Grayscaled image", gray)  # pylint: disable=no-member

            # Display the detected motion:
            cv2.imshow("Camera", motion_view)  # pylint: disable=no-member

            # Save the current frame as previous blurred for the following frame:
            previous_blurred = blurred

            # Check if the progam should be stoped:
            if cv2.waitKey(1) & 0xFF == ord("q"):  # pylint: disable=no-member
                break

            # Delay the processing of the next frame by 0.1s:
            time.sleep(0.1)

    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        # Close the camera if it's still opened:
        if cap is not None:
            close_camera(cap)


if __name__ == "__main__":
    main()
