"""
Main file for the wurf-coach-project for the module BV3.
"""

# import time
import json
from typing import Optional, List, Tuple
import cv2
import numpy as np
from camera import close_camera, open_camera, read_frame
from motion_detection import detect_motion
from preprocessing import preprocess_frame
from throw_detection import ThrowDetector
from evaluation import ThrowEvaluator


def draw_status(
    frame: np.ndarray,  # pylint: disable=no-member
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


def draw_boxes(frame: np.ndarray, boxes: List) -> None:  # pylint: disable=no-member
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

    with open("config.json", encoding="utf-8", mode="r") as config_file:
        config_data = json.load(config_file)

    camera_index: int = config_data.get("camera_index", 0)
    frame_width: int = config_data.get("frame_width", 320)
    frame_height: int = config_data.get("frame_height", 240)
    min_motion_area: int = config_data.get("min_motion_area", 500)
    dilate_iterations: int = config_data.get("dilate_iterations", 2)
    throw_history_length: int = config_data.get("throw_history_length", 8)
    throw_min_upward_delta: int = config_data.get("throw_min_upward_delta", 20)
    throw_cooldown_frames: int = config_data.get("throw_cooldown_frames", 30)
    throw_zone_start_ratio: float = config_data.get("throw_zone_start_ratio", 0.65)
    throw_zone_end_ratio: float = config_data.get("throw_zone_end_ratio", 0.35)
    evaluation_history_length: int = config_data.get("evaluation_history_length", 8)
    evaluation_good_delta: int = config_data.get("evaluation_good_delta", 35)
    show_gray_window: bool = config_data.get("show_gray_window", True)
    show_mask_window: bool = config_data.get("show_mask_window", True)
    draw_throw_zone: bool = config_data.get("draw_throw_zone", True)
    throw_zone_ratio: float = config_data.get("throw_zone_ratio", 0.4)

    paused: bool = False
    last_result: Optional[Tuple[np.ndarray, List, bool, str, str, str]] = None

    # Try to open the camera stream:
    try:
        cap: cv2.VideoCapture = open_camera(camera_index)  # pylint: disable=no-member

        print("Camera is running.")
        print("Press 'q' to end the script.")

        previous_blurred: Optional[np.ndarray] = None  # pylint: disable=no-member

        # Setup the class throw detector:
        throw_detector = ThrowDetector(
            history_length=throw_history_length,
            min_upward_delta=throw_min_upward_delta,
            cooldown_frames=throw_cooldown_frames,
            zone_start_ratio=throw_zone_start_ratio,
            zone_end_ratio=throw_zone_end_ratio,
        )

        # Setup the class throw evaluator:
        throw_evaluator = ThrowEvaluator(
            history_length=evaluation_history_length,
            good_delta=evaluation_good_delta,
            zone_end_ratio=throw_zone_end_ratio,
        )

        # Run the program until the user stops the program:
        while True:
            # Get the next frame:
            frame = read_frame(cap)

            # Check if frame is a valid frame:
            if frame is None:
                print("Error, couldn't get the next frame from the camera.")
                break

            # Preprocess the frame:
            resized, gray, blurred = preprocess_frame(frame, frame_width, frame_height)

            # Copy the resized frame into a new variable:
            motion_view: np.ndarray = resized.copy()  # pylint: disable=no-member

            # Check if there is a valid blurred previous frame:
            if not paused and previous_blurred is not None:
                # Detect motion in the frame:
                motion_mask, boxes = detect_motion(
                    blurred,
                    min_area=min_motion_area,
                    dilate_iterations=dilate_iterations,
                )

                # Draw the boxes for the detected motion:
                draw_boxes(motion_view, boxes)

                # Update the throw detector:
                throw_detected, status_text = throw_detector.update(boxes, frame_height)
                evaluation_label, evaluation_status = throw_evaluator.update(
                    boxes,
                    frame_height,
                    throw_detected=throw_detected,
                )

                # Check if the throw zone should be drawn:
                if draw_throw_zone:
                    zone_y: int = int(frame_height * throw_zone_ratio)

                    cv2.line(  # pylint: disable=no-member
                        motion_view,
                        (0, zone_y),
                        (frame_width, zone_y),
                        (255, 0, 0),
                        2,
                    )

                # Check if there was a throw detected:
                if throw_detected:
                    paused = True
                    print(f"Throw detected: {status_text} - {evaluation_label}")
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
                if show_mask_window:
                    cv2.imshow("Motion mask", motion_mask)  # pylint: disable=no-member

                # store last result for freezing view
                last_result = (
                    motion_mask,
                    boxes,
                    throw_detected,
                    status_text,
                    evaluation_label,
                    evaluation_status,
                )

            if paused and last_result is not None:
                (
                    motion_mask,
                    boxes,
                    throw_detected,
                    status_text,
                    evaluation_label,
                    evaluation_status,
                ) = last_result

                draw_boxes(motion_view, boxes)

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

            # Check if the gray scaled image should be displayed:
            if show_gray_window:
                cv2.imshow("Grayscaled image", gray)  # pylint: disable=no-member

            # Display the detected motion:
            cv2.imshow("Camera", motion_view)  # pylint: disable=no-member

            # Save the current frame as previous blurred for the following frame:
            previous_blurred = blurred

            # Check if the progam should be stoped:
            if cv2.waitKey(1) & 0xFF == ord("q"):  # pylint: disable=no-member
                paused = False
                break

            if cv2.waitKey(1) & 0xFF == ord("p"):  # pylint: disable=no-member
                paused = False

            # Delay the processing of the next frame by 0.1s:
            # time.sleep(0.1)

    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        # Close the camera if it's still opened:
        if cap is not None:
            close_camera(cap)


if __name__ == "__main__":
    main()
