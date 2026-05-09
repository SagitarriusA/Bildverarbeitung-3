"""
Main file for the wurf-coach-project for the module BV3.
"""

import json
from typing import Optional, List, Tuple
import cv2
import numpy as np
from camera import close_camera, open_camera, read_frame
from motion_detection import detect_motion
from preprocessing import preprocess_frame
from throw_detection import ThrowDetector
from evaluation import calc_gradient_and_angle, analyze_throw
from plotting import plot_throw_trajectory


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

    camera_index: int = config_data.get("CAMERA_INDEX", 0)
    frame_width: int = config_data.get("FRAME_WIDTH", 320)
    frame_height: int = config_data.get("FRAME_HEIGHT", 240)
    min_motion_area: int = config_data.get("MIN_MOTION_AREA", 500)
    dilate_iterations: int = config_data.get("DILATE_ITERATIONS", 2)
    throw_history_length: int = config_data.get("THROW_HISTORY_LENGTH", 8)
    throw_min_upward_delta: int = config_data.get("THROW_MIN_UPWARD_DELTA", 20)
    throw_zone_start_ratio: float = config_data.get("THROW_ZONE_START_RATIO", 0.65)
    throw_zone_end_ratio: float = config_data.get("THROW_ZONE_END_RATIO", 0.2)
    show_gray_window: bool = config_data.get("SHOW_GRAY_WINDOW", True)
    show_mask_window: bool = config_data.get("SHOW_MASK_WINDOW", True)
    draw_throw_zone: bool = config_data.get("DRAW_THROW_ZONE", True)

    paused: bool = False
    valide_throw: bool = False
    frame_cnt: int = 0
    frame_cnt_los: int = 0
    last_result: Optional[Tuple[np.ndarray, List, bool, str]] = None
    coords_history: List[Optional[Tuple[float, float]]] = []

    # Try to open the camera stream:
    try:
        cap: cv2.VideoCapture = open_camera(camera_index)  # pylint: disable=no-member

        print("Camera is running.")
        print("Press 'q' to end the script.")
        print("Press q to close the plot.")
        print("Press p to confirm the analysis.")

        previous_blurred: Optional[np.ndarray] = None  # pylint: disable=no-member

        # Setup the class throw detector:
        throw_detector = ThrowDetector(
            history_length=throw_history_length,
            min_upward_delta=throw_min_upward_delta,
            zone_start_ratio=throw_zone_start_ratio,
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
                    resized,
                    min_area=min_motion_area,
                    dilate_iterations=dilate_iterations,
                )

                # Draw the boxes for the detected motion:
                draw_boxes(motion_view, boxes)

                # Update the throw detector:
                throw_detected, status_text, center_x, center_y = throw_detector.update(
                    boxes, frame_height
                )

                # Check if the throw zone should be drawn:
                if draw_throw_zone:
                    zone_y_lower: int = int(frame_height * throw_zone_start_ratio)
                    zone_y_upper: int = int(frame_height * throw_zone_end_ratio)

                    cv2.line(  # pylint: disable=no-member
                        motion_view,
                        (0, zone_y_lower),
                        (frame_width, zone_y_lower),
                        (255, 0, 0),
                        2,
                    )

                    cv2.line(  # pylint: disable=no-member
                        motion_view,
                        (0, zone_y_upper),
                        (frame_width, zone_y_upper),
                        (255, 0, 0),
                        2,
                    )

                if valide_throw is False and throw_detected:
                    # store last result for freezing view
                    last_result = (
                        motion_mask,
                        boxes,
                        throw_detected,
                        status_text,
                    )
                    valide_throw = True
                    frame_cnt = 1
                    frame_cnt_los = 0

                    if center_x is not None and center_y is not None:
                        coords_history.append((center_x, center_y))
                elif valide_throw:
                    if throw_detected:
                        if center_x is not None and center_y is not None:
                            coords_history.append((center_x, center_y))
                        frame_cnt += 1
                    else:
                        frame_cnt_los += 1

                if frame_cnt_los > 5:
                    paused = True
                    print(coords_history)

                    clean_coords = [
                        coord for coord in coords_history if coord is not None
                    ]

                    if clean_coords:
                        gradient, angle = calc_gradient_and_angle(clean_coords)
                        analyze = analyze_throw(gradient, angle)
                        print(analyze)
                        plot_throw_trajectory(clean_coords)

                    frame_cnt = 0
                    frame_cnt_los = 0
                    valide_throw = False
                    coords_history.clear()
                    throw_detector.reset()

                elif not valide_throw:
                    draw_status(motion_view, status_text)
                    # store last result for freezing view
                    last_result = (
                        motion_mask,
                        boxes,
                        throw_detected,
                        status_text,
                    )

                # Check if the mask should be shown:
                if show_mask_window:
                    cv2.imshow("Motion mask", motion_mask)  # pylint: disable=no-member

            if paused and last_result is not None:
                (
                    motion_mask,
                    boxes,
                    throw_detected,
                    status_text,
                ) = last_result

                draw_boxes(motion_view, boxes)

            # Check if the gray scaled image should be displayed:
            if show_gray_window:
                cv2.imshow("Grayscaled image", gray)  # pylint: disable=no-member

            # Display the detected motion:
            cv2.imshow("Camera", motion_view)  # pylint: disable=no-member

            # Save the current frame as previous blurred for the following frame:
            previous_blurred = blurred

            key = cv2.waitKey(1) & 0xFF  # pylint: disable=no-member

            # Check if the progam should be stoped:
            if key == ord("q"):  # pylint: disable=no-member
                paused = False
                break

            if key == ord("p"):  # pylint: disable=no-member
                paused = False

    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        # Close the camera if it's still opened:
        if cap is not None:
            close_camera(cap)


if __name__ == "__main__":
    main()
