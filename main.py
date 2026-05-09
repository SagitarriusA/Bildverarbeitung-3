"""
Main file for the wurf-coach-project for the module BV3.
"""

import json
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
from camera import close_camera, open_camera, read_frame
from motion_detection import detect_motion
from preprocessing import preprocess_frame
from throw_detection import ThrowDetector
from evaluation import ThrowEvaluator


def calc_gradient_and_angle(
    coords: List[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate the gradient and release angle from the first two points of the trajectory.
    """

    # Need at least 2 points to calculate gradient and angle:
    if len(coords) < 2:
        print("Not enough points to calculate gradient and angle.")
        return None, None

    # Get the first two points:
    x1, y1 = coords[0]
    x2, y2 = coords[1]

    # Calculate the gradient and angle:
    dx = x2 - x1
    dy = y2 - y1

    # Handle the case where dx is zero to avoid division by zero:
    gradient = dy / dx if dx != 0 else float("inf")
    angle_deg = np.rad2deg(np.arctan2(-dy, dx))

    return gradient, angle_deg


def analyze_throw(
    gradient: Optional[float], angle: Optional[float], ideal_angle: int = 50
) -> str:
    """
    Analyze the throw based on the calculated gradient and angle.
    """

    # Check if we have valid gradient and angle values before analyzing:
    if gradient is None or angle is None:
        print("Cannot analyze throw without valid gradient and angle.")
        return "Insufficient data for analysis"

    # Print the value for the gradient and the angle for debugging and analysis:
    print(
        f"Throw analysis - Gradient: {gradient:.2f}, Release Angle: {angle:.2f} degrees"
    )

    # Analyze the throw based on the deviation from the ideal angle:
    deviation = abs(angle - ideal_angle)

    # Define thresholds for evaluation:
    if deviation <= 5:
        return "Good throw"

    if deviation <= 12:
        if angle < ideal_angle:
            return "Released too flat; release sooner"
        return "Released too steep; release later"

    if angle < ideal_angle:
        return "Too flat, release much sooner"

    return "Too steep, release much later"


def plot_throw_trajectory(
    coords: List[Tuple[float, float]],
) -> None:
    """
    Plot throw trajectory points and fit a 2nd degree polynomial.

    Args:
        coords: List of (x, y) coordinates.
    """

    # Need at least 3 points for quadratic fit
    if len(coords) < 3:
        print("Not enough points for polyfit.")
        return

    # Convert coordinates to numpy arrays
    x_coords = np.array([point[0] for point in coords])
    y_coords = np.array([point[1] for point in coords])

    # Fit polynomial of degree 2
    coefficients = np.polyfit(x_coords, y_coords, deg=2)

    # Create polynomial function
    polynomial = np.poly1d(coefficients)

    # Smooth x values for fitted curve
    x_smooth = np.linspace(
        x_coords.min(),
        x_coords.max(),
        300,
    )

    y_smooth = polynomial(x_smooth)

    # Plot
    plt.figure(figsize=(10, 6))

    # Original points as crosses
    plt.scatter(
        x_coords,
        y_coords,
        marker="x",
        s=100,
        label="Detected points",
    )

    # Polyfit curve
    plt.plot(
        x_smooth,
        y_smooth,
        label="2nd degree polyfit",
    )

    # Flip y-axis (camera coordinates)
    plt.gca().invert_yaxis()

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Throw trajectory")
    plt.grid(True)
    plt.legend()

    plt.show()


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
    evaluation_history_length: int = config_data.get("EVALUATION_HISTORY_LENGTH", 8)
    evaluation_good_delta: int = config_data.get("EVALUATION_GOOD_DELTA", 35)
    show_gray_window: bool = config_data.get("SHOW_GRAY_WINDOW", True)
    show_mask_window: bool = config_data.get("SHOW_MASK_WINDOW", True)
    draw_throw_zone: bool = config_data.get("DRAW_THROW_ZONE", True)

    paused: bool = False
    valide_throw: bool = False
    frame_cnt: int = 0
    frame_cnt_los: int = 0
    last_result: Optional[Tuple[np.ndarray, List, bool, str, str, str]] = None
    coords_history: List[Optional[Tuple[float, float]]] = []

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
                evaluation_label, evaluation_status = throw_evaluator.update(
                    boxes,
                    frame_height,
                    throw_detected=throw_detected,
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
                    print(f"Throw detected: {status_text} - {evaluation_label}")

                    # store last result for freezing view
                    last_result = (
                        motion_mask,
                        boxes,
                        throw_detected,
                        status_text,
                        evaluation_label,
                        evaluation_status,
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
                        print("valid throw frame")
                    else:
                        frame_cnt_los += 1
                        print("frame loss")

                if frame_cnt_los > 5:
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
                    throw_evaluator.reset()

                elif not valide_throw:
                    draw_status(motion_view, status_text)
                    # store last result for freezing view
                    last_result = (
                        motion_mask,
                        boxes,
                        throw_detected,
                        status_text,
                        evaluation_label,
                        evaluation_status,
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

            key = cv2.waitKey(1) & 0xFF  # pylint: disable=no-member

            # Check if the progam should be stoped:
            if key == ord("q"):  # pylint: disable=no-member
                paused = False
                break

            if key == ord("p"):  # pylint: disable=no-member
                paused = False
                print(valide_throw)
                print(coords_history)
                print(frame_cnt)
                print(frame_cnt_los)
                print(coords_history)

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
