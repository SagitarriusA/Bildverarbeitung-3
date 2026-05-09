"""
File to evaluate the frames.
"""

from typing import List, Optional, Tuple
import numpy as np


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
