"""
File to generate plots for throw trajectory.
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_throw_trajectory(
    coords: List[Tuple[float, float]],
) -> None:
    """
    Plot throw trajectory points and fit a 2nd degree polynomial.
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
