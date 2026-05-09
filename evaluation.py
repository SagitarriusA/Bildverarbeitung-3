"""
File to evaluate the frames.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple


class ThrowEvaluator:
    """
    Evaluate a detected throw as good or bad based on motion history.
    """

    def __init__(
        self,
        history_length: int = 8,
        good_delta: int = 35,
        zone_end_ratio: float = 0.35,
    ) -> None:
        """
        Initialize the evaluator with parameters for history length, good delta, and zone end ratio.
        """

        # Ensure parameters are within reasonable bounds and initialize history and status:
        self.history_length: int = max(3, history_length)
        self.good_delta: int = good_delta
        self.zone_end_ratio: float = max(0.0, min(1.0, zone_end_ratio))
        self.history: Deque[Optional[int]] = deque(maxlen=self.history_length)
        self.last_label: str = "No evaluation"
        self.last_status: str = "Waiting for movement"

    def reset(self) -> None:
        """
        Reset evaluator state and history.
        """

        # Clear the history and reset labels and status:
        self.history.clear()
        self.last_label = "No evaluation"
        self.last_status = "Set back to initial state"

    def _get_largest_box_center(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> Optional[int]:
        """
        Private method to get the center y-coordinate of the largest motion box.
        """

        # If no boxes are detected, return None:
        if not boxes:
            return None

        # Find the largest box based on area and return its center y-coordinate:
        largest_box: Tuple[int, int, int, int] = max(
            boxes, key=lambda box: box[2] * box[3]
        )
        _, y, _, h = largest_box

        return y + h // 2

    def update(
        self,
        boxes: List[Tuple[int, int, int, int]],
        frame_height: int,
        throw_detected: bool = False,
    ) -> Tuple[str, str]:
        """
        Update evaluator with boxes and return label and status.
        """

        # Get the center y-coordinate of the largest box and update history:
        center_y: Optional[int] = self._get_largest_box_center(boxes)
        self.history.append(center_y)

        # If no valid center is detected, we cannot evaluate the throw:
        if center_y is None:
            self.last_label = "No movement"
            self.last_status = "Waiting for movement"
            return self.last_label, self.last_status

        # If a throw has not been detected yet, we cannot evaluate it:
        if not throw_detected:
            self.last_label = "Movement"
            self.last_status = "No throw detected yet"
            return self.last_label, self.last_status

        # Filter out None values from history to analyze valid positions:
        valid_positions = [pos for pos in self.history if pos is not None]

        # If there are less than 2 valid positions, we cannot calculate a trend:
        if len(valid_positions) < 2:
            self.last_label = "Throw detected"
            self.last_status = "Waiting for evaluation"
            return self.last_label, self.last_status

        # Calculate the upward movement delta and height ratio to evaluate the throw:
        upward_delta = valid_positions[0] - valid_positions[-1]
        height_ratio = valid_positions[-1] / frame_height

        # Evaluate the throw based on the upward movement and height ratio vs defined thresholds:
        if upward_delta >= self.good_delta and height_ratio < self.zone_end_ratio:
            self.last_label = "Good throw"
            self.last_status = "Strong, clear upward movement"
        elif upward_delta >= self.good_delta:
            self.last_label = "Average throw"
            self.last_status = "Good movement, but not high enough"
        else:
            self.last_label = "Bad throw"
            self.last_status = "Little upward movement"

        return self.last_label, self.last_status
