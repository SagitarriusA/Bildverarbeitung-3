"""
File to detect the throw detection.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple


class ThrowDetector:  # pylint: disable=too-many-instance-attributes
    """
    Class to analyze the throw, based on motion history.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        history_length: int = 8,
        min_upward_delta: int = 20,
        cooldown_frames: int = 30,
        zone_start_ratio: float = 0.65,
        zone_end_ratio: float = 0.35,
    ) -> None:
        """
        Init function for the class.
        """

        # Set the max frames for the history:
        self.history_length: int = max(3, history_length)
        self.min_upward_delta: int = min_upward_delta
        self.cooldown_frames: int = max(0, cooldown_frames)
        self.zone_start_ratio: float = max(0.0, min(1.0, zone_start_ratio))
        self.zone_end_ratio: float = max(0.0, min(1.0, zone_end_ratio))
        self.history: Deque[Optional[int]] = deque(maxlen=self.history_length)
        self.cooldown: int = 0
        self.last_status: str = "Waiting for movement"

    def reset(self) -> None:
        """
        Reset detector state and history.
        """

        # Clear the history and reset cooldown and status:
        self.history.clear()
        self.cooldown = 0
        self.last_status = "Zurückgesetzt"

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

    def _calculate_trend(self) -> Optional[int]:
        """
        Private method to calculate the vertical movement trend based on history.
        """

        # Filter out None values from history and calculate the trend:
        valid_positions = [
            position for position in self.history if position is not None
        ]

        # If there are less than 2 valid positions, we cannot calculate a trend:
        if len(valid_positions) < 2:
            return None

        return valid_positions[-1] - valid_positions[0]

    def update(
        self, boxes: List[Tuple[int, int, int, int]], frame_height: int
    ) -> Tuple[bool, str]:
        """
        Update detector with current motion boxes and return detection state.
        """

        # Get the center y-coordinate of the largest box and update history:
        center_y = self._get_largest_box_center(boxes)
        self.history.append(center_y)

        # If we are in cooldown, decrement it and return the current status:
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_status = "Cooldown"
            return False, self.last_status

        # If no valid center is detected, we cannot analyze movement:
        if center_y is None:
            self.last_status = "No motion detected"
            return False, self.last_status

        # Calculate the movement trend and analyze it against the defined zones:
        trend: Optional[int] = self._calculate_trend()

        # If we cannot calculate a trend, we are still tracking movement:
        if trend is None:
            self.last_status = "Detect motion"
            return False, self.last_status

        # Filter out None values from history to analyze valid positions:
        valid_positions: List[int] = [p for p in self.history if p is not None]

        # If there are no valid positions, we cannot analyze movement:
        if not valid_positions:
            self.last_status = "No motion detected"
            return False, self.last_status

        # Define the zones based on the frame height and check if the trend indicates a throw:
        zone_start: float = frame_height * self.zone_start_ratio
        zone_end: float = frame_height * self.zone_end_ratio

        # Check if the trend indicates a upward movement and if the positions are within the zones:
        if (
            trend < -self.min_upward_delta
            and valid_positions[0] > zone_start
            and valid_positions[-1] < zone_end
        ):
            self.cooldown = self.cooldown_frames
            self.last_status = "WURF ERKANNT"
            return True, self.last_status

        # If the trend does not indicate a throw, update the status accordingly:
        self.last_status = "No motion detected"
        return False, self.last_status
