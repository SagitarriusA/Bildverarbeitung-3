"""
File for the throw detection.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple


class ThrowDetector:
    """Simple throw detector based on upward motion history."""

    def __init__(
        self,
        history_length: int = 8,
        min_upward_delta: int = 20,
        cooldown_frames: int = 30,
        zone_start_ratio: float = 0.65,
        zone_end_ratio: float = 0.35,
    ):
        self.history_length = max(3, history_length)
        self.min_upward_delta = min_upward_delta
        self.cooldown_frames = max(0, cooldown_frames)
        self.zone_start_ratio = max(0.0, min(1.0, zone_start_ratio))
        self.zone_end_ratio = max(0.0, min(1.0, zone_end_ratio))
        self.history: Deque[Optional[int]] = deque(maxlen=self.history_length)
        self.cooldown = 0
        self.last_status = "Warten auf Bewegung"

    def reset(self) -> None:
        """Reset detector state and history."""
        self.history.clear()
        self.cooldown = 0
        self.last_status = "Zurückgesetzt"

    def _get_largest_box_center(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> Optional[int]:
        if not boxes:
            return None

        largest_box = max(boxes, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_box
        return y + h // 2

    def _calculate_trend(self) -> Optional[int]:
        valid_positions = [
            position for position in self.history if position is not None
        ]
        if len(valid_positions) < 2:
            return None
        return valid_positions[-1] - valid_positions[0]

    def update(
        self, boxes: List[Tuple[int, int, int, int]], frame_height: int
    ) -> Tuple[bool, str]:
        """Update detector with current motion boxes and return detection state."""
        center_y = self._get_largest_box_center(boxes)
        self.history.append(center_y)

        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_status = "Cooldown"
            return False, self.last_status

        if center_y is None:
            self.last_status = "Keine Bewegung"
            return False, self.last_status

        trend = self._calculate_trend()
        if trend is None:
            self.last_status = "Bewegung verfolgt"
            return False, self.last_status

        valid_positions = [p for p in self.history if p is not None]
        if not valid_positions:
            self.last_status = "Warten auf Bewegung"
            return False, self.last_status

        zone_start = frame_height * self.zone_start_ratio
        zone_end = frame_height * self.zone_end_ratio
        if (
            trend < -self.min_upward_delta
            and valid_positions[0] > zone_start
            and valid_positions[-1] < zone_end
        ):
            self.cooldown = self.cooldown_frames
            self.last_status = "WURF ERKANNT"
            return True, self.last_status

        self.last_status = "Bewegung erkannt"
        return False, self.last_status
