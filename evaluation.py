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
    ):
        self.history_length = max(3, history_length)
        self.good_delta = good_delta
        self.zone_end_ratio = max(0.0, min(1.0, zone_end_ratio))
        self.history: Deque[Optional[int]] = deque(maxlen=self.history_length)
        self.last_label = "Keine Auswertung"
        self.last_status = "Warten auf Bewegung"

    def reset(self) -> None:
        """Reset evaluator state and history."""
        self.history.clear()
        self.last_label = "Keine Auswertung"
        self.last_status = "Zurückgesetzt"

    def _get_largest_box_center(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> Optional[int]:
        if not boxes:
            return None

        largest_box = max(boxes, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_box
        return y + h // 2

    def update(
        self,
        boxes: List[Tuple[int, int, int, int]],
        frame_height: int,
        throw_detected: bool = False,
    ) -> Tuple[str, str]:
        """Update evaluator with boxes and return label and status."""
        center_y = self._get_largest_box_center(boxes)
        self.history.append(center_y)

        if center_y is None:
            self.last_label = "Keine Bewegung"
            self.last_status = "Warten auf Bewegung"
            return self.last_label, self.last_status

        if not throw_detected:
            self.last_label = "Bewegung"
            self.last_status = "Noch kein Wurf"
            return self.last_label, self.last_status

        valid_positions = [pos for pos in self.history if pos is not None]
        if len(valid_positions) < 2:
            self.last_label = "Wurf erkannt"
            self.last_status = "Warte auf Auswertung"
            return self.last_label, self.last_status

        upward_delta = valid_positions[0] - valid_positions[-1]
        height_ratio = valid_positions[-1] / frame_height

        if upward_delta >= self.good_delta and height_ratio < self.zone_end_ratio:
            self.last_label = "Guter Wurf"
            self.last_status = "Starker, klarer Aufwärtszug"
        elif upward_delta >= self.good_delta:
            self.last_label = "Mittelguter Wurf"
            self.last_status = "Gute Bewegung, aber noch nicht hoch genug"
        else:
            self.last_label = "Schlechter Wurf"
            self.last_status = "Wenig Aufwärtsbewegung"

        return self.last_label, self.last_status
