from dataclasses import dataclass

from typing import Tuple

@dataclass(frozen=True)
class Detection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float # 0.0 to 1.0
    class_id: int
    class_name: str
    