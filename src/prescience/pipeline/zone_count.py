"""Zone crossing counting with anti-double-count controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from prescience.types import TrackedDetection


Direction = str


@dataclass(frozen=True)
class CountEventDecision:
    track_id: int
    centroid: tuple[int, int]


@dataclass
class _TrackState:
    centroid: tuple[int, int]
    inside: bool
    last_frame_idx: int
    counted: bool = False
    last_count_time: float | None = None


class ZoneCrossCounter:
    """Count each tracked object once when exiting zone in configured direction."""

    def __init__(
        self,
        polygon: list[tuple[int, int]],
        direction: Direction,
        min_separation_seconds: float,
        track_cooldown_seconds: float,
        max_track_idle_frames: int,
    ) -> None:
        if len(polygon) < 3:
            raise ValueError("Zone polygon requires at least 3 points")

        self.polygon = polygon
        self.direction = direction
        self.min_separation_seconds = min_separation_seconds
        self.track_cooldown_seconds = track_cooldown_seconds
        self.max_track_idle_frames = max_track_idle_frames

        self.total_count = 0
        self._states: dict[int, _TrackState] = {}
        self._last_global_count_time = -1e9

    @staticmethod
    def _centroid(box: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _inside_polygon(self, point: tuple[int, int]) -> bool:
        polygon_arr = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon_arr, point, False) >= 0

    def _direction_ok(self, prev_pt: tuple[int, int], curr_pt: tuple[int, int]) -> bool:
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]

        if self.direction == "left_to_right":
            return dx > 0
        if self.direction == "right_to_left":
            return dx < 0
        if self.direction == "top_to_bottom":
            return dy > 0
        if self.direction == "bottom_to_top":
            return dy < 0
        return True

    def update(
        self,
        tracked: Iterable[TrackedDetection],
        frame_idx: int,
        now_seconds: float,
    ) -> list[CountEventDecision]:
        """Process tracked detections for one frame and return new counts."""
        decisions: list[CountEventDecision] = []

        for detection in tracked:
            if detection.track_id is None:
                continue

            track_id = int(detection.track_id)
            centroid = self._centroid(detection.box)
            inside = self._inside_polygon(centroid)
            prev = self._states.get(track_id)

            counted = False
            if prev is not None:
                exited_zone = prev.inside and not inside
                cooldown_ok = (
                    prev.last_count_time is None
                    or (now_seconds - prev.last_count_time) >= self.track_cooldown_seconds
                )
                global_sep_ok = (now_seconds - self._last_global_count_time) >= self.min_separation_seconds

                if exited_zone and not prev.counted and cooldown_ok and global_sep_ok:
                    if self._direction_ok(prev.centroid, centroid):
                        counted = True

            state = prev or _TrackState(centroid=centroid, inside=inside, last_frame_idx=frame_idx)
            state.centroid = centroid
            state.inside = inside
            state.last_frame_idx = frame_idx

            if counted:
                state.counted = True
                state.last_count_time = now_seconds
                self._last_global_count_time = now_seconds
                self.total_count += 1
                decisions.append(CountEventDecision(track_id=track_id, centroid=centroid))

            self._states[track_id] = state

        self._cleanup_stale_tracks(frame_idx)
        return decisions

    def _cleanup_stale_tracks(self, frame_idx: int) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if (frame_idx - state.last_frame_idx) > self.max_track_idle_frames
        ]
        for track_id in stale_ids:
            self._states.pop(track_id, None)


def draw_zone_overlay(frame: np.ndarray, polygon: list[tuple[int, int]], direction: str, total_count: int) -> None:
    """Draw zone polygon and count information on frame."""
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.putText(
        frame,
        f"Direction: {direction}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Total: {total_count}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
