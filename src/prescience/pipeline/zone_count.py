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
    box: tuple[int, int, int, int]
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
        id_stitch_max_distance_px: float = 90.0,
        id_stitch_max_gap_frames: int = 4,
    ) -> None:
        if len(polygon) < 3:
            raise ValueError("Zone polygon requires at least 3 points")

        self.polygon = polygon
        self.direction = direction
        self.min_separation_seconds = min_separation_seconds
        self.track_cooldown_seconds = track_cooldown_seconds
        self.max_track_idle_frames = max_track_idle_frames
        self.id_stitch_max_distance_px = id_stitch_max_distance_px
        self.id_stitch_max_gap_frames = id_stitch_max_gap_frames

        self.total_count = 0
        self._states: dict[int, _TrackState] = {}
        self._last_global_count_time = -1e9
        self._next_anonymous_track_id = -1
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        self._zone_left = int(min(xs))
        self._zone_right = int(max(xs))
        self._zone_top = int(min(ys))
        self._zone_bottom = int(max(ys))

    @staticmethod
    def _centroid(box: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _normalize_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    def _inside_polygon(self, point: tuple[int, int]) -> bool:
        polygon_arr = np.array(self.polygon, dtype=np.int32)
        return cv2.pointPolygonTest(polygon_arr, point, False) >= 0

    def _box_overlaps_zone_bounds(self, box: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = box
        return not (
            x2 < self._zone_left
            or x1 > self._zone_right
            or y2 < self._zone_top
            or y1 > self._zone_bottom
        )

    def _direction_ok(self, prev_pt: tuple[int, int], curr_pt: tuple[int, int]) -> bool:
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]

        if self.direction == "in_and_out":
            return True
        if self.direction == "left_to_right":
            return dx > 0
        if self.direction == "right_to_left":
            return dx < 0
        if self.direction == "top_to_bottom":
            return dy > 0
        if self.direction == "bottom_to_top":
            return dy < 0
        return True

    def _distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _find_stitch_candidate(
        self,
        *,
        track_id: int,
        centroid: tuple[int, int],
        frame_idx: int,
    ) -> int | None:
        """Find recently seen, nearby unmatched state likely to be same object after ID switch."""
        best_track_id: int | None = None
        best_distance = float("inf")
        for candidate_id, state in self._states.items():
            if candidate_id == track_id:
                continue
            if state.counted:
                continue
            gap = frame_idx - state.last_frame_idx
            if gap <= 0 or gap > self.id_stitch_max_gap_frames:
                continue
            distance = self._distance(centroid, state.centroid)
            if distance > self.id_stitch_max_distance_px:
                continue
            if distance < best_distance:
                best_distance = distance
                best_track_id = candidate_id
        return best_track_id

    def _resolve_track_id(self, detection: TrackedDetection, frame_idx: int) -> int:
        """Use model track_id when present, otherwise maintain anonymous IDs by proximity."""
        if detection.track_id is not None:
            return int(detection.track_id)
        centroid = self._centroid(detection.box)
        for candidate_id, state in self._states.items():
            if candidate_id >= 0:
                continue
            gap = frame_idx - state.last_frame_idx
            if gap <= 0 or gap > self.id_stitch_max_gap_frames:
                continue
            if self._distance(centroid, state.centroid) <= self.id_stitch_max_distance_px:
                return candidate_id
        anon_id = self._next_anonymous_track_id
        self._next_anonymous_track_id -= 1
        return anon_id

    def _edge_exit_crossed(
        self,
        prev_box: tuple[int, int, int, int],
        curr_box: tuple[int, int, int, int],
    ) -> bool:
        px1, py1, px2, py2 = prev_box
        cx1, cy1, cx2, cy2 = curr_box
        crossed_right = px2 < self._zone_right and cx2 >= self._zone_right
        crossed_left = px1 > self._zone_left and cx1 <= self._zone_left
        crossed_bottom = py2 < self._zone_bottom and cy2 >= self._zone_bottom
        crossed_top = py1 > self._zone_top and cy1 <= self._zone_top

        if self.direction == "left_to_right":
            return crossed_right
        if self.direction == "right_to_left":
            return crossed_left
        if self.direction == "top_to_bottom":
            return crossed_bottom
        if self.direction == "bottom_to_top":
            return crossed_top
        if self.direction == "in_and_out":
            return crossed_right or crossed_left or crossed_bottom or crossed_top
        return False

    def update(
        self,
        tracked: Iterable[TrackedDetection],
        frame_idx: int,
        now_seconds: float,
    ) -> list[CountEventDecision]:
        """Process tracked detections for one frame and return new counts."""
        decisions: list[CountEventDecision] = []

        for detection in tracked:
            track_id = self._resolve_track_id(detection, frame_idx)
            box = self._normalize_box(detection.box)
            centroid = self._centroid(box)
            inside = self._inside_polygon(centroid)
            prev = self._states.get(track_id)
            if prev is None:
                stitched = self._find_stitch_candidate(
                    track_id=track_id,
                    centroid=centroid,
                    frame_idx=frame_idx,
                )
                if stitched is not None:
                    prev = self._states.pop(stitched)
                    self._states[track_id] = prev

            counted = False
            if prev is not None:
                crossed_exit_edge = self._edge_exit_crossed(prev.box, box)
                was_in_or_touching_zone = prev.inside or self._box_overlaps_zone_bounds(prev.box)
                cooldown_ok = (
                    prev.last_count_time is None
                    or (now_seconds - prev.last_count_time) >= self.track_cooldown_seconds
                )
                global_sep_ok = (now_seconds - self._last_global_count_time) >= self.min_separation_seconds

                if crossed_exit_edge and was_in_or_touching_zone and not prev.counted and cooldown_ok and global_sep_ok:
                    counted = True

            state = prev or _TrackState(centroid=centroid, box=box, inside=inside, last_frame_idx=frame_idx)
            state.centroid = centroid
            state.box = box
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
