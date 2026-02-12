from __future__ import annotations

from pathlib import Path

import yaml

from prescience.config import AppSettings
from prescience.pipeline.count_stream import _load_zone_from_yaml
from prescience.pipeline.zone_count import ZoneCrossCounter
from prescience.types import TrackedDetection


def td(track_id: int, box: tuple[int, int, int, int]) -> TrackedDetection:
    return TrackedDetection(
        box=box,
        confidence=0.9,
        class_id=0,
        class_name="product",
        track_id=track_id,
    )


def test_counts_once_on_correct_direction_exit() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="left_to_right",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    decisions = counter.update([td(1, (130, 130, 170, 170))], frame_idx=1, now_seconds=0.0)
    assert len(decisions) == 0

    decisions = counter.update([td(1, (210, 130, 250, 170))], frame_idx=2, now_seconds=1.0)
    assert len(decisions) == 1
    assert decisions[0].track_id == 1
    assert counter.total_count == 1

    decisions = counter.update([td(1, (220, 130, 260, 170))], frame_idx=3, now_seconds=1.1)
    assert len(decisions) == 0
    assert counter.total_count == 1


def test_wrong_direction_does_not_count() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="left_to_right",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    counter.update([td(5, (150, 130, 190, 170))], frame_idx=1, now_seconds=0.0)
    decisions = counter.update([td(5, (60, 130, 95, 170))], frame_idx=2, now_seconds=1.0)

    assert len(decisions) == 0
    assert counter.total_count == 0


def test_in_and_out_counts_on_zone_exit_regardless_of_axis() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="in_and_out",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    counter.update([td(8, (150, 130, 190, 170))], frame_idx=1, now_seconds=0.0)
    decisions = counter.update([td(8, (70, 130, 95, 170))], frame_idx=2, now_seconds=1.0)
    assert len(decisions) == 1
    assert counter.total_count == 1


def test_track_oscillation_does_not_double_count() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="left_to_right",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    counter.update([td(9, (140, 120, 180, 160))], frame_idx=1, now_seconds=0.0)
    counter.update([td(9, (205, 120, 245, 160))], frame_idx=2, now_seconds=1.0)
    assert counter.total_count == 1

    counter.update([td(9, (150, 120, 190, 160))], frame_idx=3, now_seconds=1.2)
    counter.update([td(9, (210, 120, 250, 160))], frame_idx=4, now_seconds=1.4)
    assert counter.total_count == 1


def test_min_separation_suppresses_burst_counts() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="left_to_right",
        min_separation_seconds=0.5,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    counter.update([td(1, (130, 120, 170, 160)), td(2, (140, 140, 180, 180))], frame_idx=1, now_seconds=0.0)
    counter.update([td(1, (210, 120, 250, 160)), td(2, (145, 140, 185, 180))], frame_idx=2, now_seconds=1.0)
    assert counter.total_count == 1

    counter.update([td(2, (210, 140, 250, 180))], frame_idx=3, now_seconds=1.2)
    assert counter.total_count == 1


def test_counts_when_tracker_id_switches_mid_path() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="left_to_right",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    counter.update([td(1, (130, 120, 170, 160))], frame_idx=1, now_seconds=0.0)
    decisions = counter.update([td(99, (210, 120, 250, 160))], frame_idx=2, now_seconds=1.0)
    assert len(decisions) == 1
    assert counter.total_count == 1


def test_counts_when_tracker_id_missing_but_motion_is_consistent() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="left_to_right",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    inside = TrackedDetection(
        box=(130, 120, 170, 160),
        confidence=0.9,
        class_id=0,
        class_name="product",
        track_id=None,
    )
    outside = TrackedDetection(
        box=(210, 120, 250, 160),
        confidence=0.9,
        class_id=0,
        class_name="product",
        track_id=None,
    )

    counter.update([inside], frame_idx=1, now_seconds=0.0)
    decisions = counter.update([outside], frame_idx=2, now_seconds=1.0)
    assert len(decisions) == 1
    assert counter.total_count == 1


def test_left_to_right_counts_on_right_edge_cross_even_if_centroid_still_inside() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="left_to_right",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    counter.update([td(3, (140, 120, 180, 160))], frame_idx=1, now_seconds=0.0)
    # Right edge crosses zone boundary while centroid can still be inside.
    decisions = counter.update([td(3, (170, 120, 210, 160))], frame_idx=2, now_seconds=1.0)
    assert len(decisions) == 1
    assert counter.total_count == 1


def test_in_and_out_counts_when_any_edge_crosses_zone_boundary() -> None:
    counter = ZoneCrossCounter(
        polygon=[(100, 100), (200, 100), (200, 200), (100, 200)],
        direction="in_and_out",
        min_separation_seconds=0.1,
        track_cooldown_seconds=0.1,
        max_track_idle_frames=30,
    )

    counter.update([td(4, (120, 120, 160, 160))], frame_idx=1, now_seconds=0.0)
    decisions = counter.update([td(4, (90, 120, 130, 160))], frame_idx=2, now_seconds=1.0)
    assert len(decisions) == 1
    assert counter.total_count == 1


def test_zone_config_round_trip_loader(tmp_path: Path) -> None:
    path = tmp_path / "line.yaml"
    payload = {
        "zone": {
            "polygon": [[10, 20], [110, 20], [110, 120], [10, 120]],
            "direction": "top_to_bottom",
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    polygon, direction = _load_zone_from_yaml(path, AppSettings())
    assert polygon == [(10, 20), (110, 20), (110, 120), (10, 120)]
    assert direction == "top_to_bottom"
