"""Interactive zone calibration helper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from prescience.config import parse_source


DIRECTION_MAP = {
    ord("1"): "left_to_right",
    ord("2"): "right_to_left",
    ord("3"): "top_to_bottom",
    ord("4"): "bottom_to_top",
}


def calibrate_zone_config(source: str, out_path: Path, line_id: str) -> None:
    """Capture one frame, collect polygon clicks, save zone yaml."""
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame from source")

    points: list[tuple[int, int]] = []
    direction = "left_to_right"

    window = "Zone Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        disp = frame.copy()

        for p in points:
            cv2.circle(disp, p, 4, (0, 255, 255), -1)
        if len(points) >= 2:
            cv2.polylines(disp, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
        if len(points) >= 3:
            cv2.polylines(disp, [np.array(points, dtype=np.int32)], True, (0, 180, 255), 2)

        instructions = [
            "Left click: add point",
            "u: undo last point, c: clear",
            "1:left->right 2:right->left 3:top->bottom 4:bottom->top",
            "s: save  q: quit",
            f"Current direction: {direction}",
            f"Points: {len(points)}",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(
                disp,
                line,
                (10, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF

        if key in DIRECTION_MAP:
            direction = DIRECTION_MAP[key]
        elif key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key == ord("s"):
            if len(points) < 3:
                print("Need at least 3 points to save polygon")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "line": {"line_id": line_id},
                "zone": {
                    "polygon": [[int(x), int(y)] for x, y in points],
                    "direction": direction,
                },
            }
            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            print(f"Saved zone config to {out_path}")
            break
        elif key == ord("q"):
            break

    cv2.destroyWindow(window)
