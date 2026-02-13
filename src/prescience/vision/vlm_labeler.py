"""Gemini-based VLM box proposal helpers for onboarding flows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional runtime dependency
    genai = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None


class GeminiLabelerError(RuntimeError):
    """Raised when Gemini labeler cannot be initialized or parse output."""


@dataclass(frozen=True)
class GeminiBox:
    """Normalized (0..1) xyxy bounding box."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def clamp(self) -> "GeminiBox":
        xmin = max(0.0, min(1.0, self.xmin))
        ymin = max(0.0, min(1.0, self.ymin))
        xmax = max(0.0, min(1.0, self.xmax))
        ymax = max(0.0, min(1.0, self.ymax))
        return GeminiBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def valid(self) -> bool:
        return self.xmax > self.xmin and self.ymax > self.ymin

    def to_xyxy_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(round(self.xmin * width))
        y1 = int(round(self.ymin * height))
        x2 = int(round(self.xmax * width))
        y2 = int(round(self.ymax * height))
        return (x1, y1, x2, y2)

    @staticmethod
    def from_pixels(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> "GeminiBox":
        return GeminiBox(
            xmin=float(x1) / max(float(width), 1.0),
            ymin=float(y1) / max(float(height), 1.0),
            xmax=float(x2) / max(float(width), 1.0),
            ymax=float(y2) / max(float(height), 1.0),
        ).clamp()

    @staticmethod
    def from_any(raw: list[float], width: int, height: int) -> "GeminiBox | None":
        if len(raw) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in raw]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
            box = GeminiBox.from_pixels(x1, y1, x2, y2, width, height)
        else:
            box = GeminiBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2).clamp()
        if not box.valid():
            return None
        return box


@dataclass(frozen=True)
class GeminiBatchResult:
    """Parsed Gemini proposals and per-image parse/generation errors."""

    boxes_by_image: dict[str, list[GeminiBox]]
    errors_by_image: dict[str, str]


class GeminiLabeler:
    """Thin wrapper around Gemini image+JSON batch prompting."""

    def __init__(self, *, api_key: str, model_name: str) -> None:
        if genai is None:
            raise GeminiLabelerError("google-generativeai is not installed")
        if Image is None:
            raise GeminiLabelerError("Pillow is not installed")
        if not api_key.strip():
            raise GeminiLabelerError("Gemini API key is missing")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {"results": []}
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            return {"results": []}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match is None:
                return {"results": []}
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload
            return {"results": []}

    @staticmethod
    def _build_batch_prompt(image_names: list[str], object_description: str, max_boxes: int) -> str:
        lines = [
            "Task: detect product boxes for detector training.",
            f"Object target: {object_description}",
            "You will receive multiple images in order.",
            "Return JSON only using this schema:",
            "{\"results\": [{\"index\": 0, \"image\": \"name.jpg\", \"boxes\": [[xmin, ymin, xmax, ymax]]}]}",
            "Coordinates must be normalized 0..1.",
            f"Return up to {max_boxes} boxes per image.",
            "If object is absent, return boxes: [].",
            "Image index mapping:",
        ]
        for idx, name in enumerate(image_names):
            lines.append(f"- {idx}: {name}")
        return "\n".join(lines)

    def propose_batch(
        self,
        *,
        image_paths: list[Path],
        object_description: str,
        max_boxes: int,
    ) -> GeminiBatchResult:
        if not image_paths:
            return GeminiBatchResult(boxes_by_image={}, errors_by_image={})

        image_names = [path.name for path in image_paths]
        dims: dict[str, tuple[int, int]] = {}
        payload_images: list[Any] = []
        for path in image_paths:
            if Image is None:  # pragma: no cover - guarded in constructor
                raise GeminiLabelerError("Pillow is not installed")
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                dims[path.name] = rgb.size
                payload_images.append(rgb.copy())

        boxes_by_image: dict[str, list[GeminiBox]] = {name: [] for name in image_names}
        errors_by_image: dict[str, str] = {}
        try:
            response = self.model.generate_content(
                [self._build_batch_prompt(image_names, object_description, max_boxes), *payload_images]
            )
        except Exception as exc:  # noqa: BLE001
            message = f"gemini_request_failed: {exc}"
            for name in image_names:
                errors_by_image[name] = message
            return GeminiBatchResult(boxes_by_image=boxes_by_image, errors_by_image=errors_by_image)
        finally:
            for image in payload_images:
                image.close()

        raw_payload = self._extract_json(getattr(response, "text", "") or "")
        entries = raw_payload.get("results", [])
        if not isinstance(entries, list):
            message = "gemini_parse_failed: missing results list"
            for name in image_names:
                errors_by_image[name] = message
            return GeminiBatchResult(boxes_by_image=boxes_by_image, errors_by_image=errors_by_image)

        matched_names: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("image") or entry.get("filename") or "").strip()
            idx_raw = entry.get("index")
            if not name and isinstance(idx_raw, int) and 0 <= idx_raw < len(image_names):
                name = image_names[idx_raw]
            if name not in boxes_by_image:
                continue
            matched_names.add(name)
            raw_boxes = entry.get("boxes", [])
            if not isinstance(raw_boxes, list):
                errors_by_image[name] = "invalid_boxes_payload"
                continue
            width, height = dims[name]
            parsed: list[GeminiBox] = []
            for raw in raw_boxes[: max_boxes]:
                if not isinstance(raw, list):
                    continue
                box = GeminiBox.from_any(raw, width=width, height=height)
                if box is not None:
                    parsed.append(box)
            boxes_by_image[name] = parsed

        for name in image_names:
            if name not in matched_names:
                errors_by_image[name] = "missing_result"

        return GeminiBatchResult(boxes_by_image=boxes_by_image, errors_by_image=errors_by_image)
