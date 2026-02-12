"""Hybrid auto-labeling test script.

Detector-first proposals (fast) with optional Gemini fallback (slow, targeted).
Writes YOLO labels + preview overlays + a manifest for analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional runtime dependency
    genai = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None

from prescience.profiles.io import load_profile
from prescience.vision.embeddings import build_embedder


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "test_images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "labeled_hybrid"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def clamp(self) -> "Box":
        xmin = max(0.0, min(1.0, self.xmin))
        ymin = max(0.0, min(1.0, self.ymin))
        xmax = max(0.0, min(1.0, self.xmax))
        ymax = max(0.0, min(1.0, self.ymax))
        return Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def valid(self) -> bool:
        return self.xmax > self.xmin and self.ymax > self.ymin

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1e-9)

    def to_xyxy_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(round(self.xmin * width))
        y1 = int(round(self.ymin * height))
        x2 = int(round(self.xmax * width))
        y2 = int(round(self.ymax * height))
        return (x1, y1, x2, y2)

    def to_yolo_line(self, class_id: int = 0) -> str:
        xc = (self.xmin + self.xmax) * 0.5
        yc = (self.ymin + self.ymax) * 0.5
        w = self.width
        h = self.height
        return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def from_pixels(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> "Box":
        return Box(
            xmin=float(x1) / max(float(width), 1.0),
            ymin=float(y1) / max(float(height), 1.0),
            xmax=float(x2) / max(float(width), 1.0),
            ymax=float(y2) / max(float(height), 1.0),
        ).clamp()

    @staticmethod
    def from_any(raw: list[float], width: int, height: int) -> "Box | None":
        if len(raw) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in raw]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
            box = Box.from_pixels(x1, y1, x2, y2, width, height)
        else:
            box = Box(xmin=x1, ymin=y1, xmax=x2, ymax=y2).clamp()
        if not box.valid():
            return None
        return box


class DetectorProposer:
    def __init__(
        self,
        model_path: str,
        conf: float,
        imgsz: int,
        max_det: int,
        classes: list[int] | None = None,
    ) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed; install it to use detector proposals.")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.max_det = max_det
        self.classes = classes

    def propose(self, image_path: Path) -> tuple[list[Box], list[float]]:
        with Image.open(image_path) as image:
            width, height = image.size
        result = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return [], []
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        pairs: list[tuple[Box, float]] = []
        for coords, conf in zip(xyxy, confs):
            box = Box.from_pixels(
                x1=float(coords[0]),
                y1=float(coords[1]),
                x2=float(coords[2]),
                y2=float(coords[3]),
                width=width,
                height=height,
            )
            if box.valid():
                pairs.append((box, float(conf)))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in pairs], [item[1] for item in pairs]


class GeminiProposer:
    def __init__(self, api_key: str, model_name: str, object_name: str, max_boxes: int) -> None:
        if genai is None:
            raise RuntimeError("google-generativeai is not installed; install it to use Gemini fallback.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.object_name = object_name
        self.max_boxes = max_boxes

    def _prompt(self) -> str:
        return (
            "Task: detect product boxes for detector training.\n"
            f"Object target: {self.object_name}\n"
            f"Return up to {self.max_boxes} boxes.\n"
            "Return JSON only: {\"boxes\": [[xmin, ymin, xmax, ymax], ...]}\n"
            "Coordinates must be normalized 0..1.\n"
            "If absent: {\"boxes\": []}\n"
        )

    def _batch_prompt(self, image_names: list[str]) -> str:
        lines = [
            "Task: detect product boxes for detector training.",
            f"Object target: {self.object_name}",
            "You will receive multiple images in order.",
            "Return JSON only using this schema:",
            "{\"results\": [{\"index\": 0, \"image\": \"name.jpg\", \"boxes\": [[xmin, ymin, xmax, ymax]]}]}",
            "Coordinates must be normalized 0..1.",
            f"Return up to {self.max_boxes} boxes per image.",
            "If object not present, return boxes: [].",
            "Image index mapping:",
        ]
        for idx, name in enumerate(image_names):
            lines.append(f"- {idx}: {name}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {"boxes": []}
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
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match is None:
                return {"boxes": []}
            return json.loads(match.group(0))

    def propose(self, image_path: Path) -> list[Box]:
        with Image.open(image_path) as image:
            width, height = image.size
            response = self.model.generate_content([self._prompt(), image])
        payload = self._extract_json(getattr(response, "text", "") or "")
        raw_boxes = payload.get("boxes", [])
        out: list[Box] = []
        if not isinstance(raw_boxes, list):
            return out
        for raw in raw_boxes[: self.max_boxes]:
            if not isinstance(raw, list):
                continue
            box = Box.from_any(raw, width=width, height=height)
            if box is not None:
                out.append(box)
        return out

    def propose_batch(self, image_paths: list[Path]) -> dict[str, list[Box]]:
        if not image_paths:
            return {}

        image_names = [path.name for path in image_paths]
        dims: dict[str, tuple[int, int]] = {}
        payload_images: list[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                dims[path.name] = rgb.size
                payload_images.append(rgb.copy())
        try:
            response = self.model.generate_content([self._batch_prompt(image_names), *payload_images])
        finally:
            for img in payload_images:
                img.close()

        raw_payload = self._extract_json(getattr(response, "text", "") or "")
        out: dict[str, list[Box]] = {name: [] for name in image_names}
        entries = raw_payload.get("results", [])
        if not isinstance(entries, list):
            return out

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("image") or entry.get("filename") or "").strip()
            idx_raw = entry.get("index")
            if not name and isinstance(idx_raw, int) and 0 <= idx_raw < len(image_names):
                name = image_names[idx_raw]
            if name not in out:
                continue
            raw_boxes = entry.get("boxes", [])
            if not isinstance(raw_boxes, list):
                continue
            width, height = dims[name]
            boxes: list[Box] = []
            for raw in raw_boxes[: self.max_boxes]:
                if not isinstance(raw, list):
                    continue
                box = Box.from_any(raw, width=width, height=height)
                if box is not None:
                    boxes.append(box)
            out[name] = boxes
        return out


def filter_boxes(
    boxes: list[Box],
    *,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
) -> list[Box]:
    filtered: list[Box] = []
    for box in boxes:
        if not box.valid():
            continue
        if box.area < min_area or box.area > max_area:
            continue
        if box.aspect < min_aspect or box.aspect > max_aspect:
            continue
        filtered.append(box)
    return filtered


def write_yolo_labels(path: Path, boxes: list[Box], class_id: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not boxes:
        path.write_text("", encoding="utf-8")
        return
    lines = [box.to_yolo_line(class_id=class_id) for box in boxes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_preview(image_path: Path, boxes: list[Box], out_path: Path, source: str) -> None:
    with Image.open(image_path) as image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        if source in {"review_required", "none"}:
            color = "#ff5e5e"
        elif source.startswith("detector"):
            color = "#00d084"
        else:
            color = "#ff9f1c"
        for box in boxes:
            draw.rectangle(box.to_xyxy_pixels(width, height), outline=color, width=3)
        draw.text((8, 8), f"source={source} boxes={len(boxes)}", fill=color)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)


def list_images(image_dir: Path, limit: int) -> list[Path]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    return images if limit <= 0 else images[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid auto-label test: detector first, Gemini fallback.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--object-name", type=str, default="Chick-fil-A sauce bottle")
    parser.add_argument("--limit", type=int, default=40)

    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Detector model for proposals. Use 'yolov8n.pt' (default), explicit path, or 'auto_latest'.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.03)
    parser.add_argument("--yolo-imgsz", type=int, default=768)
    parser.add_argument("--yolo-max-det", type=int, default=6)
    parser.add_argument(
        "--yolo-class-ids",
        type=str,
        default="",
        help="Optional comma-separated detector class ids to limit proposals (example: 39 for bottle).",
    )
    parser.add_argument("--detector-accept-conf", type=float, default=0.20)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=1,
        help="Expected number of target objects per frame for safe auto-labeling.",
    )
    parser.add_argument(
        "--min-top1-margin",
        type=float,
        default=0.12,
        help="Required (top1 - top2) detector confidence margin to auto-accept when many detections exist.",
    )
    parser.add_argument(
        "--allow-uncertain-write",
        action="store_true",
        help="If set, uncertain detector frames still write tentative labels. Default is review-only (safe).",
    )
    parser.add_argument(
        "--target-sku",
        type=str,
        default="",
        help="Target SKU id for embedding verification filter (example: chickfila_sauce).",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path("data/profiles"),
        help="Root directory containing SKU profiles.",
    )
    parser.add_argument(
        "--use-profile-filter",
        action="store_true",
        help="Enable SKU-specific embedding filter on detector proposals.",
    )
    parser.add_argument(
        "--profile-threshold",
        type=float,
        default=-1.0,
        help="Override profile similarity threshold. Use <0 to use threshold from profile metadata.",
    )
    parser.add_argument(
        "--embedding-backbone",
        type=str,
        default="resnet18",
        help="Embedder backbone for proposal verification.",
    )

    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--gemini-model", type=str, default="gemini-3-pro-preview")
    parser.add_argument("--gemini-api-key-env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--max-boxes", type=int, default=6)
    parser.add_argument(
        "--vlm-only",
        action="store_true",
        help="Skip detector and label directly with Gemini.",
    )
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=1,
        help="Number of images per Gemini API call in --vlm-only mode (e.g. 24).",
    )

    parser.add_argument("--min-area", type=float, default=0.002)
    parser.add_argument("--max-area", type=float, default=0.85)
    parser.add_argument("--min-aspect", type=float, default=0.20)
    parser.add_argument("--max-aspect", type=float, default=5.0)
    return parser.parse_args()


def parse_class_ids(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    return out or None


def resolve_yolo_model_path(raw_value: str) -> str:
    if raw_value != "auto_latest":
        # For named Ultralytics models (e.g. yolov8n.pt), pass through directly.
        if raw_value.startswith("yolo") and raw_value.endswith(".pt"):
            return raw_value
        path = Path(raw_value)
        if not path.exists():
            raise RuntimeError(f"YOLO model path does not exist: {path}")
        return str(path)

    candidates = sorted(Path("data/models/yolo").glob("*_v*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(
            "Could not resolve local latest model automatically; pass --yolo-model <path_to_best.pt> "
            "or use default --yolo-model yolov8n.pt."
        )
    return str(candidates[0])


def load_profile_filter(
    *,
    enabled: bool,
    target_sku: str,
    profile_root: Path,
    threshold_override: float,
    backbone: str,
) -> tuple[Any | None, np.ndarray | None, float | None]:
    if not enabled:
        return None, None, None
    sku = target_sku.strip()
    if not sku:
        print("[warn] --use-profile-filter requested but --target-sku is empty; disabling profile filter.")
        return None, None, None
    profile_dir = profile_root / sku
    if not profile_dir.exists():
        print(f"[warn] Profile dir not found for SKU '{sku}' at {profile_dir}; disabling profile filter.")
        return None, None, None

    try:
        profile, embeddings = load_profile(profile_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not load profile for '{sku}' ({exc}); disabling profile filter.")
        return None, None, None
    if embeddings.size == 0:
        print(f"[warn] Profile embeddings are empty for '{sku}'; disabling profile filter.")
        return None, None, None
    threshold = float(profile.metadata.threshold) if threshold_override < 0 else float(threshold_override)
    embedder = build_embedder(backbone)
    return embedder, embeddings.astype(np.float32), threshold


def filter_by_profile_similarity(
    *,
    image_path: Path,
    boxes: list[Box],
    detector_scores: list[float],
    embedder: Any,
    profile_embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[Box], list[float], list[float]]:
    if not boxes:
        return [], [], []

    image = cv2.imread(str(image_path))
    if image is None:
        return [], [], []
    h, w = image.shape[:2]

    kept_boxes: list[Box] = []
    kept_det_scores: list[float] = []
    sim_scores: list[float] = []

    for box, det_score in zip(boxes, detector_scores):
        x1, y1, x2, y2 = box.to_xyxy_pixels(width=w, height=h)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        emb = embedder.encode(crop)
        score = float(np.max(profile_embeddings @ emb))
        if score >= threshold:
            kept_boxes.append(box)
            kept_det_scores.append(float(det_score))
            sim_scores.append(score)

    return kept_boxes, kept_det_scores, sim_scores


def choose_detector_result(
    *,
    boxes: list[Box],
    scores: list[float],
    expected_count: int,
    accept_conf: float,
    min_top1_margin: float,
) -> tuple[list[Box], bool, str]:
    """Return selected boxes, whether review is needed, and reason."""
    if not boxes:
        return [], True, "no_detector_boxes"
    top1 = float(scores[0]) if scores else 0.0
    top2 = float(scores[1]) if len(scores) > 1 else -1.0
    margin = top1 - top2
    too_many = expected_count > 0 and len(boxes) > expected_count
    ambiguous = too_many and margin < min_top1_margin

    k = expected_count if expected_count > 0 else len(boxes)
    selected = boxes[:k]
    if top1 >= accept_conf and not ambiguous:
        return selected, False, "detector_high_conf"
    if ambiguous:
        return selected, True, f"ambiguous_multi(top1={top1:.3f},margin={margin:.3f})"
    return selected, True, f"low_conf(top1={top1:.3f})"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(args.image_dir, args.limit)
    if not images:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    yolo_model = resolve_yolo_model_path(args.yolo_model)
    class_ids = parse_class_ids(args.yolo_class_ids)

    detector = DetectorProposer(
        model_path=yolo_model,
        conf=args.yolo_conf,
        imgsz=args.yolo_imgsz,
        max_det=args.yolo_max_det,
        classes=class_ids,
    )
    embedder, profile_embeddings, profile_threshold = load_profile_filter(
        enabled=bool(args.use_profile_filter),
        target_sku=args.target_sku,
        profile_root=args.profile_root,
        threshold_override=args.profile_threshold,
        backbone=args.embedding_backbone,
    )

    vlm: GeminiProposer | None = None
    if args.enable_vlm:
        api_key = os.getenv(args.gemini_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"--enable-vlm was set but env var {args.gemini_api_key_env} is empty. "
                "Set it before running."
            )
        vlm = GeminiProposer(
            api_key=api_key,
            model_name=args.gemini_model,
            object_name=args.object_name,
            max_boxes=args.max_boxes,
        )
    if args.vlm_only and vlm is None:
        raise RuntimeError("--vlm-only requires --enable-vlm and a valid Gemini API key environment variable.")
    if args.vlm_batch_size < 1:
        raise RuntimeError("--vlm-batch-size must be >= 1.")

    labels_dir = args.output_dir / "labels"
    previews_dir = args.output_dir / "previews"
    review_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []

    detector_only = 0
    vlm_fallback = 0
    review_required = 0
    detector_uncertain_written = 0
    none_count = 0
    profile_rejected = 0
    vlm_batch_labeled = 0

    print(f"Labeling {len(images)} images from {args.image_dir}")
    if args.vlm_only and vlm is not None:
        for start in range(0, len(images), args.vlm_batch_size):
            chunk = images[start : start + args.vlm_batch_size]
            print(f"- VLM batch {start + 1}-{start + len(chunk)}")
            if len(chunk) == 1:
                batch_boxes = {chunk[0].name: vlm.propose(chunk[0])}
            else:
                batch_boxes = vlm.propose_batch(chunk)

            for image_path in chunk:
                source = "vlm_batch"
                review_reason = ""
                needs_review = False
                filtered = filter_boxes(
                    batch_boxes.get(image_path.name, []),
                    min_area=args.min_area,
                    max_area=args.max_area,
                    min_aspect=args.min_aspect,
                    max_aspect=args.max_aspect,
                )
                k = args.expected_count if args.expected_count > 0 else args.max_boxes
                final_boxes = filtered[:k]
                if not final_boxes:
                    source = "review_required"
                    review_reason = "no_vlm_boxes"
                    needs_review = True
                    review_required += 1
                    none_count += 1
                else:
                    vlm_batch_labeled += 1

                label_path = labels_dir / f"{image_path.stem}.txt"
                preview_path = previews_dir / image_path.name
                write_yolo_labels(label_path, final_boxes, class_id=0)
                draw_preview(image_path, final_boxes, preview_path, source=source)

                manifest_records.append(
                    {
                        "image": image_path.name,
                        "source": source,
                        "detector_max_conf": None,
                        "profile_max_similarity": None,
                        "num_boxes": len(final_boxes),
                        "needs_review": needs_review,
                        "review_reason": review_reason,
                        "label_path": str(label_path),
                        "preview_path": str(preview_path),
                    }
                )
                if needs_review:
                    review_records.append(
                        {
                            "image": image_path.name,
                            "reason": review_reason,
                            "suggested_boxes": [],
                            "preview_path": str(preview_path),
                        }
                    )

        manifest = {
            "image_dir": str(args.image_dir),
            "output_dir": str(args.output_dir),
            "detector_model": yolo_model,
            "detector_conf": args.yolo_conf,
            "detector_classes": class_ids,
            "detector_accept_conf": args.detector_accept_conf,
            "use_profile_filter": bool(embedder is not None),
            "target_sku": args.target_sku,
            "profile_root": str(args.profile_root),
            "profile_threshold": profile_threshold,
            "embedding_backbone": args.embedding_backbone,
            "expected_count": args.expected_count,
            "min_top1_margin": args.min_top1_margin,
            "allow_uncertain_write": bool(args.allow_uncertain_write),
            "enable_vlm": True,
            "vlm_only": True,
            "vlm_batch_size": args.vlm_batch_size,
            "stats": {
                "total_images": len(images),
                "vlm_batch_labeled": vlm_batch_labeled,
                "review_required": review_required,
                "none": none_count,
            },
            "records": manifest_records,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (args.output_dir / "needs_review.json").write_text(json.dumps(review_records, indent=2), encoding="utf-8")
        print(
            "Done: "
            f"vlm_batch_labeled={vlm_batch_labeled} "
            f"review_required={review_required} "
            f"none={none_count}"
        )
        print(
            "Outputs: "
            f"labels={labels_dir} "
            f"previews={previews_dir} "
            f"manifest={args.output_dir / 'manifest.json'} "
            f"needs_review={args.output_dir / 'needs_review.json'}"
        )
        return

    for image_path in images:
        print(f"- {image_path.name}")
        det_boxes, det_scores = detector.propose(image_path)
        det_pairs = [
            (box, score)
            for box, score in zip(det_boxes, det_scores)
            if box.valid()
            and args.min_area <= box.area <= args.max_area
            and args.min_aspect <= box.aspect <= args.max_aspect
        ]
        det_boxes = [item[0] for item in det_pairs]
        det_scores = [float(item[1]) for item in det_pairs]
        det_max = max(det_scores) if det_scores else 0.0
        profile_max = None

        if embedder is not None and profile_embeddings is not None and profile_threshold is not None:
            before_count = len(det_boxes)
            det_boxes, det_scores, sim_scores = filter_by_profile_similarity(
                image_path=image_path,
                boxes=det_boxes,
                detector_scores=det_scores,
                embedder=embedder,
                profile_embeddings=profile_embeddings,
                threshold=profile_threshold,
            )
            profile_max = max(sim_scores) if sim_scores else 0.0
            if before_count > 0 and len(det_boxes) == 0:
                profile_rejected += 1

        source = "none"
        review_reason = ""
        needs_review = False
        final_boxes: list[Box] = []
        selected_det, det_review, det_reason = choose_detector_result(
            boxes=det_boxes[: args.max_boxes],
            scores=det_scores[: args.max_boxes],
            expected_count=args.expected_count,
            accept_conf=args.detector_accept_conf,
            min_top1_margin=args.min_top1_margin,
        )

        if not det_review:
            source = "detector_high_conf"
            final_boxes = selected_det
            detector_only += 1
        else:
            review_reason = det_reason
            needs_review = True
            if vlm is not None:
                vlm_boxes = filter_boxes(
                    vlm.propose(image_path),
                    min_area=args.min_area,
                    max_area=args.max_area,
                    min_aspect=args.min_aspect,
                    max_aspect=args.max_aspect,
                )
                if vlm_boxes:
                    k = args.expected_count if args.expected_count > 0 else args.max_boxes
                    source = "vlm_fallback"
                    final_boxes = vlm_boxes[:k]
                    needs_review = False
                    review_reason = ""
                    vlm_fallback += 1
                elif args.allow_uncertain_write and selected_det:
                    source = "detector_uncertain_written"
                    final_boxes = selected_det
                    detector_uncertain_written += 1
                else:
                    source = "review_required"
                    if embedder is not None and det_reason == "no_detector_boxes":
                        review_reason = (
                            "profile_filter_rejected_all" if profile_max is not None else "no_detector_boxes"
                        )
                    review_required += 1
                    if not selected_det:
                        none_count += 1
            else:
                if args.allow_uncertain_write and selected_det:
                    source = "detector_uncertain_written"
                    final_boxes = selected_det
                    needs_review = False
                    detector_uncertain_written += 1
                else:
                    source = "review_required"
                    review_required += 1
                    if not selected_det:
                        none_count += 1

        label_path = labels_dir / f"{image_path.stem}.txt"
        preview_path = previews_dir / image_path.name
        write_yolo_labels(label_path, final_boxes, class_id=0)
        draw_preview(image_path, final_boxes, preview_path, source=source)

        manifest_records.append(
            {
                "image": image_path.name,
                "source": source,
                "detector_max_conf": det_max,
                "profile_max_similarity": profile_max,
                "num_boxes": len(final_boxes),
                "needs_review": needs_review,
                "review_reason": review_reason,
                "label_path": str(label_path),
                "preview_path": str(preview_path),
            }
        )
        if needs_review:
            review_records.append(
                {
                    "image": image_path.name,
                    "reason": review_reason or "uncertain",
                    "suggested_boxes": [box.to_yolo_line(class_id=0) for box in selected_det],
                    "preview_path": str(preview_path),
                }
            )

    manifest = {
        "image_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "detector_model": yolo_model,
        "detector_conf": args.yolo_conf,
        "detector_classes": class_ids,
        "detector_accept_conf": args.detector_accept_conf,
        "use_profile_filter": bool(embedder is not None),
        "target_sku": args.target_sku,
        "profile_root": str(args.profile_root),
        "profile_threshold": profile_threshold,
        "embedding_backbone": args.embedding_backbone,
        "expected_count": args.expected_count,
        "min_top1_margin": args.min_top1_margin,
        "allow_uncertain_write": bool(args.allow_uncertain_write),
        "enable_vlm": bool(vlm is not None),
        "stats": {
            "total_images": len(images),
            "detector_high_conf": detector_only,
            "detector_uncertain_written": detector_uncertain_written,
            "review_required": review_required,
            "vlm_fallback": vlm_fallback,
            "none": none_count,
            "profile_rejected_all_boxes": profile_rejected,
        },
        "records": manifest_records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "needs_review.json").write_text(json.dumps(review_records, indent=2), encoding="utf-8")
    print(
        "Done: "
        f"detector_high_conf={detector_only} "
        f"detector_uncertain_written={detector_uncertain_written} "
        f"review_required={review_required} "
        f"vlm_fallback={vlm_fallback} "
        f"none={none_count}"
    )
    print(
        "Outputs: "
        f"labels={labels_dir} "
        f"previews={previews_dir} "
        f"manifest={args.output_dir / 'manifest.json'} "
        f"needs_review={args.output_dir / 'needs_review.json'}"
    )


if __name__ == "__main__":
    main()
