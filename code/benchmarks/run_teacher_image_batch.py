from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageScore:
    path: Path
    rel_path: str
    width: int
    height: int
    aspect_ratio: float
    brightness: float
    contrast: float
    blur_laplacian: float
    edge_density: float
    saliency_components: int
    medium_components: int
    largest_component_pct: float
    face_count: int
    person_count: int
    point_feedback_score: float
    text_failure_score: float
    matcher_score: float
    suitability_label: str
    notes: str


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def list_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_rgb(path: Path, max_side: int = 768) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    scale = min(1.0, float(max_side) / float(max(image.size)))
    if scale < 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def image_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def saliency_map(rgb: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "saliency"):
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, sal = saliency.computeSaliency(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if ok:
                return np.clip(sal.astype(np.float32), 0.0, 1.0)
        except Exception:
            pass
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 7)
    dog = np.abs(gray - blur)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    sal = 0.65 * dog + 0.35 * saturation
    sal = sal - float(sal.min())
    return sal / max(float(sal.max()), 1e-6)


def component_stats(rgb: np.ndarray) -> tuple[int, int, float]:
    sal = saliency_map(rgb)
    threshold = float(np.quantile(sal, 0.88))
    mask = (sal >= threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]
    total = float(mask.shape[0] * mask.shape[1])
    medium = [area for area in areas if 0.002 <= float(area) / total <= 0.20]
    largest = max(areas) if areas else 0
    return len(areas), len(medium), round(100.0 * float(largest) / total, 4)


def face_count(rgb: np.ndarray) -> int:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    return int(len(faces))


def person_count_hog(rgb: np.ndarray) -> int:
    image = rgb
    if max(image.shape[:2]) > 640:
        scale = 640.0 / float(max(image.shape[:2]))
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.08,
    )
    confident = [float(w) for w in weights.reshape(-1).tolist()] if len(weights) else []
    return int(sum(1 for value in confident if value > 0.2))


def score_image(path: Path, *, skip_hog: bool) -> ImageScore:
    rgb = load_rgb(path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = 100.0 * float(np.count_nonzero(edges)) / float(edges.size)
    components, medium, largest = component_stats(rgb)
    faces = face_count(rgb)
    persons = 0 if skip_hog else person_count_hog(rgb)
    entropy = image_entropy(gray)

    quality = min(1.0, blur / 180.0) * 0.35 + min(1.0, contrast / 70.0) * 0.35 + min(1.0, entropy / 7.5) * 0.30
    multi_object = min(1.0, medium / 8.0)
    clutter = min(1.0, edge_density / 14.0)
    not_dominated = 1.0 if largest < 35.0 else max(0.0, 1.0 - (largest - 35.0) / 40.0)
    point_score = 100.0 * (0.45 * multi_object + 0.25 * clutter + 0.20 * quality + 0.10 * not_dominated)
    text_score = 100.0 * (0.45 * min(1.0, (faces + persons) / 2.0) + 0.25 * clutter + 0.20 * multi_object + 0.10 * quality)
    matcher_score = 100.0 * (0.38 * multi_object + 0.24 * not_dominated + 0.20 * quality + 0.18 * min(1.0, components / 18.0))

    notes: list[str] = []
    if faces or persons:
        notes.append("person-like candidate")
    if medium >= 3:
        notes.append("multiple salient regions")
    if largest >= 45.0:
        notes.append("dominant foreground; may be less useful for multi-instance")
    if blur < 35.0:
        notes.append("possibly blurry")
    if point_score >= 70 and text_score >= 50:
        label = "strong_for_teacher_demo"
    elif point_score >= 60 or text_score >= 55 or matcher_score >= 65:
        label = "candidate"
    else:
        label = "weak_or_needs_manual_check"
    return ImageScore(
        path=path,
        rel_path=rel(path),
        width=w,
        height=h,
        aspect_ratio=round(float(w) / max(float(h), 1.0), 4),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        blur_laplacian=round(blur, 4),
        edge_density=round(edge_density, 4),
        saliency_components=components,
        medium_components=medium,
        largest_component_pct=largest,
        face_count=faces,
        person_count=persons,
        point_feedback_score=round(point_score, 4),
        text_failure_score=round(text_score, 4),
        matcher_score=round(matcher_score, 4),
        suitability_label=label,
        notes="; ".join(notes),
    )


def feature_for_pair(path: Path) -> np.ndarray:
    rgb = load_rgb(path, max_side=384)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    hist = hist.astype(np.float32)
    hist /= max(float(np.linalg.norm(hist)), 1e-8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.resize(cv2.Canny(gray, 80, 160), (32, 32)).astype(np.float32).reshape(-1)
    edges /= max(float(np.linalg.norm(edges)), 1e-8)
    return np.concatenate([hist, edges * 0.35]).astype(np.float32)


def build_pairs(scores: list[ImageScore], top_n: int = 80) -> list[dict[str, Any]]:
    candidates = sorted(scores, key=lambda item: item.matcher_score, reverse=True)[:top_n]
    features = [feature_for_pair(item.path) for item in candidates]
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(candidates):
        best_j = None
        best_sim = -1.0
        for j, right in enumerate(candidates):
            if i == j:
                continue
            sim = float(np.dot(features[i], features[j]))
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is None:
            continue
        right = candidates[best_j]
        pairs.append(
            {
                "support": left.rel_path,
                "query": right.rel_path,
                "support_score": left.matcher_score,
                "query_score": right.matcher_score,
                "visual_similarity": round(best_sim, 6),
                "pair_score": round((left.matcher_score + right.matcher_score) * 0.35 + 100.0 * best_sim * 0.30, 4),
            }
        )
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        key = tuple(sorted([str(pair["support"]), str(pair["query"])]))
        if key not in unique or float(pair["pair_score"]) > float(unique[key]["pair_score"]):
            unique[key] = pair
    return sorted(unique.values(), key=lambda item: float(item["pair_score"]), reverse=True)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3], fill=(255, 255, 255, 230))
    draw.text((x, y), text, fill=(20, 30, 42), font=font)


def tile_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#eef3f7")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def contact_sheet(items: list[ImageScore], path: Path, *, title: str, limit: int = 24) -> None:
    selected = items[:limit]
    cell_w, cell_h = 220, 190
    cols = 4
    rows = int(math.ceil(max(len(selected), 1) / cols))
    header = 52
    sheet = Image.new("RGB", (cols * cell_w, header + rows * cell_h), "#f6f8fb")
    draw = ImageDraw.Draw(sheet, "RGBA")
    draw_label(draw, (14, 14), title)
    for idx, item in enumerate(selected):
        x = (idx % cols) * cell_w
        y = header + (idx // cols) * cell_h
        sheet.paste(tile_image(item.path, (cell_w, cell_h - 36)), (x, y))
        name = item.path.name
        label = f"{idx+1}. {name} | P {item.point_feedback_score:.1f} T {item.text_failure_score:.1f}"
        draw_label(draw, (x + 6, y + cell_h - 32), label[:48])
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def pair_sheet(pairs: list[dict[str, Any]], path: Path, *, limit: int = 12) -> None:
    cell_w, cell_h = 260, 190
    cols = 2
    rows = int(math.ceil(max(min(len(pairs), limit), 1) / cols))
    header = 52
    sheet = Image.new("RGB", (cols * cell_w * 2, header + rows * cell_h), "#f6f8fb")
    draw = ImageDraw.Draw(sheet, "RGBA")
    draw_label(draw, (14, 14), "Potential reference-query pairs for Matcher Stage1")
    for idx, pair in enumerate(pairs[:limit]):
        base_x = (idx % cols) * cell_w * 2
        y = header + (idx // cols) * cell_h
        left = tile_image(ROOT / str(pair["support"]), (cell_w, cell_h - 36))
        right = tile_image(ROOT / str(pair["query"]), (cell_w, cell_h - 36))
        sheet.paste(left, (base_x, y))
        sheet.paste(right, (base_x + cell_w, y))
        label = f"{idx+1}. sim {float(pair['visual_similarity']):.3f}"
        draw_label(draw, (base_x + 8, y + cell_h - 32), label)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_to_row(item: ImageScore) -> dict[str, Any]:
    return {
        "image": item.rel_path,
        "width": item.width,
        "height": item.height,
        "aspect_ratio": item.aspect_ratio,
        "brightness": item.brightness,
        "contrast": item.contrast,
        "blur_laplacian": item.blur_laplacian,
        "edge_density": item.edge_density,
        "saliency_components": item.saliency_components,
        "medium_components": item.medium_components,
        "largest_component_pct": item.largest_component_pct,
        "face_count": item.face_count,
        "person_count": item.person_count,
        "point_feedback_score": item.point_feedback_score,
        "text_failure_score": item.text_failure_score,
        "matcher_score": item.matcher_score,
        "suitability_label": item.suitability_label,
        "notes": item.notes,
    }


def write_summary(out_dir: Path, scores: list[ImageScore], pairs: list[dict[str, Any]], *, image_dir: Path, skip_hog: bool) -> None:
    strong = [item for item in scores if item.suitability_label == "strong_for_teacher_demo"]
    candidates = [item for item in scores if item.suitability_label == "candidate"]
    person_like = [item for item in scores if item.face_count + item.person_count > 0]
    top_point = sorted(scores, key=lambda item: item.point_feedback_score, reverse=True)[:10]
    top_text = sorted(scores, key=lambda item: item.text_failure_score, reverse=True)[:10]
    top_matcher = sorted(scores, key=lambda item: item.matcher_score, reverse=True)[:10]
    lines = [
        "# 图片目录批量测试：是否能支撑老师要求",
        "",
        f"- Image dir: `{rel(image_dir)}`",
        f"- Tested images: `{len(scores)}`",
        f"- Strong candidates: `{len(strong)}`",
        f"- Secondary candidates: `{len(candidates)}`",
        f"- Person-like candidates by face/HOG: `{len(person_like)}`",
        f"- HOG person detector: `{'skipped' if skip_hog else 'enabled'}`",
        "",
        "## 当前结论",
        "",
        "这批图可以作为老师方向的候选图池，但不能单独作为最终证据。原因是：目录里只有原图，没有 COCO category annotation、实例 mask 或人工点位。因此本轮批量测试完成的是候选筛选和可视化预审；真正正面回应老师时，还需要对筛出的少量候选图跑 SAM3 text/point 或 Matcher reference-mask 实验。",
        "",
        "可支持的部分：",
        "",
        "- `point prompt 优于/更可控于 text`：有多显著区域、多实例或人物候选图，可以继续人工点 foreground/background 验证。",
        "- `white girl / execute text 失败`：若候选图中有人物，适合批量跑 `girl`、`white girl`、`execute` 对照；本轮只标出人像候选，不伪装成已经完成 SAM3 text 推理。",
        "- `Matcher Stage1 reference-query`：已输出潜在 reference-query pair，但由于没有 reference mask，只能作为后续人工标注或手动画 mask 的候选。",
        "",
        "不充分的部分：",
        "",
        "- 不能自动确认图中是否有 `girl`、`elephant` 或动作 `execute`，除非补 COCO annotations 或运行可靠检测/文本分割模型。",
        "- 不能评价 mask 质量 Dice/IoU，因为没有 ground-truth mask。",
        "- 不能直接证明 text 失败，因为 text 失败必须来自 SAM3 实际返回 instances=0 的结果。",
        "",
        "## Top point / feedback candidates",
        "",
        "| Rank | Image | Point score | Text score | Matcher score | Notes |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, item in enumerate(top_point, start=1):
        lines.append(f"| {idx} | `{item.rel_path}` | {item.point_feedback_score:.2f} | {item.text_failure_score:.2f} | {item.matcher_score:.2f} | {item.notes} |")
    lines.extend(["", "## Top text-failure candidates", "", "| Rank | Image | Text score | Face | Person | Notes |", "| ---: | --- | ---: | ---: | ---: | --- |"])
    for idx, item in enumerate(top_text, start=1):
        lines.append(f"| {idx} | `{item.rel_path}` | {item.text_failure_score:.2f} | {item.face_count} | {item.person_count} | {item.notes} |")
    lines.extend(["", "## Top Matcher candidates", "", "| Rank | Image | Matcher score | Components | Largest % | Notes |", "| ---: | --- | ---: | ---: | ---: | --- |"])
    for idx, item in enumerate(top_matcher, start=1):
        lines.append(f"| {idx} | `{item.rel_path}` | {item.matcher_score:.2f} | {item.medium_components} | {item.largest_component_pct:.2f} | {item.notes} |")
    lines.extend(["", "## Potential reference-query pairs", "", "| Rank | Support | Query | Similarity | Pair score |", "| ---: | --- | --- | ---: | ---: |"])
    for idx, pair in enumerate(pairs[:10], start=1):
        lines.append(f"| {idx} | `{pair['support']}` | `{pair['query']}` | {float(pair['visual_similarity']):.4f} | {float(pair['pair_score']):.2f} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `image_batch_scores.csv`: 每张图的图像质量、显著区域、人脸/HOG 人体、三类 suitability score。",
            "- `candidate_pairs.csv`: 用颜色直方图和边缘结构近似筛出的 reference-query 候选对。",
            "- `top_point_feedback_candidates.jpg`: 适合点/正负反馈演示的候选图 contact sheet。",
            "- `top_text_failure_candidates.jpg`: 适合跑 `girl / white girl / execute` 的候选图 contact sheet。",
            "- `top_matcher_pairs.jpg`: 潜在 Matcher reference-query pair contact sheet。",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-screen local images for teacher-direction SAM3/Matcher evidence.")
    parser.add_argument("--image-dir", default=str(ROOT / "图片"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "teacher_image_batch"))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--skip-hog", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    images = list_images(image_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise ValueError(f"No images found under {image_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: list[ImageScore] = []
    failures: list[dict[str, str]] = []
    for index, path in enumerate(images, start=1):
        try:
            scores.append(score_image(path, skip_hog=bool(args.skip_hog)))
        except Exception as exc:
            failures.append({"image": rel(path), "error": repr(exc)})
        if index % 50 == 0:
            print(f"processed {index}/{len(images)}", flush=True)

    pairs = build_pairs(scores)
    rows = [score_to_row(item) for item in scores]
    fields = list(rows[0].keys()) if rows else ["image"]
    write_csv(out_dir / "image_batch_scores.csv", rows, fields)
    write_csv(
        out_dir / "candidate_pairs.csv",
        pairs,
        ["support", "query", "support_score", "query_score", "visual_similarity", "pair_score"],
    )
    if failures:
        write_csv(out_dir / "failures.csv", failures, ["image", "error"])
    (out_dir / "image_batch_scores.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    contact_sheet(sorted(scores, key=lambda item: item.point_feedback_score, reverse=True), out_dir / "top_point_feedback_candidates.jpg", title="Top point / feedback candidates")
    contact_sheet(sorted(scores, key=lambda item: item.text_failure_score, reverse=True), out_dir / "top_text_failure_candidates.jpg", title="Top text-failure candidates")
    pair_sheet(pairs, out_dir / "top_matcher_pairs.jpg")
    write_summary(out_dir, scores, pairs, image_dir=image_dir, skip_hog=bool(args.skip_hog))
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
