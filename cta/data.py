from __future__ import annotations

import hashlib
import json
import random
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import COCO80


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_VAL_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    target_label: str
    target_class_id: int
    target_area: float
    labels: list[str]
    source_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_coco128(root: Path) -> None:
    image_dir = root / "images" / "train2017"
    if image_dir.exists() and len(list(image_dir.glob("*.jpg"))) >= 100:
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "coco128.zip"
    if not archive.exists():
        urllib.request.urlretrieve(COCO128_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root.parent)
    if not image_dir.exists():
        raise RuntimeError(f"COCO128 extraction did not create {image_dir}")


def load_coco128(root: str | Path, n: int, seed: int) -> list[Sample]:
    root = Path(root)
    download_coco128(root)
    image_dir = root / "images" / "train2017"
    label_dir = root / "labels" / "train2017"
    candidates: list[Sample] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        boxes = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if not (0 <= class_id < len(COCO80)):
                continue
            area = float(parts[3]) * float(parts[4])
            boxes.append((area, class_id))
        if not boxes:
            continue
        area, class_id = max(boxes)
        labels = sorted({COCO80[c] for _, c in boxes})
        candidates.append(
            Sample(
                sample_id=image_path.stem,
                image_path=str(image_path.resolve()),
                target_label=COCO80[class_id],
                target_class_id=class_id,
                target_area=area,
                labels=labels,
                source_sha256=sha256_file(image_path),
            )
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise ValueError(f"Requested {n} samples but only found {len(candidates)} labelled images")
    return candidates[:n]


def download_coco_val2017(root: Path) -> None:
    image_dir = root / "val2017"
    ann_path = root / "annotations" / "instances_val2017.json"
    if image_dir.exists() and ann_path.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    for url, name in ((COCO_VAL_URL, "val2017.zip"), (COCO_ANN_URL, "annotations_trainval2017.zip")):
        archive = root / name
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    if not image_dir.exists() or not ann_path.exists():
        raise RuntimeError("COCO val2017 download/extraction is incomplete")


def load_coco_val2017(root: str | Path, n: int, seed: int) -> list[Sample]:
    root = Path(root)
    download_coco_val2017(root)
    payload = json.loads((root / "annotations" / "instances_val2017.json").read_text())
    images = {int(i["id"]): i for i in payload["images"]}
    categories = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
    grouped: dict[int, list[dict]] = {}
    for ann in payload["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    ids = sorted(set(images) & set(grouped))
    rng = random.Random(seed)
    rng.shuffle(ids)
    candidates: list[Sample] = []
    for image_id in ids:
        info = images[image_id]
        anns = grouped[image_id]
        target = max(anns, key=lambda a: float(a.get("area", a["bbox"][2] * a["bbox"][3])))
        class_id = int(target["category_id"])
        image_path = root / "val2017" / info["file_name"]
        if not image_path.exists() or class_id not in categories:
            continue
        normalized_area = float(target.get("area", target["bbox"][2] * target["bbox"][3])) / (float(info["width"]) * float(info["height"]))
        labels = sorted({categories[int(a["category_id"])] for a in anns if int(a["category_id"]) in categories})
        candidates.append(Sample(
            sample_id=f"coco-{image_id:012d}", image_path=str(image_path.resolve()),
            target_label=categories[class_id], target_class_id=class_id,
            target_area=normalized_area, labels=labels, source_sha256=sha256_file(image_path),
        ))
        if len(candidates) >= n:
            break
    if len(candidates) < n:
        raise ValueError(f"Requested {n} COCO val2017 samples but found {len(candidates)}")
    return candidates


def load_dataset(name: str, root: str | Path, n: int, seed: int) -> list[Sample]:
    if name == "coco128":
        return load_coco128(root, n, seed)
    if name == "coco_val2017":
        return load_coco_val2017(root, n, seed)
    raise ValueError(f"Unsupported dataset: {name}")
