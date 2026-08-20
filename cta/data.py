from __future__ import annotations

import hashlib
import io
import json
import random
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import COCO80


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_VAL_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VOC2007_TEST_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
VOC_LABEL_MAP = {
    "aeroplane": "airplane",
    "diningtable": "dining table",
    "motorbike": "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
}
VOC20 = [
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorcycle", "person", "potted plant", "sheep", "couch",
    "train", "tv",
]
VOC_COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
]


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


def _apply_sample_filters(
    candidates: list[Sample],
    n: int,
    include_sample_ids: set[str] | None,
    exclude_sample_ids: set[str] | None,
) -> list[Sample]:
    if include_sample_ids:
        candidates = [s for s in candidates if s.sample_id in include_sample_ids]
    if exclude_sample_ids:
        candidates = [s for s in candidates if s.sample_id not in exclude_sample_ids]
    if len(candidates) < n:
        raise ValueError(f"Requested {n} samples but only found {len(candidates)} after filtering")
    return candidates[:n]


def _split_ids(all_ids: list[int], dataset_split: str | None) -> list[int]:
    if dataset_split == "primary":
        return all_ids[::2]
    if dataset_split == "secondary":
        return all_ids[1::2]
    return all_ids


def _unique_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


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


def load_coco128(
    root: str | Path,
    n: int,
    seed: int,
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
) -> list[Sample]:
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
    return _apply_sample_filters(candidates, n, include_sample_ids, exclude_sample_ids)


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


def load_coco_val2017(
    root: str | Path,
    n: int,
    seed: int,
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
    dataset_split: str | None = None,
) -> list[Sample]:
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
    ids = _split_ids(ids, dataset_split)
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
    return _apply_sample_filters(candidates, n, include_sample_ids, exclude_sample_ids)


def download_coco_val2017_hf(root: Path) -> None:
    shards = sorted((root / "data").glob("validation-*.parquet"))
    if len(shards) >= 2:
        return
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="BrandonLSX/coco-2017", repo_type="dataset",
        allow_patterns=["data/validation-*.parquet"], local_dir=str(root),
    )


def load_coco_val2017_hf(
    root: str | Path,
    n: int,
    seed: int,
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
    dataset_split: str | None = None,
) -> list[Sample]:
    import pyarrow.parquet as pq

    root = Path(root)
    download_coco_val2017_hf(root)
    shards = sorted((root / "data").glob("validation-*.parquet"))
    all_ids: list[int] = []
    for shard in shards:
        all_ids.extend(int(x) for x in pq.read_table(shard, columns=["image_id"])["image_id"].to_pylist())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    all_ids = _unique_preserve_order(all_ids)
    split_ids = _split_ids(all_ids, dataset_split)
    candidate_ids = split_ids[: min(len(split_ids), n + 100)]
    if len(candidate_ids) < n:
        raise ValueError(f"Requested {n} samples but mirror exposes {len(candidate_ids)}")
    selected = set(candidate_ids)
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    found: dict[int, Sample] = {}
    columns = ["image", "image_id", "file_name", "width", "height", "annotations"]
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                image_id = int(row["image_id"])
                if image_id not in selected:
                    continue
                ann = row["annotations"]
                valid = [i for i, crowd in enumerate(ann["iscrowd"]) if int(crowd) == 0]
                if not valid:
                    continue
                best = max(valid, key=lambda i: float(ann["area"][i]))
                image_path = extracted / row["file_name"]
                image_bytes = row["image"]["bytes"]
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)
                labels = sorted({str(ann["category_name"][i]) for i in valid})
                area = float(ann["area"][best]) / (float(row["width"]) * float(row["height"]))
                found[image_id] = Sample(
                    sample_id=f"coco-{image_id:012d}", image_path=str(image_path.resolve()),
                    target_label=str(ann["category_name"][best]), target_class_id=int(ann["category_id"][best]),
                    target_area=area, labels=labels, source_sha256=hashlib.sha256(image_bytes).hexdigest(),
                )
    ordered = [found[i] for i in candidate_ids if i in found]
    return _apply_sample_filters(ordered, n, include_sample_ids, exclude_sample_ids)


def download_voc2007_test(root: Path) -> None:
    voc_root = root / "VOCdevkit" / "VOC2007"
    if (voc_root / "JPEGImages").exists() and (voc_root / "Annotations").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "VOCtest_06-Nov-2007.tar"
    if not archive.exists():
        urllib.request.urlretrieve(VOC2007_TEST_URL, archive)
    resolved_root = root.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (root / member.name).resolve()
            if resolved_root != target and resolved_root not in target.parents:
                raise RuntimeError(f"Refusing unsafe archive member: {member.name}")
        handle.extractall(root)
    if not (voc_root / "JPEGImages").exists() or not (voc_root / "Annotations").exists():
        raise RuntimeError("Pascal VOC 2007 test extraction is incomplete")


def load_voc2007_test(
    root: str | Path,
    n: int,
    seed: int,
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
    dataset_split: str | None = None,
) -> list[Sample]:
    root = Path(root)
    download_voc2007_test(root)
    voc_root = root / "VOCdevkit" / "VOC2007"
    test_ids = [line.strip() for line in (voc_root / "ImageSets" / "Main" / "test.txt").read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(test_ids)
    if dataset_split == "primary":
        test_ids = test_ids[::2]
    elif dataset_split == "secondary":
        test_ids = test_ids[1::2]
    candidates: list[Sample] = []
    for voc_id in test_ids:
        image_path = voc_root / "JPEGImages" / f"{voc_id}.jpg"
        annotation_path = voc_root / "Annotations" / f"{voc_id}.xml"
        if not image_path.exists() or not annotation_path.exists():
            continue
        root_xml = ET.parse(annotation_path).getroot()
        size = root_xml.find("size")
        width = float(size.findtext("width", "1")) if size is not None else 1.0
        height = float(size.findtext("height", "1")) if size is not None else 1.0
        objects: list[tuple[float, str]] = []
        for obj in root_xml.findall("object"):
            raw_label = obj.findtext("name", "").strip().lower()
            box = obj.find("bndbox")
            if not raw_label or box is None:
                continue
            xmin = float(box.findtext("xmin", "0")); ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0")); ymax = float(box.findtext("ymax", "0"))
            area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin) / max(1.0, width * height)
            objects.append((area, VOC_LABEL_MAP.get(raw_label, raw_label)))
        if not objects:
            continue
        target_area, target_label = max(objects)
        labels = sorted({label for _, label in objects})
        candidates.append(Sample(
            sample_id=f"voc2007-{voc_id}", image_path=str(image_path.resolve()),
            target_label=target_label, target_class_id=-1, target_area=target_area,
            labels=labels, source_sha256=sha256_file(image_path),
        ))
    return _apply_sample_filters(candidates, n, include_sample_ids, exclude_sample_ids)


def download_voc2012_segmentation_hf(root: Path) -> None:
    if list((root / "data").glob("*.parquet")):
        return
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="nateraw/pascal-voc-2012", repo_type="dataset",
        allow_patterns=["data/*.parquet"], local_dir=str(root),
    )


def _embedded_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value["bytes"]
    raise ValueError("unsupported embedded image value")


def load_voc2012_segmentation_hf(
    root: str | Path,
    n: int,
    seed: int,
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
    dataset_split: str | None = None,
) -> list[Sample]:
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    root = Path(root)
    download_voc2012_segmentation_hf(root)
    shards = sorted((root / "data").glob("*.parquet"))
    rows: list[dict] = []
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        columns = parquet.schema_arrow.names
        image_column = "image" if "image" in columns else "pixel_values"
        mask_column = "mask" if "mask" in columns else "label"
        for batch in parquet.iter_batches(batch_size=32, columns=[image_column, mask_column]):
            rows.extend(batch.to_pylist())
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if dataset_split == "primary":
        indices = indices[::2]
    elif dataset_split == "secondary":
        indices = indices[1::2]
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    candidates: list[Sample] = []
    for index in indices:
        row = rows[index]
        image_value = row.get("image", row.get("pixel_values"))
        mask_value = row.get("mask", row.get("label"))
        image_bytes = _embedded_bytes(image_value)
        mask_bytes = _embedded_bytes(mask_value)
        mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
        if mask.ndim == 3:
            class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            for palette_index, color in enumerate(VOC_COLORS[1:], start=1):
                class_mask[np.all(mask == color, axis=-1)] = palette_index
            mask = class_mask
        class_ids, counts = np.unique(mask[(mask >= 1) & (mask <= 20)], return_counts=True)
        if len(class_ids) == 0:
            continue
        best_position = int(np.argmax(counts))
        class_id = int(class_ids[best_position])
        labels = sorted({VOC20[int(value) - 1] for value in class_ids})
        sample_id = f"voc2012-{index:06d}"
        image_path = extracted / f"{sample_id}.jpg"
        if not image_path.exists():
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(image_path, quality=95)
        candidates.append(Sample(
            sample_id=sample_id, image_path=str(image_path.resolve()),
            target_label=VOC20[class_id - 1], target_class_id=class_id,
            target_area=float(counts[best_position]) / float(mask.size), labels=labels,
            # The materialized JPEG is the exact artifact consumed by the
            # renderer, so provenance must hash those bytes rather than the
            # pre-encoding parquet payload.
            source_sha256=sha256_file(image_path),
        ))
        if not include_sample_ids and not exclude_sample_ids and len(candidates) >= n:
            break
    return _apply_sample_filters(candidates, n, include_sample_ids, exclude_sample_ids)


def load_dataset(
    name: str,
    root: str | Path,
    n: int,
    seed: int,
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
    dataset_split: str | None = None,
) -> list[Sample]:
    if name == "coco128":
        return load_coco128(root, n, seed, include_sample_ids=include_sample_ids, exclude_sample_ids=exclude_sample_ids)
    if name == "coco_val2017":
        return load_coco_val2017(
            root, n, seed,
            include_sample_ids=include_sample_ids,
            exclude_sample_ids=exclude_sample_ids,
            dataset_split=dataset_split,
        )
    if name == "coco_val2017_hf":
        return load_coco_val2017_hf(
            root, n, seed,
            include_sample_ids=include_sample_ids,
            exclude_sample_ids=exclude_sample_ids,
            dataset_split=dataset_split,
        )
    if name == "voc2007_test":
        return load_voc2007_test(
            root, n, seed,
            include_sample_ids=include_sample_ids,
            exclude_sample_ids=exclude_sample_ids,
            dataset_split=dataset_split,
        )
    if name == "voc2012_segmentation_hf":
        return load_voc2012_segmentation_hf(
            root, n, seed,
            include_sample_ids=include_sample_ids,
            exclude_sample_ids=exclude_sample_ids,
            dataset_split=dataset_split,
        )
    raise ValueError(f"Unsupported dataset: {name}")
