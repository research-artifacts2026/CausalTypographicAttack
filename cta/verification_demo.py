"""Full-ink, matched flat records for user-selected exploratory images."""
from pathlib import Path
import hashlib
from PIL import Image, ImageDraw, ImageFont, ImageOps
from cta.scei_attack import compile_counterfactual, fallback_scene_plan
from cta.scene_question_designer import build_scene_question
from cta.contraledger import neutral_record
from cta.contraledger_threeway import render_item
from cta.verification_workbench import sha


def demo_rows(image_path: Path, label: str, family: str, root: Path) -> list[dict]:
    if not label.strip() or len(label) > 60:
        raise ValueError("Provide the visible object label (1–60 characters).")
    root.mkdir(parents=True, exist_ok=False)
    photo = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if photo.width < 32 or photo.height < 32:
        raise ValueError("Image is too small.")
    photo.thumbnail((1200, 900))
    canvas = Image.new("RGB", (1200, photo.height + 360), "white")
    canvas.paste(photo, ((1200 - photo.width) // 2, 0))
    source = root / "source.png"
    canvas.save(source)
    item = "upload-" + hashlib.sha256((sha(source) + label + family).encode()).hexdigest()[:16]
    record = neutral_record(compile_counterfactual(label, family, variant_key=item, seed=20260914))
    plan = fallback_scene_plan(label, family, item)
    question = build_scene_question(record, visible_object=label, truth="false", item_id=item)
    row = {"item_id": item, "dataset": "interactive-upload", "family": family,
           "scenario_id": record.scenario_id, "target_label": label,
           "source_path": str(source), "source_sha256": sha(source),
           "record": record.to_dict(), "plan": plan.to_dict(), "scene_question": question.to_dict()}
    rows = render_item(row, root, permutation_index=int(item[-4:], 16) % 6)
    # Replace only this new exploratory packet's images. Historical rendering
    # remains untouched. The footer is explicitly a flat digital carrier.
    texts = [[plan.title.upper(), plan.anchor_phrase.upper(), *r["registered_read_text"].split(" | ")]
             for r in rows[1:]]
    mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(mask).rectangle((0, photo.height, 1199, canvas.height - 1), fill=255)
    mask_path = root / "flat_mask.png"
    mask.save(mask_path)
    font_path = "DejaVuSans.ttf"
    # Use identical font sizes and line positions for both members of the pair.
    for size in range(28, 9, -1):
        try:
            font = ImageFont.truetype(font_path, size)
        except OSError:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        boxes = [font.getbbox(line) for lines in texts for line in lines]
        line_height = max(b[3] - b[1] for b in boxes) + 12
        if max(b[2] - b[0] for b in boxes) <= 1144 and max(map(len, texts)) * line_height <= 320:
            break
    else:
        raise ValueError("Complete record cannot fit; use a shorter object label.")
    for r, lines in zip(rows[1:], texts):
        image = canvas.copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, photo.height, 1199, image.height - 1), fill="#f0f5f4")
        for index, line in enumerate(lines):
            box = font.getbbox(line)
            draw.text((28 - box[0], photo.height + 20 + index * line_height - box[1]), line, font=font, fill="#17352e")
        path = root / (r["condition"] + ".png")
        image.save(path)
        r.update(image_path=str(path), image_sha256=sha(path), mask_path=str(mask_path), mask_sha256=sha(mask_path),
                 carrier_quad=[[0, photo.height], [1200, photo.height], [1200, image.height], [0, image.height]],
                 renderer="workbench-full-ink-flat-v1", overlay_area_fraction=360 / canvas.height)
    return rows
