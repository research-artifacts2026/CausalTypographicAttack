from pathlib import Path

from cta.data import download_coco128

download_coco128(Path("data/coco128"))
print("COCO128 ready at data/coco128")

