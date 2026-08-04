import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.data import download_coco128

download_coco128(Path("data/coco128"))
print("COCO128 ready at data/coco128")
