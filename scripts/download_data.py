import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cta.data import download_coco128, download_coco_val2017, download_coco_val2017_hf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["coco128", "coco_val2017", "coco_val2017_hf"], default="coco128")
args = parser.parse_args()
if args.dataset == "coco128":
    download_coco128(Path("data/coco128"))
    print("COCO128 ready at data/coco128")
elif args.dataset == "coco_val2017":
    download_coco_val2017(Path("data/coco2017"))
    print("COCO val2017 ready at data/coco2017")
else:
    download_coco_val2017_hf(Path("data/coco2017_hf"))
    print("Hugging Face COCO val2017 mirror ready at data/coco2017_hf")
