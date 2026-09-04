# Source image licenses

The underlying photographs are COCO val2017 images originally sourced from
Flickr. This repository does not claim ownership of those photographs. Each
row in `source_licenses.jsonl` records the COCO image ID, source URLs, and the
license reported in COCO's official `instances_val2017.json` metadata.

Modified image pixels are included only for source license IDs 1, 2, 4, and 5:

- 1: CC BY-NC-SA 2.0;
- 2: CC BY-NC 2.0;
- 4: CC BY 2.0;
- 5: CC BY-SA 2.0.

Modified image pixels are withheld for license IDs 3 and 6 because those
licenses contain a NoDerivs restriction:

- 3: CC BY-NC-ND 2.0;
- 6: CC BY-ND 2.0.

The license filter leaves 206 scenes with released clean/false/corrected
pixels and withholds modified pixels for 94 scenes. All 94 identifiers are
listed in `release_audit.json`. The masks, symbolic records, and complete
metadata remain available because they do not redistribute the source image
pixels.

Follow the attribution, noncommercial, and ShareAlike obligations of each
source license. The Flickr page in `source_licenses.jsonl` is the source link
for identifying the original creator and photograph. No repository-wide
license overrides an image's upstream license.
