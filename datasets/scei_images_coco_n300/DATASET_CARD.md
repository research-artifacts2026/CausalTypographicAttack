# SCEI-Images-300

SCEI-Images-300 is an image-first research dataset for evaluating whether
vision-language models treat scene-compatible but mechanically false text as
visual evidence. The registered source suite contains 300 COCO val2017 scenes
and eight balanced counterfactual families.

## Contents

| Artifact | Public count | Meaning |
|---|---:|---|
| Source scenes represented in metadata | 300 | Complete registered selection |
| Scenes with released pixels | 206 | Source license permits derivatives |
| Released images | 618 | Clean, false, and corrected variants for 206 scenes |
| Released masks | 600 | False and corrected overlay masks for all 300 scenes |
| Symbolic records | 300 | Registered false/corrected measurements and assumptions |
| Pixel-withheld scenes | 94 | Source license is CC BY-NC-ND 2.0 or CC BY-ND 2.0 |

Each released scene has three image variants:

- `images/clean`: the registered COCO source image without an overlay;
- `images/attack_false`: a scene-adaptive carrier containing one mechanically
  false field;
- `images/control_true`: the corresponding corrected field using matched
  carrier geometry and layout.

The `masks/` directory contains the two project-generated overlay masks for
every registered scene. The `records/` directory contains the symbolic
counterfactual definition for every scene.

## Counterfactual families

The full registered suite contains range/threshold, unit conversion, temporal
ledger, capacity conservation, causal order, geometry feasibility, probability
ledger, and phase/state examples. The pixel-release family counts are recorded
in `release_audit.json`; the licensing filter is not an experimental split and
must not be used to claim performance on all 300 scenes.

## Manifests

- `manifest.jsonl` and `selection.jsonl`: the 206-scene pixel release;
- `full_n300_manifest.jsonl` and `full_n300_selection.jsonl`: complete
  300-scene metadata, including entries whose pixels are withheld;
- `source_licenses.jsonl`: COCO image ID, Flickr/COCO source URL, license ID,
  license name, and license URL for all 300 scenes;
- `full_n300_provenance.json`: original construction configuration and model
  provenance;
- `full_n300_audit_report.json`: independent audit of the complete source
  suite;
- `release_audit.json`: release counts, allowed licenses, and all withheld IDs;
- `SHA256SUMS`: hashes for every packaged artifact except the checksum file
  itself.

## Construction boundary

The planner observes the clean image, visible labels, target label, and a
registered counterfactual family. It chooses a scene anchor, carrier, title,
placement, and short rationale. It never sees victim-model outputs and does not
choose the registered numeric truth values. Rendering uses deterministic
perspective, local tone, texture, placement, and shadow adaptation. It is
synthetic compositing, not diffusion inpainting or physical capture.

The complete 300-scene construction audit reports 900 images, 600 masks, 300
valid planner outputs, zero planner fallbacks, and zero audit errors. This does
not measure attack success. Victim-model outcomes must be collected under a
separately frozen protocol.

## Licensing and intended use

Image pixels remain governed by their per-image upstream licenses. Read
`SOURCE_LICENSES.md` and `source_licenses.jsonl` before redistribution. The
dataset is intended for authorized robustness research, evaluation, and
defense development. Do not present generated claims as factual annotations or
use them to mislead people.

## Integrity check

From this directory, run:

```bash
sha256sum -c SHA256SUMS
```
