# tcg-vision

[![ci](https://github.com/IfanTsai/tcg-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/IfanTsai/tcg-vision/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/IfanTsai/tcg-vision.svg)](https://pkg.go.dev/github.com/IfanTsai/tcg-vision)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Recognize trading cards in photos, in pure Go (ONNX Runtime under the hood):

```
photo ──▶ oriented-box detector ──▶ perspective rectification per card
      ──▶ artwork-window ViT embedding ──▶ cosine top-K against a reference index
```

Works on sleeved, foil-heavy, rotated cards in real photos. Flat scans and
official renders — which the photo-trained detector scores near zero — are
handled by a whole-image fallback when the image itself has a card-like
aspect ratio (`Profile.FlatAspectTol`, reported as `Detection.Flat`).

The pipeline is game-agnostic; game specifics (detector model, card aspect
ratio, artwork window) live in a `Profile` — Yu-Gi-Oh! is the first built-in
profile.

## How it works

1. **Detect** — a YOLO oriented-bounding-box model finds cards in the photo
   (any rotation, multiple cards per photo).
2. **Rectify** — each quad is perspective-warped to an upright canonical card.
3. **Embed** — only the artwork window is embedded with a DINOv2-small
   backbone (CLS + mean-patch concat, 768d, L2-normalized). Card frames and
   foil textures look alike across cards; artwork does not — embedding only
   the artwork is what makes foil/prismatic printings match reliably.
   (Perceptual hashing and other pixel-level methods fall apart exactly
   there: glare and holo patterns dominate the hash while the artwork
   signal drowns. A semantic embedding of just the artwork survives them.)
4. **Search** — brute-force cosine against an index of reference card images
   (official renders and/or printing photos). ~48k references search in
   ~10 ms on one CPU core.

## Does it work?

On a private evaluation set of 9 handheld photos of **sleeved foil cards**
(prismatic secrets, holo patterns, glare, tilt) against a ~48k-image
reference index: **9/9 card-level top-1**, with top-1 similarity 0.82–0.92
and a margin ≥ 0.03 over the runner-up card. Typical timings: 1–3 s per
photo end-to-end on 2 vCPUs (CPU-only), of which the cosine search is
~10 ms.

Real output for a photo of a sleeved prismatic card (`recognize -k 3`):

```json
[
 {
  "Poly": [[53.2, 155.5], [783.7, 160.1], [777.6, 1118.6], [47.2, 1114.0]],
  "Conf": 0.923,
  "Flat": false,
  "Matches": [
   {"Key": "yugipedia/5/55/StarvingVenomFusionDragonFourHeavenlyDragons-LOCR-JP-UR.jpg",    "Sim": 0.852},
   {"Key": "yugipedia/5/57/StarvingVenomFusionDragonFourHeavenlyDragons-LOCR-JP-PScR.jpg", "Sim": 0.851},
   {"Key": "yugipedia/a/ae/StarvingVenomFusionDragonFourHeavenlyDragons-LOCR-JP-UR-EA.jpg","Sim": 0.844}
  ]
 }
]
```

Keys are whatever you indexed — here, reference image paths whose names
encode the card and printing. What similarity counts as "confident" is
yours to calibrate per domain: in our data, photo-vs-photo hits land at
0.80+, photo-vs-official-render hits at 0.65–0.75.

## Requirements

- Go 1.22+, cgo enabled
- ONNX Runtime shared library: `make ort` downloads the official **linux-x64**
  release into `third_party/`. On other platforms, grab the matching build from
  the [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
  yourself — the library is loaded at runtime from the path you pass in, so any
  OS/arch onnxruntime supports works.
- The two ONNX models (not distributed with this repo): `make models` exports
  them with Python (see `export/`), or bring your own

## Install

```sh
go get github.com/IfanTsai/tcg-vision
```

## Model licensing

The code in this repository is MIT. The default models are **not** part of
this repository and carry their own licenses:

- Detector: [HichTala/draw2](https://huggingface.co/HichTala/draw2) YOLO
  weights, **AGPL-3.0** — evaluate whether that license fits your deployment
  before using it; swap in your own detector to avoid it.
- Embedder: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small),
  Apache-2.0.

## CLI

```sh
make cli    # builds ./tcgvision (or: go install github.com/IfanTsai/tcg-vision/cmd/tcgvision@latest)

# build a reference index from a directory of card images
./tcgvision index -ort third_party/onnxruntime-*/lib/libonnxruntime.so \
  -embedder models/embedder.onnx -images ./reference-images -out index.bin

# recognize cards in a photo
./tcgvision recognize -ort third_party/onnxruntime-*/lib/libonnxruntime.so \
  -detector models/detector.onnx -embedder models/embedder.onnx \
  -index index.bin -photo photo.jpg
```

## Library

```go
pipe, err := tcgvision.New(tcgvision.Config{
    ORTLibPath:   "libonnxruntime.so",
    DetectorPath: "detector.onnx",
    EmbedderPath: "embedder.onnx",
    Threads:      2,                  // CPU-friendly default
    Profile:      tcgvision.YuGiOh(),
})

idx, err := tcgvision.LoadIndex("index.bin")

dets, err := pipe.Recognize(photo, idx, 5)
// Pass the photo at full resolution: anything larger than
// Profile.MaxPhotoSide is downscaled internally, and coordinates come back
// in the input photo's pixel space. Detections are in reading order
// (rows top to bottom, left to right within a row).
// dets[i].Poly    — card quad in photo coordinates
// dets[i].Conf    — detector confidence
// dets[i].Matches — top-K {reference image key, cosine similarity}
```

Indexing reference images:

```go
vec, err := pipe.EmbedReference(cardImage) // flat render or printing photo
idx.Add("cards/89631139.jpg", vec)
idx.Save("index.bin")
```

## Limitations

- **No rarity recognition.** Embedding only the artwork window deliberately
  suppresses glare and holo texture — which is also the signal that would
  distinguish a foil printing from a common. The pipeline tells you *which
  card* it is (and which reference images it resembles), not which rarity.
- **Domain gap between photos and flat renders.** A photo matched against
  another photo scores noticeably higher than against an official render of
  the same card (0.80+ vs 0.65–0.75 in our data). If your index mixes both,
  calibrate thresholds per domain rather than globally.
- **Detector license.** The default DRAW 2 detector weights are AGPL-3.0
  (see Model licensing) — evaluate that before deploying, or train and swap
  in your own detector; the pipeline only assumes YOLO-OBB output geometry.

## Testing

`go test ./...` runs the pure-math tests everywhere. The golden alignment
suite (real photos vs. the Python reference implementation) only runs when
private assets exist under `testdata/private/` — real card photos are not
distributed with the repository.

## License

MIT — see [LICENSE](LICENSE). The default models carry their own licenses
(see [Model licensing](#model-licensing)).
