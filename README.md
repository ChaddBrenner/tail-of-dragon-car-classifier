# Tail of Dragon Car Classifier

Image-classification portfolio project for Tail of the Dragon car photos. The
model classifies images into eight gallery labels: `bmw`, `corvette`, `honda`,
`jeep`, `miata`, `mustang`, `porsche`, and `vw`.

The website is intentionally gallery-only. Visitors choose curated validation
photos, refresh through a larger static photo pool, and see the model's full
ranked probability weights. There is no upload endpoint, no public inference
API, no database, and no user accounts.

## Portfolio Features

- Refreshable validation gallery with full ranked class probabilities.
- Interactive error analysis across high-confidence misses, low-margin misses,
  random misses, low-confidence correct examples, and random correct examples.
- Clickable top-confusion matrix with the images behind each confusion pair.
- Grad-CAM explainability review for representative correct and incorrect
  validation samples.
- Technical case-study page covering the baseline, search space, final metrics,
  and practical stopping rule.
- Dataset quality dashboard summarizing final misses, confidence buckets, and
  class-level error concentration.
- Curated-only browser inference demo using ONNX Runtime Web. It runs the
  exported model locally on gallery photos only and does not accept uploads.
- Model card and supply-chain pages covering use boundaries, limitations, GHCR
  image metadata, SBOM, and provenance.

## Results

The inherited checkpoint was a `convnext_tiny` model at 77.09% validation
accuracy and 0.7701 macro F1. The rebuilt champion is:

| Model | Image size | Validation accuracy | Macro F1 | Top-3 accuracy |
| --- | ---: | ---: | ---: | ---: |
| `convnext_tiny` | 288 | 99.12% | 0.9912 | 99.77% |

The validation split contains 23,980 images. The final model missed 210. Manual
review of error sheets showed many of those misses are mislabeled, ambiguous,
unreadable, distant, occluded, or contain multiple vehicles, so the practical
target for this dataset is around 99.0-99.3% rather than an arbitrary 100%.
See `reports/NOISE_REVIEW.md` for the stopping rule.

## What Changed

- Preserved the existing train/validation split and removed duplicate filename
  leakage from training.
- Rebuilt training around `timm`, AMP, cosine learning rate, class-balanced
  sampling, label smoothing, mixup, CutMix, RandAugment, random erasing, TTA,
  checkpointing, metric exports, and reproducible manifests.
- Compared ConvNeXt, EfficientNetV2, Swin, CoAtNet, MaxViT, and ViT/AugReg
  candidates on a controlled search pass.
- Exported the champion to ONNX and verified PyTorch/ONNX parity. The deployable
  browser-demo ONNX assets are tracked under `web/public/model/` with Git LFS;
  raw checkpoints and local experiment exports remain excluded from git.
- Built a static React/TypeScript gallery served by Caddy in Docker.

## Repo Layout

```text
training/              Training, report, export, and gallery generation scripts
web/                   Static React + TypeScript app
deploy/                Caddy runtime config and reverse-proxy example
reports/               Final metrics and model documentation
.github/workflows/     GHCR Docker image build workflow
docker_deploy.yml      Self-host compose file
```

The raw dataset, checkpoints, local tools, `.venv`, and training runs are
intentionally excluded from git. The public ONNX model files needed for the
curated browser demo are versioned through Git LFS.

## Local Training

Place the raw dataset next to this repo:

```text
project/
  train_validation/
    train/<class>/*.jpg
    validation/<class>/*.jpg
  tail-of-dragon-car-classifier/
```

Create the training environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

Run a smoke test:

```powershell
.\.venv\Scripts\python.exe training\train.py --data-dir ..\train_validation --mode smoke --output-dir runs\smoke
```

Run search and final training:

```powershell
.\.venv\Scripts\python.exe training\train.py --data-dir ..\train_validation --mode search --output-dir runs\search --device cuda --batch-size 32 --num-workers 4 --search-epochs 1 --search-train-fraction 0.20 --search-val-fraction 0.25
.\.venv\Scripts\python.exe training\train.py --data-dir ..\train_validation --mode final --output-dir runs\final_convnext_tiny --device cuda --batch-size 32 --num-workers 4 --final-model convnext_tiny --final-img-size 288 --final-epochs 10
```

Generate reports from saved predictions:

```powershell
.\.venv\Scripts\python.exe training\report_from_predictions.py --predictions runs\final_convnext_tiny\final_00_convnext_tiny_288\best_val_predictions.json --output-dir runs\final_convnext_tiny\final_00_convnext_tiny_288
```

## Static Gallery Build

```powershell
.\.venv\Scripts\python.exe training\export_model.py --checkpoint runs\final_convnext_tiny\final_00_convnext_tiny_288\best.pt
.\.venv\Scripts\python.exe training\build_gallery.py --data-dir ..\train_validation --checkpoint runs\final_convnext_tiny\final_00_convnext_tiny_288\best.pt --samples-per-class 24 --display-count 48
.\.venv\Scripts\python.exe training\build_portfolio_assets.py --checkpoint runs\final_convnext_tiny\final_00_convnext_tiny_288\best.pt --predictions runs\final_convnext_tiny\final_00_convnext_tiny_288\best_val_predictions.json
cd web
npm ci
npm run build
```

If you clone the repo, run `git lfs install` before checkout or `git lfs pull`
after checkout so `web/public/model/champion.onnx.data` is the real model
sidecar file rather than an LFS pointer.

## Docker Deploy

```bash
docker compose -f docker_deploy.yml up -d
```

The app listens on host port `8080` by default. Add the reverse-proxy shape from
`deploy/Caddyfile.example` to your existing host Caddy instance.
