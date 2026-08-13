# Tail of the Dragon Car Classifier

[![Live site](https://img.shields.io/badge/live-carclassifier.chadd.blog-1f9d6b)](https://carclassifier.chadd.blog/)
[![Build and publish](https://github.com/ChaddBrenner/tail-of-dragon-car-classifier/actions/workflows/docker.yml/badge.svg)](https://github.com/ChaddBrenner/tail-of-dragon-car-classifier/actions/workflows/docker.yml)
[![Portfolio story](https://img.shields.io/badge/portfolio-project_story-b23a2b)](https://www.chadd.blog/posts/car-type-detection/)

On a family trip to the Smoky Mountains, I noticed photographers along the Tail
of the Dragon sorting thousands of road photos into galleries for different
cars. That left me with a question: how much of that sorting could a model do?

This repository is where that question ended up: an eight-class ConvNeXt vehicle
classifier, a reproducible evaluation trail, and a React gallery that makes the
model's good and bad decisions easy to inspect.

**[See it running at carclassifier.chadd.blog](https://carclassifier.chadd.blog/)**

![The site header showing one validation photograph from each of the eight classes, the accuracy figure, and how the dataset was split](reports/verification/desktop.png)

## What I built

The project goes beyond a notebook and a headline accuracy number:

- A PyTorch training pipeline with controlled architecture searches,
  checkpointing, class-balanced sampling, augmentation, and metric exports.
- A held-out evaluation set of 23,980 images with class-level metrics,
  confidence buckets, and manually reviewed errors.
- Clickable confusion pairs for seeing where the model struggled.
- A static React/TypeScript portfolio experience with a photo explorer,
  failure analysis, and the stopping-rule narrative.
- A GitHub Actions release path that builds the app and publishes a multi-platform
  container with provenance and SBOM metadata.

The current gallery labels are `bmw`, `corvette`, `honda`, `jeep`, `miata`,
`mustang`, `porsche`, and `vw`.

## Result

The starting ConvNeXt checkpoint reached 77.09% validation accuracy. The final
model reached:

| Model | Image size | Validation images | Accuracy | Macro F1 | Top-3 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt Tiny | 288 | 23,980 | **99.12%** | **0.9912** | **99.77%** |

That still left 210 misses, and I read every one of them. Many are not model
failures. Several are VW-labeled photographs that look like BMW sedans, and
Honda-labeled photographs that look like VW hatchbacks. The rest are mostly
distant cars, heavy shadows, partial crops, and frames with more than one
plausible vehicle in them.

The labels are part of the problem. They come from how the photography business
sorts its galleries, not from a clean taxonomy, so some manufacturers are
grouped together and a few roadsters and muscle cars have no honest home among
the eight. Based on that review I set the practical ceiling for this version of
the dataset at 99.0% to 99.3% and stopped there. That is a practical stopping
point, not a mathematical proof.

- [Read the dataset-noise review](reports/NOISE_REVIEW.md)
- [Open the model card](reports/MODEL_CARD.md)
- [See the experiment log](reports/EXPERIMENT_LOG.md)
- [Inspect the classification report](reports/classification_report.txt)

![Confusion matrix for the final eight-class model](reports/confusion_matrix.png)

## How it fits together

```mermaid
flowchart LR
  Photos["Curated road photos"] --> Train["PyTorch training and evaluation"]
  Train --> Review["Metrics and error review"]
  Review --> Gallery["React portfolio and validation explorer"]
  Gallery --> Container["Caddy container published to GHCR"]
```

The public experience is deliberately narrow. Visitors can explore saved
predictions and analysis for curated gallery images. There is no upload
endpoint, public inference API, database, or user account system.

## Run the gallery

A plain clone is enough; no large-file extension or model download is required:

```bash
cd web
npm ci
npm run dev
```

For the production container:

```bash
docker compose -f docker_deploy.yml up -d
```

The app listens on host port `8080` by default. The example Caddy configuration
in `deploy/Caddyfile.example` shows the reverse-proxy shape I use for deployment.

## Reproduce the training workflow

The raw dataset and local checkpoints are intentionally not committed. With the
same `train/<class>` and `validation/<class>` directory structure, create an
environment and run a smoke test:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
python training/train.py \
  --data-dir ../train_validation \
  --mode smoke \
  --output-dir runs/smoke
```

The longer search, final-training, and report commands are documented in
[`training/README.md`](training/README.md). The scripts write manifests beside
their artifacts so that a result can be traced back to its configuration.

## Repository map

```text
training/              Training, report, export, and gallery-generation tools
web/                   React/TypeScript portfolio app and curated analysis
reports/               Metrics, model card, error review, and final figures
deploy/                Caddy runtime and reverse-proxy examples
.github/workflows/     Validation and GHCR publishing
docker_deploy.yml      Self-hosted deployment
```

## Data and image notes

The model is specific to one road, one photo style, and eight broad gallery
labels. It is not a general vehicle-recognition system and should not be used
for identification, safety decisions, or fine-grained trim classification.

The raw training dataset is excluded from git. The public gallery contains a
small set of low-resolution, watermarked validation previews so the model's
behavior can be reviewed. Those previews are not offered as a reusable image
dataset, and the original photographers retain their rights to the source
images.

For the story behind the first version of the project, read
[Classifying Cars on the Tail of the Dragon](https://www.chadd.blog/posts/car-type-detection/).
