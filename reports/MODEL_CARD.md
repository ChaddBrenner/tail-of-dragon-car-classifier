# Model Card

## Model

- Architecture: `convnext_tiny`
- Input size: `288x288`
- Classes: `bmw`, `corvette`, `honda`, `jeep`, `miata`, `mustang`, `porsche`, `vw`
- Validation accuracy: `99.12%`
- Macro F1: `0.9912`
- Balanced accuracy: `0.9912`
- Top-3 accuracy: `99.77%`
- Evaluated validation samples: `23,980`
- Final validation misses: `210`

## Intended Use

Classify curated Tail of the Dragon car photos into eight broad car categories
for a public portfolio demonstration.

The deployed website is gallery-only. Visitors can inspect curated validation
photos, ranked probabilities, error examples, Grad-CAM overlays, and a
curated-only browser ONNX demo.

## Out of Scope

This model is not intended for general vehicle recognition, safety-critical
use, identity/VIN lookup, fine-grained trim detection, or unrestricted
user-uploaded image classification.

## Data

The raw dataset is local-only and excluded from git. The validation split is the
existing held-out `train_validation/validation` split, with train/validation
duplicate filename leakage removed from training.

The public repo contains only curated static gallery and analysis assets plus
the ONNX files needed by the browser demo. Raw training data, PyTorch
checkpoints, and experiment runs remain excluded.

## Training Notes

The champion was selected after a small architecture search over ConvNeXt,
EfficientNetV2, Swin, CoAtNet, MaxViT, and ViT/AugReg variants. The final run
used class-balanced sampling, label smoothing, mixup, CutMix, RandAugment,
random erasing, cosine learning rate scheduling, AMP, and horizontal-flip TTA.

## Evaluation

The model was selected on full validation accuracy and macro F1 while preserving
the original split for direct comparison to the inherited `77.09%` ConvNeXt
baseline. ONNX export parity was checked against PyTorch before deployment.

The site includes interactive error groups, top confusion pairs, class-level
error counts, confidence buckets, and Grad-CAM overlays to make the result
auditable instead of only reporting one headline number.

## Limitations

The model is trained on a specific photo source and road setting. Predictions
should not be treated as a general vehicle-recognition system outside this
domain.

The validation set contains some mislabeled, unreadable, distant, occluded, and
multi-vehicle images. Manual review places the practical ceiling around
`99.0-99.3%` accuracy for this dataset.

Browser inference runs on curated static images only. It intentionally does not
expose an upload flow or backend inference endpoint.

## Deployment and Supply Chain

The static React app is built into a Caddy runtime container and published to
GHCR. The GitHub Actions workflow checks out Git LFS assets and enables Docker
Buildx provenance and SBOM metadata for pushed images.
