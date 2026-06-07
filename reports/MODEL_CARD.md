# Model Card

## Model

- Architecture: `convnext_tiny`
- Input size: `288x288`
- Classes: `bmw`, `corvette`, `honda`, `jeep`, `miata`, `mustang`, `porsche`, `vw`
- Validation accuracy: `99.12%`
- Macro F1: `0.9912`
- Top-3 accuracy: `99.77%`

## Intended Use

Classify curated Tail of the Dragon car photos into eight broad car categories
for a public portfolio demonstration.

## Data

The raw dataset is local-only and excluded from git. The validation split is the
existing held-out `train_validation/validation` split, with train/validation
duplicate filename leakage removed from training.

## Training Notes

The champion was selected after a small architecture search over ConvNeXt,
EfficientNetV2, Swin, CoAtNet, MaxViT, and ViT/AugReg variants. The final run
used class-balanced sampling, label smoothing, mixup, CutMix, RandAugment,
random erasing, cosine learning rate scheduling, AMP, and horizontal-flip TTA.

## Limitations

The model is trained on a specific photo source and road setting. Predictions
should not be treated as a general vehicle-recognition system outside this
domain.

The validation set contains some mislabeled, unreadable, distant, occluded, and
multi-vehicle images. Manual review places the practical ceiling around
`99.0-99.3%` accuracy for this dataset.
