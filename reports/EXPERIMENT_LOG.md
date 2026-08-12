# Experiment Log

## Inherited Baseline

- Model: `convnext_tiny`
- Accuracy: `0.7709`
- Macro F1: `0.7701`

## Dataset

- Classes: `bmw`, `corvette`, `honda`, `jeep`, `miata`, `mustang`, `porsche`, `vw`
- Raw images: `110,204`
- Existing validation split: `23,980`
- Training records after duplicate filename leakage cleanup: `86,197`

## Search Pass

One-epoch search on 20% train / 25% validation, preserving the existing split.

| Rank | Model | Size | Accuracy | Macro F1 |
| ---: | --- | ---: | ---: | ---: |
| 1 | `convnext_tiny` | 288 | 0.7478 | 0.7460 |
| 2 | `convnext_small` | 288 | 0.7000 | 0.6996 |
| 3 | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` | 224 | 0.6753 | 0.6642 |
| 4 | `maxvit_tiny_rw_224.sw_in1k` | 224 | 0.6162 | 0.6135 |
| 5 | `coatnet_0_rw_224.sw_in1k` | 224 | 0.5617 | 0.5494 |
| 6 | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 224 | 0.5053 | 0.4999 |
| 7 | `tf_efficientnetv2_s.in21k_ft_in1k` | 300 | 0.4608 | 0.4493 |

## Final Champion

- Model: `convnext_tiny`
- Image size: `288`
- Best epoch: `10`
- Validation accuracy: `0.9912`
- Macro F1: `0.9912`
- Balanced accuracy: `0.9912`
- Top-3 accuracy: `0.9977`
- Evaluated validation samples: `23,980`
- Validation misses: `210`

Training used AMP, cosine LR, class-balanced sampling, label smoothing, mixup,
CutMix, RandAugment, random erasing, crop-scale tuning, and horizontal-flip TTA.

## Portfolio App Additions

- One scrolling model narrative with a refreshable validation explorer
- Merged error review groups and clickable top-confusion examples
- Draggable Grad-CAM comparisons for representative validation samples
- Dataset-quality, scope, and stopping-rule reasoning integrated into the page
- Responsive light/dark themes and uncropped 3:2 photography throughout

## Stopping Decision

Manual review of the 210 misses and several correct-prediction sheets showed
enough label ambiguity and unreadable images that `99.0-99.3%` is the practical
target range. See `reports/NOISE_REVIEW.md`.
