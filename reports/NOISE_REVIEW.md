# Dataset Noise Review

## Summary

Final validation accuracy is `99.12%` on `23,980` held-out images, with `210`
misses. A manual review of generated contact sheets suggests the realistic
ceiling for this dataset is roughly `99.0-99.3%`.

That range is a practical stopping point, not a mathematical proof. The dataset
contains enough label noise, unreadable images, partial views, multiple-car
frames, and forced broad classes that spending hours for a possible one-point
or sub-one-point gain is not justified.

## Review Method

Generated sheets:

- `reports/noise_review/errors_high_conf.jpg`
- `reports/noise_review/errors_random.jpg`
- `reports/noise_review/errors_low_margin.jpg`
- `reports/noise_review/correct_low_conf.jpg`
- `reports/noise_review/correct_random.jpg`
- `reports/noise_review/correct_high_conf.jpg`

The review looked at high-confidence misses, random misses, low-margin misses,
low-confidence correct predictions, random correct predictions, and
high-confidence correct predictions.

## Observations

- Many high-confidence misses are likely mislabeled or forced into a nearby
  broad class rather than being clear model failures. Examples included
  VW-labeled images that visually looked like BMW sedans, Honda-labeled images
  that looked like VW hatchbacks, and roadsters/muscle cars that were hard to
  assign to the dataset's limited labels.
- Several misses are inherently hard images: distant vehicles, heavy shadows,
  partial crops, occlusions, motorcycles or traffic near the target, and frames
  with more than one plausible vehicle.
- The low-confidence correct sheet showed the model is appropriately uncertain
  on hard images rather than simply overfitting every validation example.
- The high-confidence correct sheet looked sane: obvious class examples were
  predicted confidently and consistently.

## Error Profile

`210` validation misses by known label:

| Class | Misses |
| --- | ---: |
| bmw | 52 |
| corvette | 29 |
| honda | 65 |
| jeep | 6 |
| miata | 10 |
| mustang | 17 |
| porsche | 11 |
| vw | 20 |

Top confusion pairs:

| True | Predicted | Count |
| --- | --- | ---: |
| honda | vw | 25 |
| bmw | vw | 15 |
| bmw | corvette | 11 |
| corvette | mustang | 11 |
| honda | bmw | 11 |
| corvette | miata | 10 |

## Stopping Rule

Treat `99.0%` full-validation accuracy and `0.99` macro F1 as the meaningful
target for this dataset. Further work is only worthwhile if a new experiment
shows a clear systematic error pattern, a cheaper architecture with similar
accuracy, or a material improvement above this range without a long search.
