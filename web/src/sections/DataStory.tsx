import { classLabel, formatNumber } from "../lib";
import type { NoiseReview } from "../types";

const confidenceLabels: Record<string, string> = {
  top_prob_ge_0_90: "≥ 0.90",
  top_prob_0_75_0_90: "0.75–0.90",
  top_prob_0_50_0_75: "0.50–0.75",
  top_prob_lt_0_50: "Below 0.50"
};

export function DataStory({ review }: { review: NoiseReview }) {
  const confidenceRows = Object.entries(review.error_confidence_buckets);
  const errorRows = Object.entries(review.errors_by_class).sort(([, a], [, b]) => b - a);
  const maxConfidence = Math.max(...confidenceRows.map(([, count]) => count));
  const maxErrors = Math.max(...errorRows.map(([, count]) => count));

  return (
    <section className="section dataSection" id="data">
      <div className="dataIntro">
        <h2>Why I stopped at 99.1%</h2>
        <p>
          The final 210 misses were more useful than another decimal place. Most remaining errors are ones the model is
          already unsure about, and manual review showed that many frames are noisy enough to cap what optimization can
          honestly accomplish.
        </p>
      </div>

      <div className="dataColumns">
        <article className="ruledStory">
          <h3>Confidence of the 210 misses</h3>
          <p>Most of the remaining errors are not confident failures.</p>
          <div className="editorialBars">
            {confidenceRows.map(([bucket, count]) => (
              <div key={bucket}>
                <span>{confidenceLabels[bucket] ?? bucket}</span>
                <i><b style={{ width: `${(count / maxConfidence) * 100}%` }} /></i>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
          <p className="inlineFinding">
            <strong>{review.low_conf_correct_count}</strong> correct predictions were also low-confidence—the useful flip
            side of the same ambiguity.
          </p>
        </article>

        <article className="ruledStory">
          <h3>Errors by known label</h3>
          <p>Honda and BMW account for more than half of all misses.</p>
          <div className="editorialBars classErrorBars">
            {errorRows.map(([className, count]) => (
              <div key={className}>
                <span>{classLabel(className)}</span>
                <i><b style={{ width: `${(count / maxErrors) * 100}%` }} /></i>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </article>

        <article className="ruledStory lineageStory">
          <h3>Data lineage</h3>
          <div className="lineageFlow">
            <div><strong>110,204</strong><span>raw images</span></div>
            <b aria-hidden="true">→</b>
            <div><strong>86,197</strong><span>training records after duplicate-filename leakage removal</span></div>
            <b aria-hidden="true">→</b>
            <div><strong>{formatNumber(review.total)}</strong><span>held-out validation images</span></div>
          </div>
          <h3>The stopping rule</h3>
          <p>
            Manual review found bad labels, unreadable or distant cars, heavy occlusion, and multi-vehicle frames. I
            treated roughly <strong>99.0–99.3%</strong> as the practical ceiling for this dataset version and stopped.
          </p>
          <a className="textLink" href="https://github.com/ChaddBrenner/tail-of-dragon-car-classifier/blob/main/reports/NOISE_REVIEW.md">
            Read the dataset-noise review
          </a>
        </article>
      </div>

      <div className="trainingNote">
        <h3>What went into the result</h3>
        <p>
          The architecture search covered ConvNeXt-Tiny and Small, EfficientNetV2-S, Swin-T, CoAtNet-0, MaxViT-T, and
          ViT-S. The final recipe used AMP, cosine learning rate, class-balanced sampling, label smoothing, mixup, CutMix,
          RandAugment, random erasing, crop-scale tuning, and horizontal-flip TTA. The one-epoch search scores were a
          screening pass—not full-training comparisons.
        </p>
      </div>

      <div className="scopeBoundary">
        <p><strong>In scope</strong> — sorting curated Tail of the Dragon photos into eight gallery labels.</p>
        <p><strong>Out of scope</strong> — general vehicle recognition, safety-critical use, VIN or trim identification, and user uploads.</p>
      </div>
    </section>
  );
}
