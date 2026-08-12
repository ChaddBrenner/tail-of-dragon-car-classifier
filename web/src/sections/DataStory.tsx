import { formatNumber } from "../lib";
import type { NoiseReview } from "../types";

const CONFIDENCE_LABELS: Record<string, string> = {
  top_prob_ge_0_90: "≥ 0.90",
  top_prob_0_75_0_90: "0.75–0.90",
  top_prob_0_50_0_75: "0.50–0.75",
  top_prob_lt_0_50: "< 0.50"
};

export function DataStory({ review }: { review: NoiseReview }) {
  const confidenceRows = Object.entries(review.error_confidence_buckets);
  const maxConfidence = Math.max(...confidenceRows.map(([, count]) => count));

  return (
    <section className="section dataSection" id="data">
      <div className="sectionHeading">
        <div>
          <h2>Why I stopped at 99.1%</h2>
          <p>I read the remaining 210 misses before deciding whether to keep optimizing.</p>
        </div>
      </div>

      <ol className="lineage">
        <li>
              <strong>110,204</strong>
              <span>raw photos</span>
            </li>
            <li>
              <strong>86,197</strong>
              <span>training records, duplicate-filename leakage removed</span>
            </li>
            <li>
              <strong>{formatNumber(review.total)}</strong>
              <span>held-out validation images, split inherited unchanged</span>
            </li>
      </ol>

      <div className="dataGrid">
        <div className="dataProse">
          <p>
            I read every one of the 210 misses. Many of them were not model failures. They were wrong labels, cars too
            distant or occluded to identify, and frames with more than one vehicle in them. Training past that point
            would have meant fitting the noise in the labels rather than the cars.
          </p>
          <p>
            So I set the practical ceiling for this version of the dataset at{" "}
            <strong className="ceiling">99.0% to 99.3%</strong> and stopped there.{" "}
            <a
              className="textLink"
              href="https://github.com/ChaddBrenner/tail-of-dragon-car-classifier/blob/main/reports/NOISE_REVIEW.md"
            >
              Read the noise review
            </a>
          </p>
          <p>
            I screened seven architectures first: ConvNeXt-Tiny and Small, EfficientNetV2-S, Swin-T, CoAtNet-0,
            MaxViT-T, and ViT-S. Then I trained the winner with AMP, cosine LR, class-balanced sampling, label
            smoothing, mixup, CutMix, RandAugment, random erasing, and horizontal-flip TTA. The screening scores came
            from one epoch on a fifth of the data. That was a filter, not a fair comparison.
          </p>
        </div>

        <div className="chartCard">
          <h3>How confident the 210 misses were</h3>
          <p>Most of the remaining errors are ones the model was already unsure about.</p>
          <div className="confidenceBars">
            {confidenceRows.map(([bucket, count]) => (
              <div key={bucket}>
                <span>{CONFIDENCE_LABELS[bucket] ?? bucket}</span>
                <i>
                  <b style={{ width: `${(count / maxConfidence) * 100}%` }} />
                </i>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
          <p className="chartFootnote">
            <strong>{review.low_conf_correct_count}</strong> correct predictions were also low-confidence. That is the
            same ambiguity, resolved the right way.
          </p>
        </div>
      </div>

      <p className="scopeNote">
        The model sorts curated Tail of the Dragon photographs into eight gallery labels. It does not recognize
        vehicles in general. It should not be used to identify trim levels, and it should not be used for anything
        safety-critical.
      </p>
    </section>
  );
}
