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

      <div className="dataGrid">
        <div className="dataProse">
          <h3>What the misses actually were</h3>
          <p>
            I read every one of the 210 misses. Many of them are not model failures. Several are VW-labeled
            photographs that look like BMW sedans, and Honda-labeled photographs that look like VW hatchbacks. The
            rest are mostly distant cars, heavy shadows, partial crops, and frames with more than one plausible
            vehicle in them.
          </p>
          <p>
            The labels are part of the problem. They come from how the photography business sorts its galleries, not
            from a clean taxonomy, so some manufacturers are grouped together and a few roadsters and muscle cars have
            no honest home among the eight. Training past this point would have meant fitting that, not the cars.
          </p>
          <p>
            So I set the practical ceiling for this version of the dataset at{" "}
            <strong className="ceiling">99.0% to 99.3%</strong> and stopped there. That is a practical stopping point,
            not a mathematical proof. More hours for a possible one-point gain was not a trade I wanted to make.{" "}
            <a
              className="textLink"
              href="https://github.com/ChaddBrenner/tail-of-dragon-car-classifier/blob/main/reports/NOISE_REVIEW.md"
            >
              Read the noise review
            </a>
          </p>
          <h3>How the model was chosen</h3>
          <p>
            I screened seven architectures first: ConvNeXt-Tiny and Small, EfficientNetV2-S, Swin-T, CoAtNet-0,
            MaxViT-T, and ViT-S. Then I trained the winner with AMP, cosine LR, class-balanced sampling, label
            smoothing, mixup, CutMix, RandAugment, random erasing, and horizontal-flip TTA. The screening scores came
            from one epoch on a fifth of the data. That was a filter, not a fair comparison.
          </p>

          <h3>When I would pick this back up</h3>
          <ul className="resumeConditions">
            <li>A clear systematic error pattern.</li>
            <li>A cheaper architecture that holds the same accuracy.</li>
            <li>A real improvement above this range that does not take a long search.</li>
          </ul>
          <p>None of those are true today.</p>
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
            <strong>{review.low_conf_correct_count}</strong> correct predictions were also low-confidence. The model
            is unsure on hard images rather than confident about everything, which is what I wanted to see.
          </p>
        </div>
      </div>

      <p className="scopeNote">
        The model sorts curated Tail of the Dragon photographs into eight gallery labels. It does not recognize
        vehicles in general. It should not be used to identify trim levels, and it should not be used for anything
        safety-critical. This site is gallery-only on purpose: there is no upload endpoint, no public inference API,
        no database, and no user accounts. Every photograph on this page is one the model was scored on.
      </p>
    </section>
  );
}
