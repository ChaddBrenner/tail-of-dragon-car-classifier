import { classLabel, formatNumber, pct } from "../lib";
import type { Metrics, Sample } from "../types";
import { Photo, PhotoSkeleton } from "../components/Photo";

export function Hero({ classes, metrics, samples }: { classes: string[]; metrics: Metrics; samples: Sample[] }) {
  const heroPhotos = classes
    .map((className) => samples.find((sample) => sample.true_class === className))
    .filter((sample): sample is Sample => Boolean(sample))
    .slice(0, 4);

  return (
    <section className="hero" id="top">
      <div className="heroLead">
        <h1>I trained a model to sort Tail of the Dragon photographs into eight car classes.</h1>
        <p>
          Photographers along the road shoot passing cars and sell the prints to the drivers. Every image has to be
          sorted into the right gallery first. This model does that sorting.
        </p>

        <p className="accuracyFigure">
          <span className="figure">{pct(metrics.accuracy, 2)}</span>
          <span className="figureLabel">accuracy</span>
        </p>
        <p className="figureCaption">
          Measured on a held-out validation split that the model never saw during training.
        </p>

        <dl className="metricStrip" aria-label="Model metrics">
          <div>
            <dt>Macro F1</dt>
            <dd>{(metrics.macro_f1 ?? 0).toFixed(4)}</dd>
          </div>
          <div>
            <dt>Top-3</dt>
            <dd>{pct(metrics.top3_accuracy, 2)}</dd>
          </div>
          <div>
            <dt>Held-out images</dt>
            <dd>{formatNumber(metrics.evaluated_samples)}</dd>
          </div>
        </dl>
      </div>

      <div className="heroPhotoGrid" aria-label="Validation photo examples">
        {heroPhotos.length
          ? heroPhotos.map((sample, index) => (
              <figure key={sample.id}>
                <Photo
                  alt={`${classLabel(sample.true_class)} on the Tail of the Dragon`}
                  eager={index < 2}
                  src={sample.image}
                />
                <figcaption>{classLabel(sample.true_class)}</figcaption>
              </figure>
            ))
          : Array.from({ length: 4 }, (_, index) => <PhotoSkeleton key={index} />)}
      </div>
    </section>
  );
}
