import { classLabel, formatNumber, gallerySrcSet, pct } from "../lib";
import type { Metrics, Sample } from "../types";
import { Photo, PhotoSkeleton } from "../components/Photo";

export function Hero({ classes, metrics, samples }: { classes: string[]; metrics: Metrics; samples: Sample[] }) {
  // One photograph per class, so the eight labels are visible before any of the
  // numbers are. This is the fastest way to explain what the model sorts.
  const classPhotos = classes
    .map((className) => samples.find((sample) => sample.true_class === className))
    .filter((sample): sample is Sample => Boolean(sample));

  const evaluated = metrics.evaluated_samples ?? 23980;
  const missed = metrics.error_count ?? 210;

  return (
    <section className="hero" id="top">
      <div className="heroLead">
        <h1>A road with 318 curves, and a photograph of every car that drives it.</h1>
        <p>
          Businesses along the Tail of the Dragon shoot the cars coming through and sell the prints back to the
          drivers. Someone has to sort thousands of those photographs into a gallery per car. I trained a model to do
          that sorting.
        </p>

        <p className="accuracyFigure">
          <span className="figure">{pct(metrics.accuracy, 2)}</span>
          <span className="figureLabel">accuracy</span>
        </p>
        <p className="figureCaption">
          That is {formatNumber(missed)} wrong out of {formatNumber(evaluated)} photographs the model had never seen.
        </p>

        <dl className="metricStrip" aria-label="How the dataset was built">
          <div>
            <dt>Photographs collected</dt>
            <dd>110,204</dd>
          </div>
          <div>
            <dt>Used for training</dt>
            <dd>86,197</dd>
          </div>
          <div>
            <dt>Held back for testing</dt>
            <dd>{formatNumber(evaluated)}</dd>
          </div>
        </dl>
      </div>

      <div className="heroPhotoGrid" aria-label="One validation photograph from each of the eight classes">
        {classPhotos.length
          ? classPhotos.map((sample, index) => (
              <figure key={sample.id}>
                <Photo
                  alt={`A ${classLabel(sample.true_class)} on the Tail of the Dragon`}
                  eager={index < 4}
                  sizes="(max-width: 520px) 44vw, (max-width: 900px) 22vw, 155px"
                  src={sample.image}
                  srcSet={gallerySrcSet(sample.image)}
                />
                <figcaption>{classLabel(sample.true_class)}</figcaption>
              </figure>
            ))
          : Array.from({ length: 8 }, (_, index) => <PhotoSkeleton key={index} />)}
      </div>
    </section>
  );
}
