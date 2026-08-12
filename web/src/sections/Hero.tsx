import { CLASS_COLORS, classLabel, formatNumber, pct } from "../lib";
import type { Metrics, Sample } from "../types";
import { Photo, PhotoSkeleton } from "../components/Photo";

const heroMetrics = [
  { key: "accuracy", label: "Accuracy" },
  { key: "macro_f1", label: "Macro F1" },
  { key: "top3_accuracy", label: "Top-3" },
  { key: "evaluated_samples", label: "Validation" },
  { key: "error_count", label: "Misses" }
] as const;

function metricValue(metric: (typeof heroMetrics)[number], metrics: Metrics) {
  if (metric.key === "macro_f1") return (metrics.macro_f1 ?? 0).toFixed(4);
  if (metric.key === "evaluated_samples" || metric.key === "error_count") return formatNumber(metrics[metric.key]);
  return pct(metrics[metric.key], 2);
}

export function Hero({ classes, metrics, samples }: { classes: string[]; metrics: Metrics; samples: Sample[] }) {
  const heroPhotos = classes
    .map((className) => samples.find((sample) => sample.true_class === className))
    .filter((sample): sample is Sample => Boolean(sample))
    .slice(0, 6);

  return (
    <section className="hero" id="top">
      <div className="heroLead">
        <h1>
          Photographers hand-sort thousands of Tail of the Dragon road photos. This model sorts them into eight gallery
          labels.
        </h1>
      </div>

      <div className="accuracyArc" aria-label="Accuracy improved from 77.09 percent to 99.12 percent">
        <span>77.09%</span>
        <span aria-hidden="true" className="arcArrow">→</span>
        <strong>99.12%</strong>
        <small>Inherited baseline → retrained champion · same validation split</small>
      </div>

      <dl className="metricStrip" aria-label="Model metrics">
        {heroMetrics.map((metric) => (
          <div className={metric.key === "error_count" ? "metricMiss" : ""} key={metric.key}>
            <dt>{metric.label}</dt>
            <dd>{metricValue(metric, metrics)}</dd>
          </div>
        ))}
      </dl>

      <div className="classLegend" aria-label="Gallery labels">
        {classes.map((className) => (
          <span key={className} style={{ "--class-color": CLASS_COLORS[className] } as React.CSSProperties}>
            <i aria-hidden="true" />
            {classLabel(className)}
          </span>
        ))}
      </div>

      <p className="modelCaption">ConvNeXt-Tiny · 288px · best epoch 10 · chosen from a 7-architecture search</p>

      <div className="heroPhotoRail" aria-label="Validation photo preview">
        {heroPhotos.length
          ? heroPhotos.map((sample, index) => (
              <figure key={sample.id}>
                <Photo alt={`${classLabel(sample.true_class)} on the Tail of the Dragon`} eager={index < 2} src={sample.image} />
                <figcaption>{classLabel(sample.true_class)}</figcaption>
              </figure>
            ))
          : Array.from({ length: 6 }, (_, index) => <PhotoSkeleton key={index} />)}
      </div>
    </section>
  );
}
