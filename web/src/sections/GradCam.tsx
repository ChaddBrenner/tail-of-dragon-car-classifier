import { useState } from "react";
import { ArrowIcon } from "../components/Icons";
import { Photo } from "../components/Photo";
import { classLabel, pct } from "../lib";
import type { AnalysisSample } from "../types";

export function GradCam({ samples }: { samples: AnalysisSample[] }) {
  const [index, setIndex] = useState(0);
  const [split, setSplit] = useState(50);
  const sample = samples[index];

  if (!sample) return null;

  function move(direction: number) {
    setIndex((current) => (current + direction + samples.length) % samples.length);
    setSplit(50);
  }

  return (
    <section className="section gradcamSection" id="explain">
      <div className="sectionHeading">
        <div>
          <h2>What the model looks at</h2>
          <p>Warm regions are the pixels that most raised the predicted class score.</p>
        </div>
        <div className="samplePager" aria-label="Grad-CAM example navigation">
          <button aria-label="Previous Grad-CAM example" onClick={() => move(-1)} type="button">
            <ArrowIcon className="previousIcon" />
          </button>
          <span>{index + 1} / {samples.length}</span>
          <button aria-label="Next Grad-CAM example" onClick={() => move(1)} type="button">
            <ArrowIcon />
          </button>
        </div>
      </div>

      <div
        className="compareSlider"
        style={{ "--split": `${split}%` } as React.CSSProperties}
      >
        <Photo alt={`${classLabel(sample.true_class)} original validation image`} eager src={sample.original ?? sample.image} />
        <Photo alt={`${classLabel(sample.top_prediction)} Grad-CAM heatmap`} className="heatmapPhoto" eager src={sample.heatmap ?? sample.image} />
        <span className="compareLabel originalLabel">Original</span>
        <span className="compareLabel heatmapLabel">Grad-CAM</span>
        <span aria-hidden="true" className="compareDivider"><i>Ⅱ</i></span>
        <input
          aria-label="Reveal more or less of the Grad-CAM heatmap"
          max="100"
          min="0"
          onChange={(event) => setSplit(Number(event.target.value))}
          type="range"
          value={split}
        />
      </div>

      <div className="compareStack" aria-label="Original and Grad-CAM comparison">
        <figure>
          <Photo alt={`${classLabel(sample.true_class)} original validation image`} src={sample.original ?? sample.image} />
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          <Photo alt={`${classLabel(sample.top_prediction)} Grad-CAM heatmap`} src={sample.heatmap ?? sample.image} />
          <figcaption>Grad-CAM</figcaption>
        </figure>
      </div>

      <p className="gradcamCaption">
        Known <strong>{classLabel(sample.true_class)}</strong>
        <span aria-hidden="true">·</span>
        Predicted <strong>{classLabel(sample.top_prediction)}</strong>
        <span aria-hidden="true">·</span>
        <b>{pct(sample.top_probability, 1)}</b>
      </p>
    </section>
  );
}
