import { useRef } from "react";
import { ShuffleIcon } from "../components/Icons";
import { Photo, PhotoSkeleton } from "../components/Photo";
import { ProbabilityBars } from "../components/ProbabilityBars";
import { classLabel, gallerySrcSet, pct, thumbFor } from "../lib";
import type { Sample } from "../types";

type ExplorerProps = {
  onSelect: (id: string) => void;
  onShuffle: () => void;
  samples: Sample[];
  selected?: Sample;
  totalSamples: number;
};

export function Explorer({ onSelect, onShuffle, samples, selected, totalSamples }: ExplorerProps) {
  const railRef = useRef<HTMLDivElement>(null);

  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (!selected || !["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const currentIndex = samples.findIndex((sample) => sample.id === selected.id);
    const direction = event.key === "ArrowDown" || event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (currentIndex + direction + samples.length) % samples.length;
    onSelect(samples[nextIndex].id);
  }

  const correct = selected?.true_class === selected?.top_prediction;

  return (
    <section className="section" id="explore">
      <div className="sectionHeading">
        <div>
          <h2>Pick a photo and see what the model predicted</h2>
          <p>These are sixteen held-out photographs, two from each class. The arrow keys move between them.</p>
        </div>
        <button className="quietButton" disabled={totalSamples <= samples.length} onClick={onShuffle} type="button">
          <ShuffleIcon />
          Shuffle
        </button>
      </div>

      {selected ? (
        <div className="explorerGrid">
          <div
            aria-label="Curated validation thumbnails"
            className="thumbnailRail"
            onKeyDown={handleKeyDown}
            ref={railRef}
            role="listbox"
            tabIndex={0}
          >
            {samples.map((sample) => (
              <button
                aria-label={`Select ${classLabel(sample.true_class)} example`}
                aria-selected={sample.id === selected.id}
                className={sample.id === selected.id ? "selected" : ""}
                data-sample-id={sample.id}
                key={sample.id}
                onClick={() => onSelect(sample.id)}
                role="option"
                type="button"
              >
                <Photo
                  alt=""
                  sizes="(max-width: 720px) 29vw, 152px"
                  src={thumbFor(sample.image)}
                  srcSet={gallerySrcSet(sample.image)}
                />
                <span>{classLabel(sample.true_class)}</span>
              </button>
            ))}
          </div>

          <div className="selectedPhoto">
            <Photo alt={`${classLabel(selected.true_class)} validation example`} eager src={selected.image} />
            <span className={`statusBadge ${correct ? "correct" : "miss"}`}>
              <i aria-hidden="true" />
              {correct ? "Correct" : "Miss"}
            </span>
          </div>

          <aside className="predictionPanel" aria-live="polite">
            <div className="predictionSummary">
              <div>
                <span>Prediction</span>
                <strong>{classLabel(selected.top_prediction)}</strong>
              </div>
              <b className={correct ? "correctText" : "missText"}>{pct(selected.top_probability, 1)}</b>
              <div>
                <span>Known label</span>
                <strong>{classLabel(selected.true_class)}</strong>
              </div>
            </div>
            <ProbabilityBars correct={correct} probabilities={selected.probabilities} />
          </aside>
        </div>
      ) : (
        <div className="explorerGrid explorerSkeleton" aria-label="Gallery is loading">
          <div className="thumbnailRail">{Array.from({ length: 6 }, (_, index) => <PhotoSkeleton key={index} />)}</div>
          <PhotoSkeleton />
          <div className="panelSkeleton" />
        </div>
      )}
    </section>
  );
}
