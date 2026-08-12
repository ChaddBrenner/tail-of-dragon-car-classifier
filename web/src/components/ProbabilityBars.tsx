import { classLabel, pct } from "../lib";
import type { Probability } from "../types";

const RANKED = 3;

export function ProbabilityBars({ probabilities, correct }: { probabilities: Probability[]; correct: boolean }) {
  const ranked = probabilities.slice(0, RANKED);
  const tail = probabilities.slice(RANKED);

  return (
    <div className="probabilityBars">
      {ranked.map((probability, index) => (
        <div className="probabilityRow" key={probability.class_name}>
          <div className="probabilityLabel">
            <span>{classLabel(probability.class_name)}</span>
            <strong>{pct(probability.probability, 1)}</strong>
          </div>
          <div className="probabilityTrack">
            <span
              className={index === 0 ? (correct ? "correctFill" : "missFill") : "tailFill"}
              style={{ width: pct(probability.probability, 3) }}
            />
          </div>
        </div>
      ))}

      {tail.length ? (
        <p className="probabilityTail">
          {tail.map((probability, index) => (
            <span key={probability.class_name}>
              {index ? " · " : ""}
              {classLabel(probability.class_name)} <b>{pct(probability.probability, 1)}</b>
            </span>
          ))}
        </p>
      ) : null}
    </div>
  );
}
