import { classLabel, pct } from "../lib";
import type { Probability } from "../types";

export function ProbabilityBars({ probabilities, correct }: { probabilities: Probability[]; correct: boolean }) {
  return (
    <div className="probabilityBars">
      {probabilities.map((probability, index) => (
        <div className={`probabilityRow ${index === 0 ? "top" : "tail"}`} key={probability.class_name}>
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
    </div>
  );
}
