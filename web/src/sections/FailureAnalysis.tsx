import { useMemo, useState } from "react";
import { Photo } from "../components/Photo";
import { CLASS_COLORS, GROUP_COPY, classLabel, pct } from "../lib";
import type { AnalysisData, AnalysisSample } from "../types";

function ErrorCard({ sample }: { sample: AnalysisSample }) {
  const runnerUp = sample.probabilities[1];
  const correct = sample.true_class === sample.top_prediction;

  return (
    <article className="errorCard">
      <Photo alt={`${classLabel(sample.true_class)} validation example`} src={sample.image} />
      <div className="errorCardBody">
        <div>
          <strong>
            {classLabel(sample.true_class)} <span aria-hidden="true">→</span> {classLabel(sample.top_prediction)}
          </strong>
          <span className={correct ? "correctText" : "missText"}>{correct ? "Correct" : "Miss"}</span>
        </div>
        <dl>
          <div>
            <dt>Confidence</dt>
            <dd>{pct(sample.top_probability, 1)}</dd>
          </div>
          <div>
            <dt>Runner-up</dt>
            <dd>{runnerUp ? `${classLabel(runnerUp.class_name)} ${pct(runnerUp.probability, 1)}` : "—"}</dd>
          </div>
        </dl>
      </div>
    </article>
  );
}

export function FailureAnalysis({ analysis, classes }: { analysis: AnalysisData; classes: string[] }) {
  const [selectedKey, setSelectedKey] = useState("honda->vw");
  const [selectedGroup, setSelectedGroup] = useState("high_confidence_misses");

  const countMap = useMemo(
    () => new Map(analysis.top_confusions.map((item) => [`${item.true_class}->${item.pred_class}`, item])),
    [analysis.top_confusions]
  );
  const exampleMap = useMemo(
    () => new Map(analysis.confusion_examples.map((item) => [`${item.true_class}->${item.pred_class}`, item])),
    [analysis.confusion_examples]
  );
  const maxCount = Math.max(1, ...analysis.top_confusions.map((item) => item.count));
  const selectedCount = countMap.get(selectedKey);
  const selectedPair = exampleMap.get(selectedKey) ?? analysis.confusion_examples[0];
  const groupSamples = analysis.error_groups[selectedGroup] ?? [];

  return (
    <section className="section failureSection" id="failures">
      <div className="sectionHeading findingHeading">
        <div>
          <h2>210 misses out of 23,980 — and they cluster</h2>
          <p>
            Honda → VW is the largest confusion pair at 25 images. Honda accounts for 65 misses; BMW for 52. Jeep has 6.
          </p>
        </div>
      </div>

      <div className="matrixMeta">
        <span>Predicted label</span>
        {selectedPair ? (
          <p aria-live="polite">
            Selected: <strong>{classLabel(selectedPair.true_class)} → {classLabel(selectedPair.pred_class)}</strong>
            <b>{selectedPair.count} misses</b>
            {selectedCount?.pct_of_true != null ? <em>{pct(selectedCount.pct_of_true, 2)} of true-label images</em> : null}
          </p>
        ) : null}
      </div>

      <div className="matrixScroller" role="region" aria-label="Confusion matrix. Scroll horizontally on small screens." tabIndex={0}>
        <div className="matrixGrid" style={{ gridTemplateColumns: `112px repeat(${classes.length}, minmax(76px, 1fr))` }}>
          <div className="matrixCorner">True label</div>
          {classes.map((predicted) => (
            <div className="matrixHeader" key={`header-${predicted}`} style={{ "--class-color": CLASS_COLORS[predicted] } as React.CSSProperties}>
              {classLabel(predicted)}
            </div>
          ))}
          {classes.map((actual) => (
            <div className="matrixRow" key={actual}>
              <div className="matrixHeader rowHeader" style={{ "--class-color": CLASS_COLORS[actual] } as React.CSSProperties}>
                {classLabel(actual)}
              </div>
              {classes.map((predicted) => {
                const key = `${actual}->${predicted}`;
                if (actual === predicted) {
                  return <div aria-label={`${classLabel(actual)} correctly predicted`} className="heatCell diagonal" key={key}>—</div>;
                }
                const countItem = countMap.get(key);
                const examples = exampleMap.get(key);
                const count = countItem?.count ?? 0;
                const share = countItem?.pct_of_true;
                const label = `${classLabel(actual)} predicted as ${classLabel(predicted)}, ${count} ${count === 1 ? "miss" : "misses"}${share != null ? `, ${pct(share, 2)} of true-label images` : ""}`;
                return (
                  <button
                    aria-label={label}
                    className={`heatCell ${count ? "filled" : "empty"} ${selectedKey === key ? "selected" : ""}`}
                    disabled={!examples}
                    key={key}
                    onClick={() => setSelectedKey(key)}
                    style={{ "--heat": count / maxCount } as React.CSSProperties}
                    title={label}
                    type="button"
                  >
                    {count || ""}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="scrollAffordance">Swipe sideways to inspect all eight predicted classes.</p>

      {selectedPair ? (
        <div className="errorGrid selectedExamples" aria-label="Examples from the selected confusion pair">
          {selectedPair.examples.slice(0, 4).map((sample) => <ErrorCard key={sample.id} sample={sample} />)}
        </div>
      ) : null}

      <div className="groupReview">
        <div className="segmented" aria-label="Error review group">
          {Object.keys(GROUP_COPY).map((key) => (
            <button
              aria-pressed={selectedGroup === key}
              className={selectedGroup === key ? "selected" : ""}
              key={key}
              onClick={() => setSelectedGroup(key)}
              type="button"
            >
              {GROUP_COPY[key].label}
            </button>
          ))}
        </div>
        <p className="groupFinding">{GROUP_COPY[selectedGroup]?.detail}</p>
        <div className="errorGrid">
          {groupSamples.slice(0, 4).map((sample) => <ErrorCard key={sample.id} sample={sample} />)}
        </div>
      </div>
    </section>
  );
}
