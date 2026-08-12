import { useMemo, useState } from "react";
import { Photo } from "../components/Photo";
import { CLASS_COLORS, classLabel, pct } from "../lib";
import type { AnalysisData, AnalysisSample } from "../types";

function ErrorCard({ sample }: { sample: AnalysisSample }) {
  const runnerUp = sample.probabilities[1];

  return (
    <article className="errorCard">
      <Photo alt={`${classLabel(sample.true_class)} validation example`} src={sample.image} />
      <div className="errorCardBody">
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

  return (
    <section className="section" id="failures">
      <div className="sectionHeading">
        <div>
          <h2>The 210 misses are not spread evenly</h2>
          <p>
            Honda and BMW account for 117 of them. Jeep accounts for six. Select any cell to see the photographs behind
            that confusion.
          </p>
        </div>
      </div>

      <div
        className="matrixScroller"
        role="region"
        aria-label="Confusion matrix. Rows are the known label, columns are the prediction."
        tabIndex={0}
      >
        <div className="matrixGrid" style={{ gridTemplateColumns: `112px repeat(${classes.length}, minmax(64px, 1fr))` }}>
          <div className="matrixCorner">True ╲ Pred</div>
          {classes.map((predicted) => (
            <div
              className="matrixHeader"
              key={`header-${predicted}`}
              style={{ "--class-color": CLASS_COLORS[predicted] } as React.CSSProperties}
            >
              {classLabel(predicted)}
            </div>
          ))}
          {classes.map((actual) => (
            <div className="matrixRow" key={actual}>
              <div
                className="matrixHeader rowHeader"
                style={{ "--class-color": CLASS_COLORS[actual] } as React.CSSProperties}
              >
                {classLabel(actual)}
              </div>
              {classes.map((predicted) => {
                const key = `${actual}->${predicted}`;
                if (actual === predicted) {
                  return (
                    <div
                      aria-label={`${classLabel(actual)} correctly predicted`}
                      className="heatCell diagonal"
                      key={key}
                    />
                  );
                }
                const countItem = countMap.get(key);
                const examples = exampleMap.get(key);
                const count = countItem?.count ?? 0;
                const share = countItem?.pct_of_true;
                const label = `${classLabel(actual)} predicted as ${classLabel(predicted)}, ${count} ${
                  count === 1 ? "miss" : "misses"
                }${share != null ? `, ${pct(share, 2)} of true-label images` : ""}`;
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
      <p className="scrollAffordance">Swipe sideways to reach all eight predicted classes.</p>

      {selectedPair ? (
        <>
          <p className="matrixCaption" aria-live="polite">
            <strong>
              {classLabel(selectedPair.true_class)} <span aria-hidden="true">&#8594;</span>{" "}
              {classLabel(selectedPair.pred_class)}
            </strong>
            <b>{selectedPair.count} misses</b>
            {selectedCount?.pct_of_true != null ? <span>{pct(selectedCount.pct_of_true, 2)} of all {classLabel(selectedPair.true_class)} images</span> : null}
          </p>
          <div className="errorGrid" aria-label="Examples from the selected confusion pair">
            {selectedPair.examples.slice(0, 4).map((sample) => (
              <ErrorCard key={sample.id} sample={sample} />
            ))}
          </div>
        </>
      ) : null}
    </section>
  );
}
