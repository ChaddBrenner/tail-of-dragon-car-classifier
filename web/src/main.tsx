import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BarChart3, Gauge, ImageIcon } from "lucide-react";
import "./styles.css";

type Probability = {
  class_name: string;
  probability: number;
};

type Sample = {
  id: string;
  image: string;
  true_class: string;
  top_prediction: string;
  top_probability: number;
  probabilities: Probability[];
};

type AppData = {
  model: {
    name: string;
    version: string;
    classes: string[];
    best_epoch?: number;
    metrics: Record<string, number>;
  };
  samples: Sample[];
};

const fallbackData: AppData = {
  model: {
    name: "model pending",
    version: "local",
    classes: ["bmw", "corvette", "honda", "jeep", "miata", "mustang", "porsche", "vw"],
    metrics: { accuracy: 0, macro_f1: 0, top3_accuracy: 0 }
  },
  samples: []
};

function pct(value: number | undefined) {
  return `${Math.round((value ?? 0) * 1000) / 10}%`;
}

function App() {
  const [data, setData] = useState<AppData>(fallbackData);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/predictions.json")
      .then((res) => (res.ok ? res.json() : fallbackData))
      .then((payload) => {
        setData(payload);
        const typedPayload = payload as AppData;
        if (typedPayload.samples?.length) setSelectedId(typedPayload.samples[0].id);
      })
      .catch(() => setData(fallbackData));
  }, []);

  const selected = useMemo(
    () => data.samples.find((sample) => sample.id === selectedId) ?? data.samples[0],
    [data.samples, selectedId]
  );

  return (
    <main className="shell">
      <section className="header">
        <div>
          <p className="eyebrow">Tail of the Dragon</p>
          <h1>Car Classifier</h1>
        </div>
        <div className="metrics" aria-label="Model metrics">
          <div className="metric">
            <Gauge size={18} />
            <span>Accuracy</span>
            <strong>{pct(data.model.metrics.accuracy)}</strong>
          </div>
          <div className="metric">
            <BarChart3 size={18} />
            <span>Macro F1</span>
            <strong>{pct(data.model.metrics.macro_f1)}</strong>
          </div>
          <div className="metric">
            <ImageIcon size={18} />
            <span>Samples</span>
            <strong>{data.samples.length}</strong>
          </div>
        </div>
      </section>

      <section className="workspace">
        <aside className="gallery" aria-label="Curated gallery">
          {data.samples.map((sample) => (
            <button
              className={`thumb ${sample.id === selected?.id ? "active" : ""}`}
              key={sample.id}
              onClick={() => setSelectedId(sample.id)}
              type="button"
              title={`${sample.true_class} predicted as ${sample.top_prediction}`}
            >
              <img src={sample.image} alt={`${sample.true_class} car`} />
              <span>{sample.true_class}</span>
            </button>
          ))}
        </aside>

        <section className="detail" aria-live="polite">
          {selected ? (
            <>
              <div className="imagePane">
                <img src={selected.image} alt={`${selected.true_class} car`} />
              </div>
              <div className="resultPane">
                <div className="predictionHead">
                  <div>
                    <span className="label">Prediction</span>
                    <h2>{selected.top_prediction}</h2>
                  </div>
                  <strong>{pct(selected.top_probability)}</strong>
                </div>
                <div className="truth">
                  Known label: <span>{selected.true_class}</span>
                </div>
                <div className="bars">
                  {selected.probabilities.map((prob) => (
                    <div className="barRow" key={prob.class_name}>
                      <div className="barLabel">
                        <span>{prob.class_name}</span>
                        <strong>{pct(prob.probability)}</strong>
                      </div>
                      <div className="track">
                        <div className="fill" style={{ width: pct(prob.probability) }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="empty">Gallery predictions have not been generated yet.</div>
          )}
        </section>
      </section>

      <footer>
        <span>{data.model.name}</span>
        <span>Version {data.model.version}</span>
        <a href="/reports/classification_report.txt">Classification report</a>
        <a href="/reports/confusion_matrix.png">Confusion matrix</a>
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
