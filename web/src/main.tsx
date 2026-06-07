import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BarChart3, Gauge, ImageIcon, RefreshCw } from "lucide-react";
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
  gallery?: {
    display_count: number;
    pool_size: number;
    samples_per_class: number;
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

const DEFAULT_DISPLAY_COUNT = 48;

function pct(value: number | undefined) {
  return `${Math.round((value ?? 0) * 1000) / 10}%`;
}

function pickGalleryPage(samples: Sample[], classes: string[], displayCount: number, page: number) {
  if (samples.length <= displayCount) return samples;
  const perClass = Math.max(1, Math.floor(displayCount / Math.max(classes.length, 1)));
  const picked: Sample[] = [];
  for (const className of classes) {
    const group = samples.filter((sample) => sample.true_class === className);
    if (!group.length) continue;
    const offset = (page * perClass) % group.length;
    for (let i = 0; i < Math.min(perClass, group.length); i += 1) {
      picked.push(group[(offset + i) % group.length]);
    }
  }
  if (picked.length >= displayCount) return picked.slice(0, displayCount);
  const used = new Set(picked.map((sample) => sample.id));
  const remaining = samples.filter((sample) => !used.has(sample.id));
  return [...picked, ...remaining.slice(0, displayCount - picked.length)];
}

function App() {
  const [data, setData] = useState<AppData>(fallbackData);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [galleryPage, setGalleryPage] = useState(0);

  useEffect(() => {
    fetch("/data/predictions.json")
      .then((res) => (res.ok ? res.json() : fallbackData))
      .then((payload) => {
        setData(payload);
        setGalleryPage(0);
        const typedPayload = payload as AppData;
        if (typedPayload.samples?.length) setSelectedId(typedPayload.samples[0].id);
      })
      .catch(() => setData(fallbackData));
  }, []);

  const displayCount = data.gallery?.display_count ?? DEFAULT_DISPLAY_COUNT;
  const displayedSamples = useMemo(
    () => pickGalleryPage(data.samples, data.model.classes, displayCount, galleryPage),
    [data.samples, data.model.classes, displayCount, galleryPage]
  );

  const selected = useMemo(
    () => displayedSamples.find((sample) => sample.id === selectedId) ?? displayedSamples[0],
    [displayedSamples, selectedId]
  );

  useEffect(() => {
    if (displayedSamples.length) setSelectedId(displayedSamples[0].id);
  }, [displayedSamples]);

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
            <strong>{displayedSamples.length}/{data.samples.length}</strong>
          </div>
        </div>
      </section>

      <section className="workspace">
        <aside className="galleryPanel" aria-label="Curated gallery">
          <div className="galleryHead">
            <span>{data.samples.length} photos</span>
            <button
              className="refreshButton"
              type="button"
              onClick={() => setGalleryPage((page) => page + 1)}
              disabled={data.samples.length <= displayedSamples.length}
            >
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
          <div className="gallery">
            {displayedSamples.map((sample) => (
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
          </div>
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
