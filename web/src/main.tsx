import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  BarChart3,
  Brain,
  FileText,
  Gauge,
  Grid2X2,
  ImageIcon,
  Microscope,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  TriangleAlert,
  Zap
} from "lucide-react";
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

type Metrics = {
  name?: string;
  accuracy?: number;
  macro_f1?: number;
  balanced_accuracy?: number;
  top3_accuracy?: number;
  evaluated_samples?: number;
  error_count?: number;
  class_names?: string[];
  errors_by_class?: Record<string, number>;
};

type AnalysisSample = Sample & {
  margin?: number;
  original?: string;
  heatmap?: string;
  explanation?: string;
};

type TopConfusion = {
  true_class: string;
  pred_class: string;
  count: number;
  pct_of_true?: number;
};

type ConfusionExample = TopConfusion & {
  examples: AnalysisSample[];
};

type NoiseReview = {
  total: number;
  errors: number;
  accuracy: number;
  top_error_pairs: TopConfusion[];
  errors_by_class: Record<string, number>;
  error_confidence_buckets: Record<string, number>;
  low_conf_correct_count: number;
};

type CaseStudy = {
  baseline_accuracy: number;
  baseline_macro_f1: number;
  champion_accuracy: number;
  champion_macro_f1: number;
  practical_accuracy_ceiling: string;
  dataset_size: number;
  validation_size: number;
  search_models: string[];
};

type BrowserInferenceConfig = {
  enabled: boolean;
  model_path: string;
  external_data_path: string;
  runtime: string;
  input_size: number;
  mean: number[];
  std: number[];
  note: string;
};

type ModelCard = {
  architecture: string;
  input_size: number;
  intended_use: string;
  out_of_scope: string;
  limitations: string[];
  classes: string[];
};

type SupplyChain = {
  image: string;
  attestation: string;
};

type AnalysisData = {
  metrics: Metrics;
  top_confusions: TopConfusion[];
  confusion_examples: ConfusionExample[];
  error_groups: Record<string, AnalysisSample[]>;
  gradcam: AnalysisSample[];
  noise_review: NoiseReview;
  case_study: CaseStudy;
  model_card: ModelCard;
  browser_inference: BrowserInferenceConfig;
  supply_chain: SupplyChain;
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

type ViewKey =
  | "gallery"
  | "errors"
  | "confusions"
  | "explain"
  | "case"
  | "quality"
  | "inference"
  | "model"
  | "supply";

type BrowserResult = {
  sampleId: string;
  elapsedMs: number;
  top_prediction: string;
  top_probability: number;
  probabilities: Probability[];
};

type OrtModule = typeof import("onnxruntime-web");

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

const navItems: { key: ViewKey; label: string; icon: LucideIcon }[] = [
  { key: "gallery", label: "Gallery", icon: Grid2X2 },
  { key: "errors", label: "Errors", icon: TriangleAlert },
  { key: "confusions", label: "Confusions", icon: Target },
  { key: "explain", label: "Grad-CAM", icon: Microscope },
  { key: "case", label: "Case Study", icon: FileText },
  { key: "quality", label: "Quality", icon: Activity },
  { key: "inference", label: "Browser ONNX", icon: Zap },
  { key: "model", label: "Model Card", icon: Brain },
  { key: "supply", label: "Supply Chain", icon: ShieldCheck }
];

const groupCopy: Record<string, { label: string; detail: string }> = {
  high_confidence_misses: {
    label: "High confidence misses",
    detail: "Likely label noise, very ambiguous angles, or visually similar vehicle families."
  },
  low_margin_misses: {
    label: "Low margin misses",
    detail: "The model is unsure and the top classes are close together."
  },
  random_misses: {
    label: "Random misses",
    detail: "A representative slice of remaining validation errors."
  },
  low_confidence_correct: {
    label: "Low confidence correct",
    detail: "Correct predictions where the model showed useful uncertainty."
  },
  random_correct: {
    label: "Random correct",
    detail: "Representative correct predictions for baseline sanity checking."
  }
};

let cachedOrtSession: Promise<{ ort: OrtModule; session: any }> | null = null;

function pct(value: number | undefined, digits = 1) {
  return `${((value ?? 0) * 100).toFixed(digits)}%`;
}

function formatNumber(value: number | undefined) {
  return (value ?? 0).toLocaleString();
}

function title(value: string) {
  return value.replaceAll("_", " ");
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

function softmax(values: number[]) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const denom = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / denom);
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to load ${src}`));
    image.src = src;
  });
}

async function imageToTensor(ort: OrtModule, imagePath: string, config: BrowserInferenceConfig) {
  const image = await loadImage(imagePath);
  const size = config.input_size;
  const resizeShort = Math.round(size * 1.15);
  const scale = resizeShort / Math.min(image.naturalWidth, image.naturalHeight);
  const cropWidth = size / scale;
  const cropHeight = size / scale;
  const sourceX = Math.max(0, (image.naturalWidth - cropWidth) / 2);
  const sourceY = Math.max(0, (image.naturalHeight - cropHeight) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas is unavailable in this browser.");
  ctx.drawImage(image, sourceX, sourceY, cropWidth, cropHeight, 0, 0, size, size);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const area = size * size;
  const data = new Float32Array(area * 3);
  for (let i = 0; i < area; i += 1) {
    const pixel = i * 4;
    data[i] = (pixels[pixel] / 255 - config.mean[0]) / config.std[0];
    data[area + i] = (pixels[pixel + 1] / 255 - config.mean[1]) / config.std[1];
    data[area * 2 + i] = (pixels[pixel + 2] / 255 - config.mean[2]) / config.std[2];
  }
  return new ort.Tensor("float32", data, [1, 3, size, size]);
}

async function getOrtSession(config: BrowserInferenceConfig) {
  if (!cachedOrtSession) {
    cachedOrtSession = (async () => {
      const ort = await import("onnxruntime-web");
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      const externalName = config.external_data_path.split("/").pop() ?? "champion.onnx.data";
      try {
        const session = await ort.InferenceSession.create(config.model_path, {
          executionProviders: ["wasm"],
          externalData: [{ path: externalName, data: config.external_data_path }]
        });
        return { ort, session };
      } catch {
        const session = await ort.InferenceSession.create(config.model_path, {
          executionProviders: ["wasm"],
          externalData: [externalName]
        });
        return { ort, session };
      }
    })();
  }
  return cachedOrtSession;
}

async function runOnnx(sample: Sample, config: BrowserInferenceConfig, classes: string[]) {
  const { ort, session } = await getOrtSession(config);
  const input = await imageToTensor(ort, sample.image, config);
  const inputName = session.inputNames?.[0] ?? "input";
  const outputName = session.outputNames?.[0];
  const outputs = await session.run({ [inputName]: input });
  const output = outputName ? outputs[outputName] : Object.values(outputs)[0];
  if (!output) throw new Error("The ONNX model did not return an output tensor.");
  const logits = Array.from(output.data as Float32Array, Number);
  const probabilities = softmax(logits)
    .map((probability, index) => ({
      class_name: classes[index] ?? `class_${index}`,
      probability
    }))
    .sort((a, b) => b.probability - a.probability);
  return {
    top_prediction: probabilities[0]?.class_name ?? "unknown",
    top_probability: probabilities[0]?.probability ?? 0,
    probabilities
  };
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="metric">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}

function ProbabilityBars({ probabilities }: { probabilities: Probability[] }) {
  return (
    <div className="bars">
      {probabilities.map((prob) => (
        <div className="barRow" key={prob.class_name}>
          <div className="barLabel">
            <span>{prob.class_name}</span>
            <strong>{pct(prob.probability, 1)}</strong>
          </div>
          <div className="track">
            <div className="fill" style={{ width: pct(prob.probability, 3) }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function SampleMiniCard({
  sample,
  onSelect,
  active = false
}: {
  sample: Sample;
  onSelect?: () => void;
  active?: boolean;
}) {
  const body = (
    <>
      <img src={sample.image} alt={`${sample.true_class} car`} />
      <span>{sample.true_class}</span>
    </>
  );
  if (!onSelect) return <div className="thumb readonly">{body}</div>;
  return (
    <button
      className={`thumb ${active ? "active" : ""}`}
      onClick={onSelect}
      type="button"
      title={`${sample.true_class} predicted as ${sample.top_prediction}`}
    >
      {body}
    </button>
  );
}

function ResultCard({ sample, label = "Prediction" }: { sample: AnalysisSample | Sample; label?: string }) {
  return (
    <article className={`sampleCard ${sample.true_class === sample.top_prediction ? "correct" : "miss"}`}>
      <img src={sample.image} alt={`${sample.true_class} validation example`} />
      <div className="sampleBody">
        <div className="predictionHead compact">
          <div>
            <span className="label">{label}</span>
            <h3>{sample.top_prediction}</h3>
          </div>
          <strong>{pct(sample.top_probability, 1)}</strong>
        </div>
        <div className="truth">
          Known label: <span>{sample.true_class}</span>
          {"margin" in sample && typeof sample.margin === "number" ? (
            <em>Margin {pct(sample.margin, 1)}</em>
          ) : null}
        </div>
        <ProbabilityBars probabilities={sample.probabilities} />
      </div>
    </article>
  );
}

function App() {
  const [data, setData] = useState<AppData>(fallbackData);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [activeView, setActiveView] = useState<ViewKey>("gallery");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [galleryPage, setGalleryPage] = useState(0);
  const [selectedErrorGroup, setSelectedErrorGroup] = useState("high_confidence_misses");
  const [selectedConfusion, setSelectedConfusion] = useState<string | null>(null);
  const [browserStatus, setBrowserStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [browserMessage, setBrowserMessage] = useState("");
  const [browserResult, setBrowserResult] = useState<BrowserResult | null>(null);

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

    fetch("/data/analysis.json")
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => setAnalysis(payload))
      .catch(() => setAnalysis(null));
  }, []);

  const classes = analysis?.metrics.class_names ?? data.model.classes;
  const displayCount = data.gallery?.display_count ?? DEFAULT_DISPLAY_COUNT;
  const displayedSamples = useMemo(
    () => pickGalleryPage(data.samples, classes, displayCount, galleryPage),
    [data.samples, classes, displayCount, galleryPage]
  );

  const selected = useMemo(
    () => displayedSamples.find((sample) => sample.id === selectedId) ?? displayedSamples[0],
    [displayedSamples, selectedId]
  );

  useEffect(() => {
    if (displayedSamples.length) setSelectedId(displayedSamples[0].id);
  }, [displayedSamples]);

  useEffect(() => {
    if (!selectedConfusion && analysis?.confusion_examples?.length) {
      const first = analysis.confusion_examples[0];
      setSelectedConfusion(`${first.true_class}->${first.pred_class}`);
    }
  }, [analysis, selectedConfusion]);

  async function handleBrowserRun(sample: Sample | undefined) {
    if (!sample || !analysis?.browser_inference?.enabled) return;
    setBrowserStatus("loading");
    setBrowserMessage("Loading ONNX Runtime Web and model weights.");
    const start = performance.now();
    try {
      const result = await runOnnx(sample, analysis.browser_inference, classes);
      setBrowserResult({ sampleId: sample.id, elapsedMs: performance.now() - start, ...result });
      setBrowserStatus("done");
      setBrowserMessage("Ran the exported ONNX model locally in this browser.");
    } catch (error) {
      setBrowserStatus("error");
      setBrowserMessage(error instanceof Error ? error.message : "Browser inference failed.");
    }
  }

  const metrics = analysis?.metrics ?? {
    accuracy: data.model.metrics.accuracy,
    macro_f1: data.model.metrics.macro_f1,
    top3_accuracy: data.model.metrics.top3_accuracy,
    evaluated_samples: data.samples.length
  };

  function renderGallery() {
    return (
      <section className="workspace">
        <aside className="galleryPanel" aria-label="Curated gallery">
          <div className="galleryHead">
            <span>
              {displayedSamples.length}/{data.samples.length} photos
            </span>
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
              <SampleMiniCard
                active={sample.id === selected?.id}
                key={sample.id}
                onSelect={() => setSelectedId(sample.id)}
                sample={sample}
              />
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
                  <strong>{pct(selected.top_probability, 1)}</strong>
                </div>
                <div className="truth">
                  Known label: <span>{selected.true_class}</span>
                </div>
                <ProbabilityBars probabilities={selected.probabilities} />
              </div>
            </>
          ) : (
            <div className="empty">Gallery predictions have not been generated yet.</div>
          )}
        </section>
      </section>
    );
  }

  function renderErrors() {
    if (!analysis) return <LoadingPanel />;
    const group = analysis.error_groups[selectedErrorGroup] ?? [];
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Interactive review</span>
            <h2>Error Analysis</h2>
          </div>
          <p>
            Manual review focused on whether the last misses are meaningful model errors or noisy validation
            examples.
          </p>
        </div>
        <div className="pillGrid">
          {Object.entries(analysis.error_groups).map(([key, items]) => (
            <button
              className={`pill ${selectedErrorGroup === key ? "active" : ""}`}
              key={key}
              onClick={() => setSelectedErrorGroup(key)}
              type="button"
            >
              <strong>{groupCopy[key]?.label ?? title(key)}</strong>
              <span>{items.length} examples</span>
            </button>
          ))}
        </div>
        <div className="insightBand">
          <TriangleAlert size={19} />
          <span>{groupCopy[selectedErrorGroup]?.detail ?? "Curated validation examples."}</span>
        </div>
        <div className="sampleGrid">
          {group.map((sample) => (
            <ResultCard key={sample.id} sample={sample} />
          ))}
        </div>
      </section>
    );
  }

  function renderConfusions() {
    if (!analysis) return <LoadingPanel />;
    const maxCount = Math.max(1, ...analysis.top_confusions.map((item) => item.count));
    const pairMap = new Map(analysis.confusion_examples.map((item) => [`${item.true_class}->${item.pred_class}`, item]));
    const selectedPair = selectedConfusion ? pairMap.get(selectedConfusion) : analysis.confusion_examples[0];
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Top off-diagonal cells</span>
            <h2>Clickable Confusion Matrix</h2>
          </div>
          <p>Click a colored cell to inspect the actual images behind that validation confusion.</p>
        </div>
        <div className="matrixWrap">
          <div className="matrixGrid" style={{ gridTemplateColumns: `110px repeat(${classes.length}, minmax(72px, 1fr))` }}>
            <div className="matrixCorner">True / Pred</div>
            {classes.map((pred) => (
              <div className="matrixHeader" key={`pred-${pred}`}>
                {pred}
              </div>
            ))}
            {classes.map((actual) => (
              <React.Fragment key={actual}>
                <div className="matrixHeader rowHeader">{actual}</div>
                {classes.map((pred) => {
                  const key = `${actual}->${pred}`;
                  const item = pairMap.get(key);
                  const intensity = item ? 0.16 + (item.count / maxCount) * 0.84 : 0;
                  return (
                    <button
                      className={`heatCell ${item ? "filled" : ""} ${selectedConfusion === key ? "active" : ""}`}
                      data-testid={`confusion-${actual}-to-${pred}`}
                      disabled={!item || actual === pred}
                      key={key}
                      onClick={() => setSelectedConfusion(key)}
                      style={{ "--intensity": intensity } as React.CSSProperties}
                      type="button"
                    >
                      {actual === pred ? "match" : item?.count ?? ""}
                    </button>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
        {selectedPair ? (
          <>
            <div className="insightBand">
              <Target size={19} />
              <span>
                {selectedPair.true_class} labeled images predicted as {selectedPair.pred_class}:{" "}
                {selectedPair.count} validation misses.
              </span>
            </div>
            <div className="sampleGrid compactGrid">
              {selectedPair.examples.map((sample) => (
                <ResultCard key={sample.id} sample={sample} />
              ))}
            </div>
          </>
        ) : null}
      </section>
    );
  }

  function renderExplainability() {
    if (!analysis) return <LoadingPanel />;
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Model explainability</span>
            <h2>Grad-CAM Review</h2>
          </div>
          <p>Warm overlays show regions that most influenced the predicted class score for selected validation samples.</p>
        </div>
        <div className="gradGrid">
          {analysis.gradcam.map((sample) => (
            <article className="gradCard" key={sample.id}>
              <div className="compareImages">
                <img src={sample.original ?? sample.image} alt={`${sample.true_class} original`} />
                <img src={sample.heatmap ?? sample.image} alt={`${sample.top_prediction} Grad-CAM heatmap`} />
              </div>
              <div className="sampleBody">
                <div className="predictionHead compact">
                  <div>
                    <span className="label">{sample.true_class === sample.top_prediction ? "Correct" : "Miss"}</span>
                    <h3>{sample.top_prediction}</h3>
                  </div>
                  <strong>{pct(sample.top_probability, 1)}</strong>
                </div>
                <div className="truth">
                  Known label: <span>{sample.true_class}</span>
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>
    );
  }

  function renderCaseStudy() {
    if (!analysis) return <LoadingPanel />;
    const study = analysis.case_study;
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Technical narrative</span>
            <h2>Case Study</h2>
          </div>
          <p>
            The project preserves the inherited validation split, removes duplicate leakage, then compares modern
            transfer-learning candidates against the old checkpoint.
          </p>
        </div>
        <div className="statGrid">
          <MetricCard icon={Gauge} label="Old accuracy" value={pct(study.baseline_accuracy, 2)} />
          <MetricCard icon={Sparkles} label="Champion accuracy" value={pct(study.champion_accuracy, 2)} />
          <MetricCard icon={BarChart3} label="Champion macro F1" value={study.champion_macro_f1.toFixed(4)} />
          <MetricCard icon={Target} label="Practical ceiling" value={study.practical_accuracy_ceiling} />
        </div>
        <div className="caseColumns">
          <article className="panel">
            <h3>What changed</h3>
            <ul className="cleanList">
              <li>Preserved the original validation split for direct comparison.</li>
              <li>Removed duplicate filename leakage before training.</li>
              <li>Used AMP, cosine LR, label smoothing, mixup, CutMix, RandAugment, random erasing, TTA, and checkpointed metrics.</li>
              <li>Selected the deploy model by full-validation accuracy and macro F1.</li>
            </ul>
          </article>
          <article className="panel">
            <h3>Search space</h3>
            <div className="chipWrap">
              {study.search_models.map((model) => (
                <span className="chip" key={model}>
                  {model}
                </span>
              ))}
            </div>
            <dl className="definitionGrid">
              <div>
                <dt>Dataset</dt>
                <dd>{formatNumber(study.dataset_size)} photos</dd>
              </div>
              <div>
                <dt>Validation</dt>
                <dd>{formatNumber(study.validation_size)} photos</dd>
              </div>
            </dl>
          </article>
        </div>
      </section>
    );
  }

  function renderQuality() {
    if (!analysis) return <LoadingPanel />;
    const review = analysis.noise_review;
    const maxClassErrors = Math.max(1, ...Object.values(review.errors_by_class));
    const maxBucket = Math.max(1, ...Object.values(review.error_confidence_buckets));
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Dataset audit</span>
            <h2>Quality Dashboard</h2>
          </div>
          <p>
            The final 210 misses were reviewed as a stopping-rule check because many are mislabeled, unreadable,
            occluded, distant, or multi-vehicle images.
          </p>
        </div>
        <div className="statGrid">
          <MetricCard icon={ImageIcon} label="Validation images" value={formatNumber(review.total)} />
          <MetricCard icon={TriangleAlert} label="Final misses" value={formatNumber(review.errors)} />
          <MetricCard icon={Gauge} label="Validation accuracy" value={pct(review.accuracy, 2)} />
          <MetricCard icon={Activity} label="Low-conf correct" value={formatNumber(review.low_conf_correct_count)} />
        </div>
        <div className="caseColumns">
          <article className="panel">
            <h3>Errors by known label</h3>
            <div className="wideBars">
              {Object.entries(review.errors_by_class).map(([className, count]) => (
                <div className="wideBarRow" key={className}>
                  <span>{className}</span>
                  <div className="track">
                    <div className="fill teal" style={{ width: `${(count / maxClassErrors) * 100}%` }} />
                  </div>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          </article>
          <article className="panel">
            <h3>Error confidence buckets</h3>
            <div className="wideBars">
              {Object.entries(review.error_confidence_buckets).map(([bucket, count]) => (
                <div className="wideBarRow" key={bucket}>
                  <span>{title(bucket)}</span>
                  <div className="track">
                    <div className="fill amber" style={{ width: `${(count / maxBucket) * 100}%` }} />
                  </div>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          </article>
        </div>
      </section>
    );
  }

  function renderInference() {
    if (!analysis) return <LoadingPanel />;
    const resultForSelected = browserResult?.sampleId === selected?.id ? browserResult : null;
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Curated-only inference</span>
            <h2>Browser ONNX Demo</h2>
          </div>
          <p>{analysis.browser_inference.note}</p>
        </div>
        <div className="inferenceLayout">
          <aside className="panel inferencePicker">
            <div className="galleryHead">
              <span>Choose a curated validation photo</span>
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
            <div className="miniGallery">
              {displayedSamples.slice(0, 24).map((sample) => (
                <SampleMiniCard
                  active={sample.id === selected?.id}
                  key={sample.id}
                  onSelect={() => {
                    setSelectedId(sample.id);
                    setBrowserResult(null);
                    setBrowserStatus("idle");
                    setBrowserMessage("");
                  }}
                  sample={sample}
                />
              ))}
            </div>
          </aside>
          <section className="panel inferencePanel">
            {selected ? (
              <>
                <img className="inferenceImage" src={selected.image} alt={`${selected.true_class} inference sample`} />
                <div className="inferenceActions">
                  <button
                    className="primaryButton"
                    data-testid="run-onnx"
                    disabled={browserStatus === "loading"}
                    onClick={() => handleBrowserRun(selected)}
                    type="button"
                  >
                    <Zap size={17} />
                    {browserStatus === "loading" ? "Running..." : "Run ONNX"}
                  </button>
                  <span className={`statusText ${browserStatus}`}>{browserMessage || "Model runs locally after click."}</span>
                </div>
                <div className="inferenceCompare">
                  <article>
                    <span className="label">Static export</span>
                    <h3>{selected.top_prediction}</h3>
                    <ProbabilityBars probabilities={selected.probabilities} />
                  </article>
                  <article>
                    <span className="label">
                      Live browser result {resultForSelected ? `in ${Math.round(resultForSelected.elapsedMs)} ms` : ""}
                    </span>
                    <h3>{resultForSelected?.top_prediction ?? "not run"}</h3>
                    {resultForSelected ? <ProbabilityBars probabilities={resultForSelected.probabilities} /> : null}
                  </article>
                </div>
              </>
            ) : (
              <div className="empty">No curated sample is selected.</div>
            )}
          </section>
        </div>
      </section>
    );
  }

  function renderModelCard() {
    if (!analysis) return <LoadingPanel />;
    const card = analysis.model_card;
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Responsible ML summary</span>
            <h2>Model Card</h2>
          </div>
          <p>
            A concise, public-facing record of intended use, out-of-scope use, evaluation, and known limitations.
          </p>
        </div>
        <div className="caseColumns">
          <article className="panel">
            <h3>Model details</h3>
            <dl className="definitionGrid">
              <div>
                <dt>Architecture</dt>
                <dd>{card.architecture}</dd>
              </div>
              <div>
                <dt>Input size</dt>
                <dd>{card.input_size}px</dd>
              </div>
              <div>
                <dt>Accuracy</dt>
                <dd>{pct(metrics.accuracy, 2)}</dd>
              </div>
              <div>
                <dt>Macro F1</dt>
                <dd>{(metrics.macro_f1 ?? 0).toFixed(4)}</dd>
              </div>
            </dl>
          </article>
          <article className="panel">
            <h3>Use boundaries</h3>
            <p className="bodyText">{card.intended_use}</p>
            <p className="bodyText">
              <strong>Out of scope:</strong> {card.out_of_scope}
            </p>
          </article>
        </div>
        <article className="panel">
          <h3>Limitations</h3>
          <ul className="cleanList twoColumn">
            {card.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <div className="chipWrap">
            {card.classes.map((className) => (
              <span className="chip" key={className}>
                {className}
              </span>
            ))}
          </div>
        </article>
      </section>
    );
  }

  function renderSupply() {
    if (!analysis) return <LoadingPanel />;
    return (
      <section className="viewStack">
        <div className="sectionHead">
          <div>
            <span className="label">Deployment hygiene</span>
            <h2>Supply Chain</h2>
          </div>
          <p>CI builds the static site into a Caddy container image and publishes it to GHCR with OCI metadata.</p>
        </div>
        <div className="caseColumns">
          <article className="panel">
            <h3>Image</h3>
            <p className="codeLine">{analysis.supply_chain.image}</p>
            <ul className="cleanList">
              <li>GitHub Actions checks out Git LFS model assets.</li>
              <li>Vite builds the React application.</li>
              <li>Docker Buildx publishes multi-stage Caddy runtime image.</li>
            </ul>
          </article>
          <article className="panel">
            <h3>Attestations</h3>
            <p className="bodyText">{analysis.supply_chain.attestation}</p>
            <ul className="cleanList">
              <li>Build provenance is enabled in the publish workflow.</li>
              <li>SBOM generation is enabled for the container artifact.</li>
              <li>The deployment compose file pulls the immutable GHCR image.</li>
            </ul>
          </article>
        </div>
      </section>
    );
  }

  function renderActiveView() {
    switch (activeView) {
      case "errors":
        return renderErrors();
      case "confusions":
        return renderConfusions();
      case "explain":
        return renderExplainability();
      case "case":
        return renderCaseStudy();
      case "quality":
        return renderQuality();
      case "inference":
        return renderInference();
      case "model":
        return renderModelCard();
      case "supply":
        return renderSupply();
      case "gallery":
      default:
        return renderGallery();
    }
  }

  return (
    <main className="shell">
      <section className="header">
        <div>
          <p className="eyebrow">Tail of the Dragon</p>
          <h1>Car Classifier</h1>
        </div>
        <div className="metrics" aria-label="Model metrics">
          <MetricCard icon={Gauge} label="Accuracy" value={pct(metrics.accuracy, 2)} />
          <MetricCard icon={BarChart3} label="Macro F1" value={(metrics.macro_f1 ?? 0).toFixed(4)} />
          <MetricCard
            icon={ImageIcon}
            label="Validation"
            value={formatNumber(metrics.evaluated_samples ?? data.samples.length)}
            detail={`${formatNumber(metrics.error_count)} misses`}
          />
        </div>
      </section>

      <nav className="viewNav" aria-label="Portfolio views">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              className={activeView === item.key ? "active" : ""}
              data-testid={`view-${item.key}`}
              key={item.key}
              onClick={() => setActiveView(item.key)}
              type="button"
            >
              <Icon size={16} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {renderActiveView()}

      <footer>
        <span>{data.model.name}</span>
        <span>Version {data.model.version}</span>
        <a href="/reports/classification_report.txt">Classification report</a>
        <a href="/reports/confusion_matrix.png">Confusion matrix</a>
        <a href="/reports/MODEL_CARD.md">Model card markdown</a>
      </footer>
    </main>
  );
}

function LoadingPanel() {
  return <div className="empty tall">Analysis assets are loading.</div>;
}

createRoot(document.getElementById("root")!).render(<App />);
