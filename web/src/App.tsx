import { useEffect, useMemo, useState } from "react";
import { GithubIcon } from "./components/Icons";
import { ThemeToggle } from "./components/ThemeToggle";
import { CLASS_NAMES, formatNumber, pickGalleryPage } from "./lib";
import { DataStory } from "./sections/DataStory";
import { Explorer } from "./sections/Explorer";
import { FailureAnalysis } from "./sections/FailureAnalysis";
import { Hero } from "./sections/Hero";
import type { AnalysisData, AppData, Metrics } from "./types";

const DEFAULT_DISPLAY_COUNT = 16;
const sectionLinks = [
  { id: "explore", label: "Explore" },
  { id: "failures", label: "Where it fails" },
  { id: "data", label: "Why I stopped" }
];

const verifiedMetrics: Metrics = {
  accuracy: 0.9912427022518766,
  macro_f1: 0.9912341247865801,
  top3_accuracy: 0.9977481234361968,
  evaluated_samples: 23980,
  error_count: 210,
  class_names: CLASS_NAMES
};

export function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [galleryPage, setGalleryPage] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState("explore");

  useEffect(() => {
    const controller = new AbortController();
    const options = { signal: controller.signal };

    Promise.all([
      fetch("/data/predictions.json", options).then((response) => {
        if (!response.ok) throw new Error("Predictions failed to load");
        return response.json() as Promise<AppData>;
      }),
      fetch("/data/analysis.json", options).then((response) => {
        if (!response.ok) throw new Error("Analysis failed to load");
        return response.json() as Promise<AnalysisData>;
      })
    ])
      .then(([predictions, analysisPayload]) => {
        setData(predictions);
        setAnalysis(analysisPayload);
        setSelectedId(predictions.samples[0]?.id ?? null);
        requestAnimationFrame(() => {
          const hashTarget = document.getElementById(window.location.hash.slice(1));
          hashTarget?.scrollIntoView();
        });
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setLoadError(true);
      });

    return () => controller.abort();
  }, []);

  useEffect(() => {
    const sections = sectionLinks
      .map(({ id }) => document.getElementById(id))
      .filter((element): element is HTMLElement => Boolean(element));
    if (!sections.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActiveSection(visible.target.id);
      },
      { rootMargin: "-24% 0px -62% 0px", threshold: [0, 0.1, 0.4] }
    );
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [analysis]);

  const classes = analysis?.metrics.class_names ?? data?.model.classes ?? CLASS_NAMES;
  const displayCount = DEFAULT_DISPLAY_COUNT;
  const displayedSamples = useMemo(
    () => pickGalleryPage(data?.samples ?? [], classes, displayCount, galleryPage),
    [classes, data?.samples, displayCount, galleryPage]
  );
  const selected = useMemo(
    () => displayedSamples.find((sample) => sample.id === selectedId) ?? displayedSamples[0],
    [displayedSamples, selectedId]
  );

  const metrics = analysis?.metrics ?? verifiedMetrics;

  function shuffleGallery() {
    setGalleryPage((page) => page + 1);
    setSelectedId(null);
  }

  return (
    <div className="siteShell">
      <header className="siteHeader">
        <a className="brand" href="#top" aria-label="Tail of the Dragon Classifier home">
          <i aria-hidden="true" />
          <span>Tail of the Dragon Classifier</span>
        </a>
        <div className="headerActions">
          <ThemeToggle />
          <a className="githubLink" href="https://github.com/ChaddBrenner/tail-of-dragon-car-classifier">
            <GithubIcon />
            <span>GitHub</span>
          </a>
        </div>
      </header>

      <main>
        <Hero classes={classes} metrics={metrics} samples={data?.samples ?? []} />

        <nav className="sectionNav" aria-label="On this page">
          <div>
            {sectionLinks.map((link) => (
              <a
                aria-current={activeSection === link.id ? "location" : undefined}
                className={activeSection === link.id ? "active" : ""}
                href={`#${link.id}`}
                key={link.id}
              >
                {link.label}
              </a>
            ))}
          </div>
        </nav>

        {loadError ? (
          <div className="loadError" role="alert">
            The static analysis files did not load. Refresh the page to try again.
          </div>
        ) : null}

        <Explorer
          onSelect={setSelectedId}
          onShuffle={shuffleGallery}
          samples={displayedSamples}
          selected={selected}
          totalSamples={data?.samples.length ?? 0}
        />

        {analysis ? (
          <>
            <FailureAnalysis analysis={analysis} classes={classes} />
            <DataStory review={analysis.noise_review} />
          </>
        ) : (
          <section className="section loadingSection" aria-label="Analysis is loading">
            <div className="headingSkeleton" />
            <div className="matrixSkeleton" />
          </section>
        )}
      </main>

      <footer className="siteFooter">
        <div>
          <strong>ConvNeXt-Tiny</strong>
          <span>288px</span>
          <span>best epoch {data?.model.best_epoch ?? 10}</span>
          <span>macro F1 {(metrics.macro_f1 ?? 0).toFixed(4)}</span>
          <span>top-3 {((metrics.top3_accuracy ?? 0) * 100).toFixed(2)}%</span>
          <span>{formatNumber(metrics.evaluated_samples)} validation images</span>
        </div>
        <nav aria-label="Project artifacts">
          <a href="/reports/classification_report.txt">Classification report</a>
          <a href="/reports/confusion_matrix.png">Confusion matrix</a>
          <a href="https://github.com/ChaddBrenner/tail-of-dragon-car-classifier">GitHub</a>
          <a href="https://www.chadd.blog/posts/car-type-detection/">Project story</a>
        </nav>
        <p>The validation previews are watermarked photographs from killboy.com. The original photographers keep their rights.</p>
      </footer>
    </div>
  );
}
