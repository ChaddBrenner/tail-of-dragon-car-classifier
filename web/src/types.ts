export type Probability = {
  class_name: string;
  probability: number;
};

export type Sample = {
  id: string;
  image: string;
  true_class: string;
  top_prediction: string;
  top_probability: number;
  probabilities: Probability[];
};

export type AnalysisSample = Sample & {
  margin?: number;
};

export type Metrics = {
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

export type TopConfusion = {
  true_class: string;
  pred_class: string;
  count: number;
  pct_of_true?: number;
};

export type ConfusionExample = TopConfusion & {
  examples: AnalysisSample[];
};

export type NoiseReview = {
  total: number;
  errors: number;
  accuracy: number;
  top_error_pairs: TopConfusion[];
  errors_by_class: Record<string, number>;
  error_confidence_buckets: Record<string, number>;
  low_conf_correct_count: number;
};

export type AnalysisData = {
  metrics: Metrics;
  top_confusions: TopConfusion[];
  confusion_examples: ConfusionExample[];
  error_groups: Record<string, AnalysisSample[]>;
  noise_review: NoiseReview;
};

export type AppData = {
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

export type Theme = "light" | "dark";
