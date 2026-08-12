import type { Sample } from "./types";

export const CLASS_NAMES = ["bmw", "corvette", "honda", "jeep", "miata", "mustang", "porsche", "vw"];

export const CLASS_LABELS: Record<string, string> = {
  bmw: "BMW",
  corvette: "Corvette",
  honda: "Honda",
  jeep: "Jeep",
  miata: "Miata",
  mustang: "Mustang",
  porsche: "Porsche",
  vw: "VW"
};

export const CLASS_COLORS: Record<string, string> = {
  bmw: "#6F9DE5",
  corvette: "#D86F62",
  honda: "#DCA147",
  jeep: "#8DA16B",
  miata: "#B47AC5",
  mustang: "#D87A56",
  porsche: "#54A4A7",
  vw: "#8994D8"
};

export function classLabel(value: string) {
  return CLASS_LABELS[value] ?? value.replaceAll("_", " ");
}

export function pct(value: number | undefined, digits = 1) {
  return `${((value ?? 0) * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | undefined) {
  return (value ?? 0).toLocaleString();
}

export function pickGalleryPage(samples: Sample[], classes: string[], displayCount: number, page: number) {
  if (samples.length <= displayCount) return samples;
  const perClass = Math.max(1, Math.floor(displayCount / Math.max(classes.length, 1)));
  const picked: Sample[] = [];

  for (const className of classes) {
    const group = samples.filter((sample) => sample.true_class === className);
    if (!group.length) continue;
    const offset = (page * perClass) % group.length;
    for (let index = 0; index < Math.min(perClass, group.length); index += 1) {
      picked.push(group[(offset + index) % group.length]);
    }
  }

  if (picked.length >= displayCount) return picked.slice(0, displayCount);
  const used = new Set(picked.map((sample) => sample.id));
  const remaining = samples.filter((sample) => !used.has(sample.id));
  return [...picked, ...remaining.slice(0, displayCount - picked.length)];
}
