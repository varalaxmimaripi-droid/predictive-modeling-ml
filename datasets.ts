export type Dataset = {
  name: string;
  features: number[][];
  labels: number[]; // 0 / 1
  featureNames: string[];
  classNames: [string, string];
  description: string;
};

// Seeded RNG so results are reproducible
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number) {
  // Box-Muller
  const u = Math.max(1e-9, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function makeMoons(n = 300, noise = 0.25, seed = 7): Dataset {
  const rng = mulberry32(seed);
  const features: number[][] = [];
  const labels: number[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = Math.PI * (i / half);
    features.push([Math.cos(t) + gauss(rng) * noise, Math.sin(t) + gauss(rng) * noise]);
    labels.push(0);
  }
  for (let i = 0; i < n - half; i++) {
    const t = Math.PI * (i / (n - half));
    features.push([1 - Math.cos(t) + gauss(rng) * noise, 0.5 - Math.sin(t) + gauss(rng) * noise]);
    labels.push(1);
  }
  return {
    name: "Two Moons",
    features,
    labels,
    featureNames: ["x₁", "x₂"],
    classNames: ["Moon A", "Moon B"],
    description: "Synthetic non-linear binary classification — classic ML benchmark.",
  };
}

export function makeBlobs(n = 300, seed = 11): Dataset {
  const rng = mulberry32(seed);
  const features: number[][] = [];
  const labels: number[] = [];
  const centers = [
    [-2, -2],
    [2, 2],
  ];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    features.push([centers[c][0] + gauss(rng) * 1.1, centers[c][1] + gauss(rng) * 1.1]);
    labels.push(c);
  }
  return {
    name: "Gaussian Blobs",
    features,
    labels,
    featureNames: ["x₁", "x₂"],
    classNames: ["Class 0", "Class 1"],
    description: "Two well-separated Gaussian clusters — linearly separable.",
  };
}

export function makeCircles(n = 300, noise = 0.1, seed = 5): Dataset {
  const rng = mulberry32(seed);
  const features: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const inner = i % 2 === 0;
    const r = inner ? 0.5 : 1.2;
    const t = 2 * Math.PI * rng();
    features.push([r * Math.cos(t) + gauss(rng) * noise, r * Math.sin(t) + gauss(rng) * noise]);
    labels.push(inner ? 0 : 1);
  }
  return {
    name: "Concentric Circles",
    features,
    labels,
    featureNames: ["x₁", "x₂"],
    classNames: ["Inner", "Outer"],
    description: "Non-linear, radially separated — tests model flexibility.",
  };
}

// A compact synthetic "medical screening" dataset: 5 features → binary diagnosis.
export function makeMedical(n = 400, seed = 21): Dataset {
  const rng = mulberry32(seed);
  const features: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const age = 30 + rng() * 50;
    const bmi = 18 + rng() * 20;
    const bp = 90 + rng() * 60;
    const glucose = 70 + rng() * 120;
    const chol = 150 + rng() * 150;
    // logistic score with noise
    const z =
      0.04 * (age - 55) +
      0.08 * (bmi - 27) +
      0.03 * (bp - 120) +
      0.02 * (glucose - 110) +
      0.01 * (chol - 220) +
      gauss(rng) * 0.8;
    const p = 1 / (1 + Math.exp(-z));
    const y = rng() < p ? 1 : 0;
    features.push([age, bmi, bp, glucose, chol]);
    labels.push(y);
  }
  return {
    name: "Medical Screening",
    features,
    labels,
    featureNames: ["Age", "BMI", "Blood Pressure", "Glucose", "Cholesterol"],
    classNames: ["Healthy", "At Risk"],
    description: "Synthetic patient data — predict at-risk diagnosis from vitals.",
  };
}

export const DATASETS: Record<string, () => Dataset> = {
  moons: () => makeMoons(),
  blobs: () => makeBlobs(),
  circles: () => makeCircles(),
  medical: () => makeMedical(),
};