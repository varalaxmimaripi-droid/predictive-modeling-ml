export function trainTestSplit(
  X: number[][],
  y: number[],
  testRatio = 0.25,
  seed = 1,
): { Xtr: number[][]; ytr: number[]; Xte: number[][]; yte: number[] } {
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  // seeded shuffle
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const cut = Math.floor(n * (1 - testRatio));
  const tr = idx.slice(0, cut);
  const te = idx.slice(cut);
  return {
    Xtr: tr.map((i) => X[i]),
    ytr: tr.map((i) => y[i]),
    Xte: te.map((i) => X[i]),
    yte: te.map((i) => y[i]),
  };
}

export function confusionMatrix(yTrue: number[], yPred: number[]) {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i],
      p = yPred[i];
    if (t === 1 && p === 1) tp++;
    else if (t === 0 && p === 1) fp++;
    else if (t === 0 && p === 0) tn++;
    else fn++;
  }
  return { tp, fp, tn, fn };
}

export function metricsFromCM(cm: { tp: number; fp: number; tn: number; fn: number }) {
  const { tp, fp, tn, fn } = cm;
  const total = tp + fp + tn + fn || 1;
  const accuracy = (tp + tn) / total;
  const precision = tp / (tp + fp || 1);
  const recall = tp / (tp + fn || 1);
  const f1 = (2 * precision * recall) / (precision + recall || 1);
  return { accuracy, precision, recall, f1 };
}

export function rocCurve(yTrue: number[], yScore: number[]) {
  const pairs = yScore.map((s, i) => ({ s, y: yTrue[i] })).sort((a, b) => b.s - a.s);
  const P = yTrue.reduce((a, b) => a + b, 0) || 1;
  const N = yTrue.length - P || 1;
  const points: { fpr: number; tpr: number }[] = [{ fpr: 0, tpr: 0 }];
  let tp = 0,
    fp = 0;
  for (const { y } of pairs) {
    if (y === 1) tp++;
    else fp++;
    points.push({ fpr: fp / N, tpr: tp / P });
  }
  // AUC via trapezoidal rule
  let auc = 0;
  for (let i = 1; i < points.length; i++) {
    auc += ((points[i].fpr - points[i - 1].fpr) * (points[i].tpr + points[i - 1].tpr)) / 2;
  }
  return { points, auc };
}