// Lightweight from-scratch ML models for binary classification.
// All models expose: fit(X, y) and predictProba(X) -> number[] in [0,1].

export interface Model {
  fit(X: number[][], y: number[]): void;
  predictProba(X: number[][]): number[];
}

function rngFactory(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ----------------------------- Logistic Regression ----------------------------- */

export class LogisticRegression implements Model {
  private w: number[] = [];
  private b = 0;
  private mean: number[] = [];
  private std: number[] = [];

  constructor(
    private lr = 0.1,
    private epochs = 400,
    private l2 = 0.001,
  ) {}

  private standardize(X: number[][]): number[][] {
    const n = X.length;
    const d = X[0].length;
    if (this.mean.length === 0) {
      this.mean = new Array(d).fill(0);
      this.std = new Array(d).fill(0);
      for (const row of X) for (let j = 0; j < d; j++) this.mean[j] += row[j] / n;
      for (const row of X)
        for (let j = 0; j < d; j++) this.std[j] += (row[j] - this.mean[j]) ** 2 / n;
      this.std = this.std.map((v) => Math.sqrt(v) || 1);
    }
    return X.map((r) => r.map((v, j) => (v - this.mean[j]) / this.std[j]));
  }

  fit(X: number[][], y: number[]): void {
    const Xs = this.standardize(X);
    const n = Xs.length;
    const d = Xs[0].length;
    this.w = new Array(d).fill(0);
    this.b = 0;
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const gw = new Array(d).fill(0);
      let gb = 0;
      for (let i = 0; i < n; i++) {
        let z = this.b;
        for (let j = 0; j < d; j++) z += this.w[j] * Xs[i][j];
        const p = 1 / (1 + Math.exp(-z));
        const err = p - y[i];
        for (let j = 0; j < d; j++) gw[j] += (err * Xs[i][j]) / n;
        gb += err / n;
      }
      for (let j = 0; j < d; j++) this.w[j] -= this.lr * (gw[j] + this.l2 * this.w[j]);
      this.b -= this.lr * gb;
    }
  }

  predictProba(X: number[][]): number[] {
    const Xs = X.map((r) => r.map((v, j) => (v - this.mean[j]) / this.std[j]));
    return Xs.map((row) => {
      let z = this.b;
      for (let j = 0; j < row.length; j++) z += this.w[j] * row[j];
      return 1 / (1 + Math.exp(-z));
    });
  }
}

/* ----------------------------- Decision Tree (CART, Gini) ----------------------------- */

type TreeNode =
  | { leaf: true; value: number /* probability of class 1 */ }
  | { leaf: false; feature: number; threshold: number; left: TreeNode; right: TreeNode };

export class DecisionTree implements Model {
  protected root: TreeNode | null = null;

  constructor(
    private maxDepth = 6,
    private minSamples = 4,
    private featureSubset?: number, // if set, pick this many features per split (for RF)
    private rng: () => number = Math.random,
  ) {}

  fit(X: number[][], y: number[]): void {
    this.root = this.build(X, y, 0);
  }

  private gini(y: number[]): number {
    if (y.length === 0) return 0;
    let ones = 0;
    for (const v of y) ones += v;
    const p = ones / y.length;
    return 1 - p * p - (1 - p) * (1 - p);
  }

  private build(X: number[][], y: number[], depth: number): TreeNode {
    const ones = y.reduce((a, b) => a + b, 0);
    const proba = ones / Math.max(1, y.length);
    if (
      depth >= this.maxDepth ||
      y.length <= this.minSamples ||
      ones === 0 ||
      ones === y.length
    ) {
      return { leaf: true, value: proba };
    }
    const d = X[0].length;
    let features = Array.from({ length: d }, (_, i) => i);
    if (this.featureSubset && this.featureSubset < d) {
      // sample without replacement
      features = features
        .map((f) => ({ f, r: this.rng() }))
        .sort((a, b) => a.r - b.r)
        .slice(0, this.featureSubset)
        .map((o) => o.f);
    }

    let bestGain = 0;
    let bestFeat = -1;
    let bestThr = 0;
    const parentImp = this.gini(y);

    for (const f of features) {
      const values = X.map((row) => row[f]).slice().sort((a, b) => a - b);
      // try ~10 candidate thresholds
      const step = Math.max(1, Math.floor(values.length / 10));
      for (let i = step; i < values.length; i += step) {
        const thr = (values[i - 1] + values[i]) / 2;
        const leftY: number[] = [];
        const rightY: number[] = [];
        for (let k = 0; k < X.length; k++) {
          if (X[k][f] <= thr) leftY.push(y[k]);
          else rightY.push(y[k]);
        }
        if (leftY.length === 0 || rightY.length === 0) continue;
        const w = leftY.length / y.length;
        const gain = parentImp - w * this.gini(leftY) - (1 - w) * this.gini(rightY);
        if (gain > bestGain) {
          bestGain = gain;
          bestFeat = f;
          bestThr = thr;
        }
      }
    }

    if (bestFeat === -1) return { leaf: true, value: proba };

    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];
    for (let k = 0; k < X.length; k++) {
      if (X[k][bestFeat] <= bestThr) {
        leftX.push(X[k]);
        leftY.push(y[k]);
      } else {
        rightX.push(X[k]);
        rightY.push(y[k]);
      }
    }

    return {
      leaf: false,
      feature: bestFeat,
      threshold: bestThr,
      left: this.build(leftX, leftY, depth + 1),
      right: this.build(rightX, rightY, depth + 1),
    };
  }

  private predictOne(row: number[]): number {
    let node = this.root!;
    while (!node.leaf) {
      node = row[node.feature] <= node.threshold ? node.left : node.right;
    }
    return node.value;
  }

  predictProba(X: number[][]): number[] {
    return X.map((r) => this.predictOne(r));
  }
}

/* ----------------------------- Random Forest ----------------------------- */

export class RandomForest implements Model {
  private trees: DecisionTree[] = [];
  constructor(
    private nTrees = 25,
    private maxDepth = 8,
    private seed = 42,
  ) {}

  fit(X: number[][], y: number[]): void {
    const rng = rngFactory(this.seed);
    const d = X[0].length;
    const featureSubset = Math.max(1, Math.round(Math.sqrt(d)));
    this.trees = [];
    for (let t = 0; t < this.nTrees; t++) {
      // bootstrap sample
      const bx: number[][] = [];
      const by: number[] = [];
      for (let i = 0; i < X.length; i++) {
        const idx = Math.floor(rng() * X.length);
        bx.push(X[idx]);
        by.push(y[idx]);
      }
      const tree = new DecisionTree(this.maxDepth, 3, featureSubset, rng);
      tree.fit(bx, by);
      this.trees.push(tree);
    }
  }

  predictProba(X: number[][]): number[] {
    const sums = new Array(X.length).fill(0);
    for (const tree of this.trees) {
      const p = tree.predictProba(X);
      for (let i = 0; i < X.length; i++) sums[i] += p[i];
    }
    return sums.map((s) => s / this.trees.length);
  }
}