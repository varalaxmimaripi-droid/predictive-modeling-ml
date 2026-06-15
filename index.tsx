import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  Legend,
  ReferenceLine,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { DATASETS, type Dataset } from "@/lib/ml/datasets";
import {
  DecisionTree,
  LogisticRegression,
  RandomForest,
  type Model,
} from "@/lib/ml/models";
import {
  confusionMatrix,
  metricsFromCM,
  rocCurve,
  trainTestSplit,
} from "@/lib/ml/metrics";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ML Playground — Predictive Modeling Lab" },
      {
        name: "description",
        content:
          "Train Logistic Regression, Decision Trees, and Random Forests in your browser. Visualize accuracy, confusion matrices, and ROC curves on real datasets.",
      },
      { property: "og:title", content: "ML Playground — Predictive Modeling Lab" },
      {
        property: "og:description",
        content:
          "An interactive supervised learning sandbox: pick a dataset, train a model, inspect confusion matrices and ROC curves.",
      },
    ],
  }),
  component: Index,
});

type ModelKey = "logreg" | "tree" | "forest";

const MODEL_INFO: Record<ModelKey, { name: string; tagline: string }> = {
  logreg: { name: "Logistic Regression", tagline: "Linear, interpretable baseline" },
  tree: { name: "Decision Tree", tagline: "Non-linear, rule-based splits" },
  forest: { name: "Random Forest", tagline: "Ensemble of bagged trees" },
};

function buildModel(key: ModelKey): Model {
  if (key === "logreg") return new LogisticRegression(0.1, 500, 0.001);
  if (key === "tree") return new DecisionTree(6, 4);
  return new RandomForest(30, 8, 42);
}

type RunResult = {
  modelKey: ModelKey;
  datasetKey: string;
  cm: { tp: number; fp: number; tn: number; fn: number };
  metrics: { accuracy: number; precision: number; recall: number; f1: number };
  roc: { points: { fpr: number; tpr: number }[]; auc: number };
  threshold: number;
  trainTimeMs: number;
  scatter: { x: number; y: number; cls: number; pred: number }[]; // test set, 2D only
  features: string[];
  classes: [string, string];
  trainSize: number;
  testSize: number;
};

function Index() {
  const [datasetKey, setDatasetKey] = useState<keyof typeof DATASETS>("moons");
  const [modelKey, setModelKey] = useState<ModelKey>("forest");
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState<RunResult | null>(null);
  const [running, setRunning] = useState(false);

  const dataset: Dataset = useMemo(() => DATASETS[datasetKey](), [datasetKey]);

  const train = async () => {
    setRunning(true);
    // yield to paint
    await new Promise((r) => setTimeout(r, 30));
    const { Xtr, ytr, Xte, yte } = trainTestSplit(dataset.features, dataset.labels, 0.25, 1);
    const model = buildModel(modelKey);
    const t0 = performance.now();
    model.fit(Xtr, ytr);
    const trainTimeMs = performance.now() - t0;
    const proba = model.predictProba(Xte);
    const yPred = proba.map((p) => (p >= threshold ? 1 : 0));
    const cm = confusionMatrix(yte, yPred);
    const metrics = metricsFromCM(cm);
    const roc = rocCurve(yte, proba);
    const scatter =
      Xte[0].length === 2
        ? Xte.map((row, i) => ({ x: row[0], y: row[1], cls: yte[i], pred: yPred[i] }))
        : [];

    setResult({
      modelKey,
      datasetKey,
      cm,
      metrics,
      roc,
      threshold,
      trainTimeMs,
      scatter,
      features: dataset.featureNames,
      classes: dataset.classNames,
      trainSize: Xtr.length,
      testSize: Xte.length,
    });
    setRunning(false);
  };

  // Recompute threshold-dependent metrics without retraining when threshold slider moves
  const liveResult = useMemo(() => {
    if (!result) return null;
    const pts = result.roc.points;
    // pick scatter pred at new threshold (only meaningful when we have probas — we stored cls/pred at training threshold)
    return { ...result, roc: { ...result.roc }, threshold };
    // We keep the metric card on retraining; threshold slider primarily tunes future runs.
  }, [result, threshold]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/60 bg-card/40 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-6 py-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-block size-2 rounded-full bg-primary shadow-[0_0_12px_currentColor]" />
              <h1 className="text-xl font-semibold tracking-tight">ML Playground</h1>
              <Badge variant="secondary" className="ml-1">
                supervised learning
              </Badge>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              Train predictive models in your browser. No server, no setup.
            </p>
          </div>
          <div className="text-xs text-muted-foreground">
            Models run client-side · {dataset.features.length} samples loaded
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 px-6 py-8 lg:grid-cols-[320px_1fr]">
        {/* ---------------- Controls ---------------- */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle className="text-base">Experiment setup</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Dataset
              </label>
              <Select value={datasetKey} onValueChange={(v) => setDatasetKey(v as keyof typeof DATASETS)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="moons">Two Moons</SelectItem>
                  <SelectItem value="blobs">Gaussian Blobs</SelectItem>
                  <SelectItem value="circles">Concentric Circles</SelectItem>
                  <SelectItem value="medical">Medical Screening</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground leading-relaxed">{dataset.description}</p>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Algorithm
              </label>
              <Select value={modelKey} onValueChange={(v) => setModelKey(v as ModelKey)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="logreg">Logistic Regression</SelectItem>
                  <SelectItem value="tree">Decision Tree</SelectItem>
                  <SelectItem value="forest">Random Forest</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">{MODEL_INFO[modelKey].tagline}</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Decision threshold
                </label>
                <span className="text-xs tabular-nums text-foreground">{threshold.toFixed(2)}</span>
              </div>
              <Slider
                value={[threshold]}
                min={0.05}
                max={0.95}
                step={0.05}
                onValueChange={(v) => setThreshold(v[0])}
              />
              <p className="text-xs text-muted-foreground">
                Probability cutoff for predicting the positive class. Re-train to apply.
              </p>
            </div>

            <Button onClick={train} disabled={running} className="w-full" size="lg">
              {running ? "Training…" : "Train & evaluate"}
            </Button>

            <div className="grid grid-cols-2 gap-2 pt-2 text-xs text-muted-foreground">
              <div className="rounded-md bg-secondary/50 p-2">
                <div className="text-foreground tabular-nums">
                  {Math.round(dataset.features.length * 0.75)}
                </div>
                <div>train rows</div>
              </div>
              <div className="rounded-md bg-secondary/50 p-2">
                <div className="text-foreground tabular-nums">
                  {Math.round(dataset.features.length * 0.25)}
                </div>
                <div>test rows</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ---------------- Results ---------------- */}
        <div className="space-y-6">
          {!liveResult ? (
            <Card className="border-dashed">
              <CardContent className="flex flex-col items-center justify-center gap-3 py-20 text-center">
                <div className="size-12 rounded-full bg-primary/10 grid place-items-center text-primary">
                  ▶
                </div>
                <div>
                  <div className="font-medium">Ready when you are</div>
                  <p className="text-sm text-muted-foreground max-w-sm">
                    Pick a dataset and algorithm on the left, then hit <em>Train & evaluate</em> to
                    see accuracy, confusion matrix, and an ROC curve.
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Results result={liveResult} />
          )}
        </div>
      </main>

      <footer className="border-t border-border/60 px-6 py-6 text-center text-xs text-muted-foreground">
        Built for hands-on practice with supervised learning · models implemented from scratch in
        TypeScript
      </footer>
    </div>
  );
}

function Results({ result }: { result: RunResult }) {
  const { metrics, cm, roc, scatter, features, classes, modelKey } = result;

  return (
    <>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Accuracy" value={metrics.accuracy} highlight />
        <MetricCard label="Precision" value={metrics.precision} />
        <MetricCard label="Recall" value={metrics.recall} />
        <MetricCard label="F1 score" value={metrics.f1} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Confusion matrix</CardTitle>
            <Badge variant="outline" className="text-xs">
              {MODEL_INFO[modelKey].name}
            </Badge>
          </CardHeader>
          <CardContent>
            <ConfusionMatrix cm={cm} classes={classes} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">ROC curve</CardTitle>
            <Badge variant="outline" className="text-xs tabular-nums">
              AUC {roc.auc.toFixed(3)}
            </Badge>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={roc.points} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                    label={{
                      value: "False Positive Rate",
                      position: "insideBottom",
                      offset: -2,
                      fill: "var(--color-muted-foreground)",
                      fontSize: 11,
                    }}
                  />
                  <YAxis
                    dataKey="tpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--color-popover)",
                      border: "1px solid var(--color-border)",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v: number) => v.toFixed(3)}
                  />
                  <ReferenceLine
                    segment={[
                      { x: 0, y: 0 },
                      { x: 1, y: 1 },
                    ]}
                    stroke="var(--color-muted-foreground)"
                    strokeDasharray="4 4"
                  />
                  <Line
                    dataKey="tpr"
                    stroke="var(--color-primary)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {scatter.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Test-set predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name={features[0]}
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name={features[1]}
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                  />
                  <ZAxis range={[40, 40]} />
                  <Tooltip
                    contentStyle={{
                      background: "var(--color-popover)",
                      border: "1px solid var(--color-border)",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    cursor={{ strokeDasharray: "3 3" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Scatter
                    name={`Correct → ${classes[0]}`}
                    data={scatter.filter((p) => p.cls === 0 && p.pred === 0)}
                    fill="var(--color-chart-1)"
                  />
                  <Scatter
                    name={`Correct → ${classes[1]}`}
                    data={scatter.filter((p) => p.cls === 1 && p.pred === 1)}
                    fill="var(--color-chart-2)"
                  />
                  <Scatter
                    name="Misclassified"
                    data={scatter.filter((p) => p.cls !== p.pred)}
                    fill="var(--color-destructive)"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
}

function MetricCard({
  label,
  value,
  highlight,
}: {
  label: string;
  value: number;
  highlight?: boolean;
}) {
  return (
    <Card className={highlight ? "border-primary/40 bg-primary/5" : ""}>
      <CardContent className="p-4">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
        <div className="mt-1 text-2xl font-semibold tabular-nums">
          {(value * 100).toFixed(1)}
          <span className="text-base text-muted-foreground">%</span>
        </div>
      </CardContent>
    </Card>
  );
}

function ConfusionMatrix({
  cm,
  classes,
}: {
  cm: { tp: number; fp: number; tn: number; fn: number };
  classes: [string, string];
}) {
  const total = cm.tp + cm.fp + cm.tn + cm.fn || 1;
  const cells = [
    { label: "TN", value: cm.tn, hint: `True ${classes[0]}`, good: true },
    { label: "FP", value: cm.fp, hint: `Predicted ${classes[1]}, actually ${classes[0]}`, good: false },
    { label: "FN", value: cm.fn, hint: `Predicted ${classes[0]}, actually ${classes[1]}`, good: false },
    { label: "TP", value: cm.tp, hint: `True ${classes[1]}`, good: true },
  ];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-xs text-muted-foreground">
        <div />
        <div className="text-center">Predicted {classes[0]}</div>
        <div className="text-center">Predicted {classes[1]}</div>

        <div className="flex items-center justify-end pr-2">Actual {classes[0]}</div>
        <Cell {...cells[0]} total={total} />
        <Cell {...cells[1]} total={total} />

        <div className="flex items-center justify-end pr-2">Actual {classes[1]}</div>
        <Cell {...cells[2]} total={total} />
        <Cell {...cells[3]} total={total} />
      </div>
    </div>
  );
}

function Cell({
  label,
  value,
  hint,
  good,
  total,
}: {
  label: string;
  value: number;
  hint: string;
  good: boolean;
  total: number;
}) {
  const intensity = Math.min(1, value / Math.max(1, total * 0.6));
  const bg = good
    ? `color-mix(in oklch, var(--color-accent) ${20 + intensity * 60}%, transparent)`
    : `color-mix(in oklch, var(--color-destructive) ${15 + intensity * 55}%, transparent)`;
  return (
    <div
      className="rounded-md p-3 text-center transition-colors"
      style={{ background: bg }}
      title={hint}
    >
      <div className="text-[10px] uppercase tracking-wider text-foreground/70">{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      <div className="text-[10px] text-foreground/60">{((value / total) * 100).toFixed(1)}%</div>
    </div>
  );
}
