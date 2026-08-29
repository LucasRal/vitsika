import metrics from "@/data/metrics.json";
import { api, safe } from "@/lib/api";
import { pct, num } from "@/lib/format";
import { ApiDown, Card, Eyebrow, Badge } from "@/components/ui";
import Link from "next/link";

/** Server component: static metrics.json (Phase C) + live /genera. */
export async function ModelCard() {
  const probe = metrics.methods.linear_probe;
  const { data, error } = await safe(api.genera());
  const nTrain = data ? data.genera.reduce((s, g) => s + g.n_train, 0) : metrics.n_train;
  const nTest = data ? data.genera.reduce((s, g) => s + g.n_test, 0) : metrics.n_test;
  const weak = data ? data.genera.filter((g) => g.f1_probe < 0.8 || g.n_test < 5).length : null;

  return (
    <Card>
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg">Model card</h2>
        <Badge tone="accent">proof of concept</Badge>
      </div>
      {error && <div className="mt-3"><ApiDown error={error} /></div>}
      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
        <div><Eyebrow>Backbone</Eyebrow><dd>BioCLIP 2 (ViT-L/14), frozen</dd></div>
        <div><Eyebrow>Classifier</Eyebrow><dd>Logistic-regression probe, balanced classes</dd></div>
        <div><Eyebrow>Scope</Eyebrow><dd>{data?.n ?? metrics.n_genera} genera · Madagascar · workers · profile view</dd></div>
        <div><Eyebrow>Training images</Eyebrow><dd>{num(nTrain)} specimens</dd></div>
        <div><Eyebrow>Held-out test</Eyebrow><dd>{num(nTest)} specimens</dd></div>
        <div><Eyebrow>Top-1 / top-3 / macro-F1</Eyebrow>
          <dd>{pct(probe.top1, 1)} / {pct(probe.top3, 1)} / {probe.macro_f1.toFixed(3)}</dd></div>
      </dl>
      <ul className="mt-4 space-y-1 text-xs text-ink-2">
        <li>· Genus level only; species shown on similar specimens come from AntWeb labels, not the model.</li>
        <li>· One photo per specimen and a split by genus, so species unseen in training are harder than the headline numbers suggest.</li>
        {weak !== null && weak > 0 && (
          <li>· {weak} of {data!.n} genera have fewer than 5 test images or F1 below 0.8 — see the reliability column in{" "}
            <Link className="link" href="/genera">Genera</Link>.</li>
        )}
        <li>· Ants outside these {metrics.n_genera} genera (or non-ants) will still get a confident-looking answer; treat probabilities below 50% as “no idea”.</li>
      </ul>
    </Card>
  );
}
