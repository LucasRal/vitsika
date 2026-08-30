"use client";
import Link from "next/link";
import { useAnalysis } from "@/lib/analysis-context";
import { imageUrl, antwebUrl } from "@/lib/api";
import { pct } from "@/lib/format";
import { subfamilyColour } from "@/lib/palette";
import { AntwebCredit, BackToIdentify, Banner, Card, Code, Eyebrow, Genus, InfoIcon, ProbBar, SectionTitle, Species, Spinner } from "@/components/ui";
import { verdict } from "@/lib/verdict";

export default function ResultPage() {
  const { analysis, hydrated } = useAnalysis();
  if (!hydrated) return <Spinner label="Loading…" />;
  if (!analysis) {
    return (
      <div className="max-w-prose space-y-3">
        <h1 className="text-2xl">No analysis to show</h1>
        <p className="text-sm text-ink-2">Results live only in this browser tab and are gone after a reload.</p>
        <Link className="btn btn-primary" href="/">Identify a specimen</Link>
      </div>
    );
  }
  const { result, imageUrl: query, filename, truth } = analysis;
  const top = result.predictions[0];
  const v = verdict(result);
  const hit = truth ? top.genus === truth.genus : null;
  const rank = truth ? result.predictions.findIndex((p) => p.genus === truth.genus) : -1;
  const star = result.atlas_position;
  const atlasHref = star
    ? `/atlas?star=1&x=${star.x.toFixed(3)}&y=${star.y.toFixed(3)}&label=${encodeURIComponent("your photo")}`
    : null;

  return (
    <div className="space-y-8">
      <div><BackToIdentify /></div>
      <div className="grid gap-8 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <section>
          <Eyebrow>Query image</Eyebrow>
          {query ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={query} alt={filename} className="hairline max-h-80 w-full rounded-md object-contain bg-surface-2" />
          ) : (
            <div className="hairline flex h-40 items-center justify-center rounded-md bg-surface-2 text-sm text-muted">photo not kept after reload (too large to store)</div>
          )}
          <p className="mt-1 text-xs text-muted">{filename} · embedded in {result.embed_ms.toFixed(0)} ms</p>
        </section>
        <section>
          <Eyebrow>Genus prediction</Eyebrow>
          <h1 className="text-3xl"><Genus name={top.genus} /> <span className="text-lg text-muted">{pct(top.probability)}</span></h1>
          <p className="text-sm text-muted">subfamily {top.subfamily}</p>
          {truth && (
            <p className={`mt-2 rounded-sm px-2 py-1 text-sm ${hit ? "bg-success-soft text-success" : "bg-warning-soft text-warning"}`} role="status">
              {hit ? "✓" : "✗"} Held-out test specimen <Code code={truth.specimen_code} /> (AntWeb label:{" "}
              <Species name={truth.species ?? truth.genus} />)
              {hit ? " · correct" : rank > 0 ? ` · the right genus is rank ${rank + 1}` : " · not in the top 3"}
            </p>
          )}
          {v.tier === "supported" && (
            <div className="hairline mt-3 flex items-start gap-2 rounded-md bg-surface-2 px-4 py-3 text-sm text-ink-2" role="status" data-tier="supported">
              <InfoIcon className="mt-0.5 shrink-0 text-muted" />
              <p>
                Classifier confidence is modest ({pct(v.top1)}), but {v.support} of the {v.nSimilar} nearest reference specimens are{" "}
                <Genus name={v.genus} /> (<a className="link" href="#similar">see below</a>).
              </p>
            </div>
          )}
          {v.tier === "low" && (
            <div className="mt-3" data-tier="low">
              <Banner tone="warning" title={`Low confidence: the best guess is only ${pct(top.probability)}.`}>
                Nothing in the training set looks quite like this. The photo may be a non-profile view, a different
                caste, a genus outside the 27 covered, or not an ant. Use the similar specimens below to judge.
              </Banner>
            </div>
          )}
          <ol className="mt-4 space-y-3">
            {result.predictions.map((p, i) => (
              <li key={p.genus} className="grid grid-cols-[1.5rem_1fr_4rem] items-center gap-3 text-sm">
                <span className="text-muted">{i + 1}.</span>
                <div>
                  <div className="flex items-baseline gap-2">
                    <Genus name={p.genus} className="font-medium" />
                    <span className="text-xs text-muted">{p.subfamily}</span>
                  </div>
                  <ProbBar value={p.probability} colour={subfamilyColour(p.subfamily)} />
                </div>
                <span className="code text-right">{pct(p.probability, 1)}</span>
              </li>
            ))}
          </ol>
          <div className="mt-5 flex flex-wrap gap-2">
            {atlasHref ? <Link className="btn" href={atlasHref}>See where this lands on the atlas →</Link>
                       : <span className="text-xs text-muted">Atlas position unavailable for this image.</span>}
            <Link className="btn" href={`/distribution?genus=${encodeURIComponent(top.genus)}`}>Where <Genus name={top.genus} /> has been collected →</Link>
            <BackToIdentify />
          </div>
          <p className="mt-4 text-xs text-muted">
            {result.model_name} · probe {result.probe_version.slice(0, 10)} · probabilities calibrated on cross-validated training folds
            (temperature scaling, 5 folds); still an answer among the 27 genera only.
          </p>
        </section>
      </div>

      <section id="similar">
        <SectionTitle hint="cosine similarity of BioCLIP 2 embeddings; species labels are AntWeb’s, not the model’s">
          Most similar training specimens
        </SectionTitle>
        <ul className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {result.similar.map((s) => (
            <li key={s.specimen_code}>
              <Card className="h-full !p-3">
                <a href={antwebUrl(s.specimen_code)} target="_blank" rel="noreferrer">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imageUrl(s.specimen_code)} alt={`${s.species ?? s.genus} ${s.specimen_code}`}
                       className="aspect-[4/3] w-full rounded-sm bg-surface-2 object-contain" loading="lazy" />
                </a>
                <p className="mt-2 text-sm leading-tight"><Species name={s.species ?? s.genus} /></p>
                <p className="text-xs text-muted">{s.subfamily}</p>
                <p className="mt-1 flex items-center justify-between text-xs">
                  <Code code={s.specimen_code} />
                  <span className="code font-semibold text-ink" title="cosine similarity">{s.similarity.toFixed(3)}</span>
                </p>
                <div className="mt-1"><AntwebCredit code={s.specimen_code} photographer={s.photographer} /></div>
              </Card>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
