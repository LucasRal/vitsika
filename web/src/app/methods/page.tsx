import type { Metadata } from "next";
import metrics from "@/data/metrics.json";
import { pct } from "@/lib/format";
import { Genus } from "@/components/ui";

export const metadata: Metadata = { title: "Methods" };

const STEPS = [
  ["GBIF harvest", "AntWeb occurrence records for Madagascar, Formicidae, workers"],
  ["Manifest", "4,354 specimens · 39 genera · profile view preferred"],
  ["Images", "1,236 profile photos obtained (Wikimedia Commons + AntWeb)"],
  ["Filter & split", "≥ 10 images per genus → 27 genera · 80/20 by genus, seed 42"],
  ["Embed", "BioCLIP 2 ViT-L/14, frozen · 768-d · L2-normalised"],
  ["Classify", "Logistic-regression probe · balanced classes · C = 1 · sigmoid-calibrated (5 folds)"],
  ["Evaluate", "250 held-out specimens · top-1 / top-3 / macro-F1"],
  ["Serve", "FastAPI · this site"],
];

const Chip = ({ title, text }: { title: string; text: string }) => (
  <li className="hairline min-w-40 flex-1 rounded-md bg-surface-2 px-3 py-2">
    <p className="text-sm font-medium">{title}</p>
    <p className="text-xs text-muted">{text}</p>
  </li>
);

export default function MethodsPage() {
  const m = metrics.methods;
  return (
    <article className="max-w-3xl space-y-10">
      <header>
        <h1 className="text-3xl">Methods, provenance and limitations</h1>
        <p className="mt-2 text-sm text-ink-2">
          Vitsika is a proof of concept for genus-level identification of Malagasy ants from a single photograph. It
          is built entirely on public data and a public, frozen vision model; nothing was fine-tuned.
        </p>
      </header>

      <section>
        <h2 className="mb-3 text-xl">Pipeline</h2>
        <ol className="flex flex-wrap gap-2">{STEPS.map(([t, x]) => <Chip key={t} title={t} text={x} />)}</ol>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl">Provenance</h2>
        <p className="text-sm">
          Records come from the AntWeb dataset on GBIF (Fisher B. L., California Academy of Sciences,{" "}
          <a className="link" href="https://doi.org/10.15468/wqmjjt" target="_blank" rel="noreferrer">doi:10.15468/wqmjjt</a>, accessed 2026-08-26), restricted to Madagascar,
          worker caste and specimens with a profile-view (lateral) image. Genus synonyms were remapped before
          thresholding (<Genus name="Oligomyrmex" /> → <Genus name="Carebara" />, <Genus name="Pyramica" /> → <Genus name="Strumigenys" />).
          Provinces use current names. The full manifest holds 4,354 specimens in 39 genera and 8 subfamilies; images
          were obtained for 1,236 of them, and the 27 genera with at least 10 images form the classifier’s scope.
        </p>
        <p className="text-sm">
          Every image is one specimen: {metrics.n_train} for training and {metrics.n_test} held out, split by genus (not
          by species), so some test species have no training example at all.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl">Results on the held-out set</h2>
        <div className="hairline overflow-x-auto rounded-md">
          <table className="w-full text-sm">
            <thead className="bg-surface-2 text-left text-xs uppercase tracking-wider text-muted">
              <tr><th className="px-3 py-2 font-medium">Method</th><th className="px-3 py-2 text-right font-medium">Top-1</th><th className="px-3 py-2 text-right font-medium">Top-3</th><th className="px-3 py-2 text-right font-medium">Macro-F1</th></tr>
            </thead>
            <tbody className="code">
              {[["Linear probe (used here)", m.linear_probe], ["Nearest neighbour", m.nearest_neighbour],
                ["Zero-shot, “a photo of {genus}, a genus of ant”", m.zero_shot_template], ["Zero-shot, bare genus name", m.zero_shot_plain]].map(([name, r]) => {
                const v = r as { top1: number; top3: number; macro_f1: number };
                return (
                  <tr key={String(name)} className="hairline-t">
                    <td className="px-3 py-2 font-sans">{String(name)}</td>
                    <td className="px-3 py-2 text-right">{pct(v.top1, 1)}</td><td className="px-3 py-2 text-right">{pct(v.top3, 1)}</td><td className="px-3 py-2 text-right">{v.macro_f1.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <figure className="hairline rounded-md bg-surface p-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/confusion_matrix.png" alt="Linear probe confusion matrix, 250 test images, 27 genera ordered by subfamily"
               width={1950} height={1800} className="w-full rounded-sm" loading="lazy" />
          <figcaption className="mt-2 text-xs text-muted">
            Linear-probe confusion matrix on the {metrics.n_test} held-out specimens, genera ordered by subfamily (grey lines).
            Nearly every off-diagonal count sits inside a subfamily block: the model confuses genera a taxonomist would
            also expect to look alike. Generated by <span className="code">scripts/07_eval.py</span>.
          </figcaption>
        </figure>
        <p className="text-sm">
          Of the 18 probe errors, 12 are genuinely hard (look-alike genera such as <Genus name="Syllophopsis" /> vs <Genus name="Tetramorium" />,
          or species unseen in training), three are poor photographs, two are odd angles and one is a specimen that the
          embedding places among <Genus name="Anochetus" /> despite its label. Two paralectotypes of{" "}
          <Genus name="Camponotus imitator" /> are called <Genus name="Aphaenogaster" />: the model is recapitulating a
          resemblance in body plan, which is exactly what its atlas shows as the “long-legged island”.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl">Limitations (please read)</h2>
        <ul className="list-disc space-y-1.5 pl-5 text-sm">
          <li>Genus only. Species names on similar specimens are AntWeb’s determinations, not predictions.</li>
          <li>Closed world: a photo of a genus outside the 27, of a queen or male, or of something that is not an ant still gets an answer. Probabilities are calibrated on cross-validated training folds (Platt sigmoid, 5 folds), so “70 %” means about 7 in 10 such calls were right on held-out data; but that holds only for photos like the training ones. Below 50 % we say so.</li>
          <li>Trained on standardised museum photographs (lateral view, white background, pinned specimen). Field photos of live ants are out of distribution and will do worse.</li>
          <li>Small test sets: several genera have 3–6 test images, so their per-genus scores are coarse; see the reliability flags in Genera.</li>
          <li>Maps show collecting effort, not abundance or range: they are where museum specimens were collected and georeferenced.</li>
          <li>Roughly 1 s per image on a 6-core CPU; not built for batch use.</li>
        </ul>
      </section>

    </article>
  );
}
