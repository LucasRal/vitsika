import type { Metadata } from "next";
import Image from "next/image";
import rise from "@/data/rise.json";
import { antwebUrl } from "@/lib/api";
import { Genus } from "@/components/ui";

export const metadata: Metadata = { title: "Where the model looks" };

type Run = {
  code: string;
  trueGenus: string;
  species: string;
  target: string;
  pFull: number;
  caption: string;
  creator: string;
};

export default function SaliencyPage() {
  const runs = rise.runs as Run[];
  return (
    <article className="max-w-4xl space-y-10">
      <header>
        <h1 className="text-3xl">Where the model looks</h1>
        <p className="mt-3 text-sm text-ink-2">
          Each heatmap below is a RISE saliency map (Petsiuk et al. 2018,{" "}
          <a className="link" href="https://arxiv.org/abs/1806.07421" target="_blank" rel="noreferrer">
            arXiv:1806.07421
          </a>
          ), reproducing the approach of Fisher et al.&rsquo;s unpublished 2018 study on our BioCLIP + probe
          pipeline. The idea is simple: cover random parts of the photo with {rise.n_masks} coarse masks, ask the
          classifier for the probability of a target genus each time, and average the masks weighted by that
          probability. Regions that must stay visible for the probability to stay high glow red; regions the model
          ignores stay blue. The model is a black box here (no gradients, no architecture access), which is exactly
          what makes the method honest about what the deployed pipeline actually uses.
        </p>
      </header>

      <section className="space-y-6">
        {runs.map((r) => (
          <div
            key={`${r.code}-${r.target}`}
            className="hairline grid grid-cols-1 gap-4 rounded-md bg-surface-2 p-4 sm:grid-cols-[1fr_1fr_1.1fr] sm:items-center"
          >
            <figure>
              <Image
                src={`/rise/${r.code}_photo.png`}
                alt={`${r.trueGenus} specimen ${r.code}, profile view`}
                width={480}
                height={480}
                className="hairline w-full rounded-sm bg-white"
              />
              <figcaption className="mt-1 text-xs text-muted">
                <Genus name={r.trueGenus} /> <span className="code">{r.code}</span> · ©{" "}
                {r.creator} ·{" "}
                <a className="link" href={antwebUrl(r.code)} target="_blank" rel="noreferrer">
                  AntWeb
                </a>{" "}
                (CC BY-SA)
              </figcaption>
            </figure>
            <figure>
              <Image
                src={`/rise/${r.code}_${r.target}_overlay.png`}
                alt={`RISE saliency for ${r.target} on specimen ${r.code}`}
                width={480}
                height={480}
                className="hairline w-full rounded-sm"
              />
              <figcaption className="mt-1 text-xs text-muted">
                RISE for <Genus name={r.target} /> · full-image P = {r.pFull.toFixed(3)}
              </figcaption>
            </figure>
            <p className="text-sm text-ink-2">{r.caption}</p>
          </div>
        ))}
      </section>

      <section className="space-y-2 text-xs text-muted">
        <p>
          Method: {rise.n_masks} random binary masks per map ({rise.grid}×{rise.grid} cell grid, keep probability{" "}
          {rise.p_keep}, bilinear upsampling with a random sub-cell shift, as in the paper), applied to the photo
          letterboxed to 224 × 224; each masked image goes through the frozen BioCLIP 2 tower and the{" "}
          {rise.probs === "raw" ? "probe (raw softmax; the deployed temperature-calibrated probabilities saturate at 1.0 and carry no signal for weighting)" : "calibrated probe"}. Script:{" "}
          <a
            className="link"
            href="https://github.com/LucasRal/vitsika/blob/master/scripts/12_rise.py"
            target="_blank"
            rel="noreferrer"
          >
            scripts/12_rise.py
          </a>
          . Saliency maps are qualitative; at {rise.n_masks} masks the blob scale is one grid cell, so read regions,
          not pixels.
        </p>
      </section>
    </article>
  );
}
