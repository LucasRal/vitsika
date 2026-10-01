import type { Metadata } from "next";
import rise from "@/data/rise.json";
import { SaliencyGallery, type RiseRun } from "@/components/SaliencyGallery";

export const metadata: Metadata = { title: "Where the model looks" };

export default function SaliencyPage() {
  const runs = rise.runs as RiseRun[];
  return (
    <article className="max-w-4xl space-y-10">
      <header>
        <h1 className="text-3xl">Where the model looks</h1>
        <p className="mt-3 text-sm text-ink-2">
          Each heatmap below is a RISE saliency map (Petsiuk et al. 2018,{" "}
          <a className="link" href="https://arxiv.org/abs/1806.07421" target="_blank" rel="noreferrer">
            arXiv:1806.07421
          </a>
          ) computed on our BioCLIP + probe pipeline, asking a question the accuracy numbers cannot answer: when the
          classifier names a genus, which part of the photograph is it actually using? The idea is simple: cover
          random parts of the photo with {rise.n_masks} coarse masks, ask our classifier for the probability of a
          target genus each time, and average the masks weighted by that probability. Regions that must stay visible
          for the probability to stay high glow red; regions the model ignores stay blue. The model is a black box
          here (no gradients, no architecture access), which is exactly what makes the method honest about what the
          deployed pipeline actually uses.
        </p>
      </header>

      <SaliencyGallery runs={runs} />

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
          not pixels. The per-map colour bar gives each map’s real range of E[P | region visible]; the shared view
          shows every map’s deviation from its own baseline (its mean) on one diverging scale, ±{rise.shared_w.toFixed(3)}{" "}
          being the largest deviation on this page.
        </p>
      </section>
    </article>
  );
}
