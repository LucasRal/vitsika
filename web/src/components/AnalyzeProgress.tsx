"use client";
/** Inline progress / error panel that takes the dropzone's place while an
 * /analyze call is in flight: the query thumbnail (from the local file, so it
 * shows at once), a slim indeterminate bar in the accent colour, and staged
 * status text. No full-screen spinner. Under prefers-reduced-motion the bar
 * is hidden by CSS (.progress) and the text carries the state alone. */

export type Stage = "idle" | "loading" | "uploading" | "embedding" | "done" | "error";

export const STAGE_TEXT: Record<Exclude<Stage, "idle" | "error">, string> = {
  loading: "Loading the specimen…",
  uploading: "Uploading…",
  embedding: "Embedding with BioCLIP (about a second)…",
  done: "Done.",
};

type Props = {
  stage: Exclude<Stage, "idle">;
  preview: string | null;
  filename?: string;
  error?: string | null;
  onRetry: () => void;
  onCancel: () => void;
  compact?: boolean;
};

export function AnalyzeProgress({ stage, preview, filename, error, onRetry, onCancel, compact }: Props) {
  const failed = stage === "error";
  return (
    <div
      className={`hairline flex items-center gap-4 rounded-md p-4 ${compact ? "min-h-40" : "min-h-56"} ${failed ? "border-l-2 border-warning bg-warning-soft" : "bg-surface-2"}`}
      role="status" aria-live="polite" aria-busy={!failed}
    >
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt="" className={`shrink-0 rounded-sm bg-surface object-contain ${compact ? "h-20 w-24" : "h-28 w-36"}`} />
      ) : (
        <div className={`shrink-0 rounded-sm bg-surface ${compact ? "h-20 w-24" : "h-28 w-36"}`} aria-hidden />
      )}
      <div className="min-w-0 flex-1">
        {failed ? (
          <>
            <p className="text-sm font-medium text-warning">Analysis failed</p>
            <p className="mt-0.5 text-sm text-ink-2">{error}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button className="btn btn-primary" onClick={onRetry}>Retry</button>
              <button className="btn" onClick={onCancel}>Choose another photo</button>
            </div>
          </>
        ) : (
          <>
            <p className="text-sm text-ink">{STAGE_TEXT[stage]}</p>
            <div className="progress mt-2" aria-hidden />
            {filename && <p className="mt-2 truncate text-xs text-muted">{filename}</p>}
          </>
        )}
      </div>
    </div>
  );
}
