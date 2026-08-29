import Link from "next/link";
import type { ReactNode } from "react";
import { antwebUrl } from "@/lib/api";

/** Genus names are always italic. */
export const Genus = ({ name, className = "" }: { name: string; className?: string }) => (
  <span className={`genus ${className}`}>{name}</span>
);
/** Species is "Genus epithet": both italic. */
export const Species = ({ name }: { name: string | null }) =>
  name ? <span className="genus">{name}</span> : <span className="text-muted">sp. (unidentified)</span>;
/** Specimen codes are always mono. */
export const Code = ({ code, className = "" }: { code: string; className?: string }) => (
  <span className={`code ${className}`}>{code}</span>
);

/** Mandatory attribution under every AntWeb image (CC BY-SA). */
export function AntwebCredit({ code, photographer }: { code: string; photographer: string | null }) {
  return (
    <p className="text-[11px] leading-tight text-muted">
      © {photographer ?? "AntWeb"} ·{" "}
      <a className="link" href={antwebUrl(code)} target="_blank" rel="noreferrer">AntWeb</a>
    </p>
  );
}

/** "← Identify another specimen": lands on / with the dropzone reset and focused
 * (the Dropzone consumes the ?new=1 flag). Secondary button or plain link. */
export const BackToIdentify = ({ label = "Identify another specimen", plain = false }: { label?: string; plain?: boolean }) => (
  <Link href="/?new=1" className={plain ? "link" : "btn"} data-testid="back-to-identify">
    <span aria-hidden>←</span> {label}
  </Link>
);

export const Card = ({ children, className = "" }: { children: ReactNode; className?: string }) => (
  <div className={`hairline rounded-md bg-surface p-5 ${className}`}>{children}</div>
);

export const SectionTitle = ({ children, hint }: { children: ReactNode; hint?: ReactNode }) => (
  <div className="mb-3 flex items-baseline justify-between gap-4">
    <h2 className="text-lg">{children}</h2>
    {hint && <span className="text-xs text-muted">{hint}</span>}
  </div>
);

export const Eyebrow = ({ children }: { children: ReactNode }) => (
  <p className="mb-1 text-[11px] font-medium uppercase tracking-[0.12em] text-muted">{children}</p>
);

type Tone = "neutral" | "accent" | "success" | "warning";
const tones: Record<Tone, string> = {
  neutral: "bg-surface-2 text-ink-2",
  accent: "bg-accent-soft text-accent",
  success: "bg-success-soft text-success",
  warning: "bg-warning-soft text-warning",
};
export const Badge = ({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) => (
  <span className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
    {children}
  </span>
);

export function Banner({ tone = "warning", title, children }: { tone?: Tone; title: string; children?: ReactNode }) {
  const border = tone === "warning" ? "border-warning" : tone === "success" ? "border-success" : "border-accent";
  return (
    <div className={`hairline rounded-md border-l-2 ${border} ${tones[tone]} px-4 py-3 text-sm`} role="status">
      <p className="font-medium">{title}</p>
      {children && <div className="mt-1 text-ink-2">{children}</div>}
    </div>
  );
}

/** Shown by every page whose data fetch failed; the API being down is not a crash. */
export function ApiDown({ error }: { error: string }) {
  return (
    <Banner tone="warning" title="The analysis service is not reachable right now.">
      <p>{error}</p>
      <p className="mt-1">
        Start it with <span className="code">uvicorn api.main:app --port 8001</span> or check{" "}
        <span className="code">NEXT_PUBLIC_API_URL</span>. Static pages (<Link className="link" href="/methods">Methods</Link>) still work.
      </p>
    </Banner>
  );
}

/** Small "i" glyph for neutral informational notes (never colour-only: always beside text). */
export const InfoIcon = ({ className = "" }: { className?: string }) => (
  <svg className={className} width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden>
    <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1" />
    <path d="M8 7v4.5M8 4.6v.2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
);

export const Spinner = ({ label = "Loading…" }: { label?: string }) => (
  <p className="flex items-center gap-2 text-sm text-muted" role="status">
    <span className="inline-block h-3 w-3 animate-spin rounded-full border border-muted border-t-transparent" />
    {label}
  </p>
);

/** Thin horizontal probability bar, value labelled in text (never colour-only). */
export function ProbBar({ value, colour = "var(--accent)" }: { value: number; colour?: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-sm bg-surface-2" aria-hidden>
      <div className="h-full rounded-sm" style={{ width: `${Math.max(1, value * 100)}%`, background: colour }} />
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="hairline rounded-md p-4">
      <Eyebrow>{label}</Eyebrow>
      <p className="font-serif text-2xl">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </div>
  );
}

/** "collecting effort, not abundance", shown wherever specimen counts are mapped. */
export const EffortNote = () => (
  <p className="text-xs text-muted">
    Points are museum specimens with a georeferenced label. Their density reflects{" "}
    <strong className="font-medium text-ink-2">collecting effort, not abundance</strong>: well-surveyed
    reserves look busy, unsurveyed regions look empty.
  </p>
);
