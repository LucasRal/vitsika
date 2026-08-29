"use client";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import type { GenusInfo } from "@/lib/api";
import { subfamilyColour } from "@/lib/palette";
import { Badge, Genus } from "@/components/ui";

type Key = "name" | "subfamily" | "n_train" | "n_test" | "f1_probe";
const COLS: { key: Key; label: string; numeric?: boolean }[] = [
  { key: "name", label: "Genus" }, { key: "subfamily", label: "Subfamily" },
  { key: "n_train", label: "Train", numeric: true }, { key: "n_test", label: "Test", numeric: true },
  { key: "f1_probe", label: "Probe F1", numeric: true },
];

/** Reliability: F1 >= 0.8 with at least 5 test images, otherwise say why it is low. */
export function reliability(g: GenusInfo): { tone: "success" | "warning"; label: string } {
  if (g.n_test < 5) return { tone: "warning", label: `low (${g.n_test} test images)` };
  if (g.f1_probe < 0.8) return { tone: "warning", label: `low (F1 ${g.f1_probe.toFixed(2)})` };
  return { tone: "success", label: "high" };
}

export function GeneraTable({ genera }: { genera: GenusInfo[] }) {
  const router = useRouter();
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: "name", dir: 1 });
  const [q, setQ] = useState("");
  const [sub, setSub] = useState("all");
  const subfamilies = useMemo(() => Array.from(new Set(genera.map((g) => g.subfamily))).sort(), [genera]);

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return genera
      .filter((g) => sub === "all" || g.subfamily === sub)
      .filter((g) => !needle || g.name.toLowerCase().includes(needle) || g.subfamily.includes(needle))
      .sort((a, b) => {
        const av = a[sort.key], bv = b[sort.key];
        const c = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
        return c * sort.dir || a.name.localeCompare(b.name);
      });
  }, [genera, q, sub, sort]);

  const toggle = (key: Key) => setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : key === "name" || key === "subfamily" ? 1 : -1 }));
  const nLow = genera.filter((g) => reliability(g).tone === "warning").length;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <input className="input w-56" placeholder="Search genus…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search genus" />
        <select className="input" value={sub} onChange={(e) => setSub(e.target.value)} aria-label="Filter by subfamily">
          <option value="all">All subfamilies</option>
          {subfamilies.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <span className="text-xs text-muted">{rows.length} of {genera.length} · {nLow} flagged low reliability</span>
      </div>
      <div className="hairline overflow-x-auto rounded-md">
        <table className="w-full text-sm">
          <thead className="bg-surface-2 text-left text-xs uppercase tracking-wider text-muted">
            <tr>
              {COLS.map((c) => (
                <th key={c.key} className={`hairline-b px-3 py-2 font-medium ${c.numeric ? "text-right" : ""}`}
                    aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
                  <button className="inline-flex items-center gap-1 uppercase tracking-wider hover:text-ink" onClick={() => toggle(c.key)}>
                    {c.label}<span className="w-2 text-[10px]">{sort.key === c.key ? (sort.dir === 1 ? "▲" : "▼") : ""}</span>
                  </button>
                </th>
              ))}
              <th className="hairline-b px-3 py-2 font-medium">Reliability</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((g) => {
              const r = reliability(g);
              return (
                <tr key={g.name} className="hairline-b cursor-pointer hover:bg-surface-2"
                    onClick={() => router.push(`/distribution?genus=${encodeURIComponent(g.name)}`)}
                    tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter") router.push(`/distribution?genus=${encodeURIComponent(g.name)}`); }}>
                  <td className="px-3 py-2"><Genus name={g.name} className="font-medium" /></td>
                  <td className="px-3 py-2">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: subfamilyColour(g.subfamily) }} aria-hidden />
                    {g.subfamily}
                  </td>
                  <td className="code px-3 py-2 text-right">{g.n_train}</td>
                  <td className="code px-3 py-2 text-right">{g.n_test}</td>
                  <td className="code px-3 py-2 text-right">{g.f1_probe.toFixed(3)}</td>
                  <td className="px-3 py-2"><Badge tone={r.tone}>{r.label}</Badge></td>
                </tr>
              );
            })}
            {rows.length === 0 && <tr><td colSpan={6} className="px-3 py-6 text-center text-muted">No genus matches.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
