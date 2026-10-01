"use client";
import { useEffect, useRef, useState } from "react";
import { apiBase } from "@/lib/api";
import { Genus } from "@/components/ui";

type Msg = { role: "user" | "assistant"; content: string };
const MAX_HISTORY = 12;

/** "Discuss this map": a small streaming chat scoped to one RISE map. The
 * server (POST /saliency/chat) owns the system prompt and the images; this
 * panel only keeps the visible history and streams the reply text. */
export function SaliencyChat({ code, target }: { code: string; target: string }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => { bottom.current?.scrollIntoView({ block: "nearest" }); }, [messages]);
  useEffect(() => () => abort.current?.abort(), []);

  const clear = () => { abort.current?.abort(); setMessages([]); setError(null); setBusy(false); };

  const send = async () => {
    const text = draft.trim();
    if (!text || busy) return;
    setError(null);
    setDraft("");
    const history = [...messages, { role: "user", content: text } as Msg].slice(-MAX_HISTORY);
    setMessages([...history, { role: "assistant", content: "" }]);
    setBusy(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    try {
      const res = await fetch(`${apiBase()}/saliency/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: ctrl.signal,
        body: JSON.stringify({ specimen: code, target, messages: history }),
      });
      if (!res.ok || !res.body) {
        let message = `${res.status} ${res.statusText}`;
        try { message = (await res.json()).message ?? message; } catch { /* non-JSON */ }
        throw new Error(message);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const piece = decoder.decode(value, { stream: true });
        setMessages((m) => {
          const last = m[m.length - 1];
          return [...m.slice(0, -1), { ...last, content: last.content + piece }];
        });
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message);
        setMessages((m) => (m[m.length - 1]?.content === "" ? m.slice(0, -1) : m));
      }
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button type="button" className="btn !py-1.5 text-xs" onClick={() => setOpen(true)}>
        Discuss this map
      </button>
    );
  }
  const full = messages.length >= MAX_HISTORY;
  return (
    <div className="hairline rounded-md bg-surface p-3 text-sm" data-testid="saliency-chat">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs text-muted">
          AI conversation about this map’s evidence (<Genus name={target} /> on <span className="code">{code}</span>).
          Regions are read from the overlay, not measured.
        </p>
        <span className="flex gap-3 text-xs">
          <button type="button" className="text-muted hover:text-ink" onClick={clear}>Clear</button>
          <button type="button" className="text-muted hover:text-ink" onClick={() => { clear(); setOpen(false); }}>Close</button>
        </span>
      </div>
      <div className="mt-2 max-h-72 space-y-2 overflow-y-auto" aria-live="polite">
        {messages.length === 0 && (
          <p className="text-xs text-muted">
            Ask, for example: “Which region carries the most weight?”, “Why is the gaster blue?”, or
            “How strong is this evidence compared with the other maps?”
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "flex justify-end" : "flex"}>
            <p className={`max-w-[85%] whitespace-pre-wrap rounded-md px-3 py-1.5 ${m.role === "user" ? "bg-accent-soft text-ink" : "bg-surface-2 text-ink-2"}`}>
              {m.content || (busy ? "…" : "")}
            </p>
          </div>
        ))}
        <div ref={bottom} />
      </div>
      {error && <p className="mt-2 text-xs text-warning">{error}</p>}
      {full && <p className="mt-2 text-xs text-muted">This conversation is at its {MAX_HISTORY}-message limit; Clear to start again.</p>}
      <form className="mt-2 flex gap-2" onSubmit={(e) => { e.preventDefault(); void send(); }}>
        <input className="input flex-1" value={draft} onChange={(e) => setDraft(e.target.value)}
               placeholder="Ask about this map…" maxLength={2000} disabled={busy || full}
               aria-label={`Ask about the ${target} map for ${code}`} />
        <button type="submit" className="btn btn-primary !py-1.5 text-xs" disabled={busy || full || !draft.trim()}>
          {busy ? "Answering…" : "Send"}
        </button>
      </form>
    </div>
  );
}
