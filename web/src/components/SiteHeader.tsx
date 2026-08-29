"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  ["/", "Identify"], ["/genera", "Genera"], ["/distribution", "Distribution"],
  ["/atlas", "Atlas"], ["/methods", "Methods"],
] as const;

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className="hairline-b">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-baseline gap-3">
          <span className="wordmark text-2xl">Vitsika</span>
          <span className="hidden text-xs text-muted sm:inline">Malagasy ant genus identification · proof of concept</span>
        </Link>
        <nav className="flex gap-1 text-sm" aria-label="Primary">
          {NAV.map(([href, label]) => {
            const active = href === "/" ? path === "/" || path === "/result" : path.startsWith(href);
            return (
              <Link key={href} href={href}
                className={`rounded-sm px-2.5 py-1 ${active ? "bg-accent-soft text-accent" : "text-ink-2 hover:bg-surface-2"}`}
                aria-current={active ? "page" : undefined}>
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
