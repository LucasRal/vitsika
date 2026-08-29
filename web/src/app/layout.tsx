import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Source_Serif_4 } from "next/font/google";
import Link from "next/link";
import { AnalysisProvider } from "@/lib/analysis-context";
import { SiteHeader } from "@/components/SiteHeader";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const serif = Source_Serif_4({ subsets: ["latin"], variable: "--font-source-serif", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Vitsika", template: "%s · Vitsika" },
  description: "Genus-level identification of Malagasy ants from a profile photo: BioCLIP 2 embeddings, a linear probe, and the AntWeb specimen record.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${serif.variable} ${mono.variable}`}>
      <body className="min-h-screen flex flex-col">
        <AnalysisProvider>
          <SiteHeader />
          <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">{children}</main>
          <footer className="hairline-t">
            <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-6 py-4 text-xs text-muted">
              <span>
                <span className="wordmark">Vitsika</span>: <span className="genus">vitsika</span> is Malagasy for ant.
                Images © their photographers via <a className="link" href="https://www.antweb.org" target="_blank" rel="noreferrer">AntWeb</a> (CC BY-SA); metadata via GBIF (CC BY).
              </span>
              <Link className="link" href="/methods">Methods &amp; limitations</Link>
            </div>
          </footer>
        </AnalysisProvider>
      </body>
    </html>
  );
}
