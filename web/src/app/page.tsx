import { Suspense } from "react";
import { Dropzone } from "@/components/Dropzone";
import { ModelCard } from "@/components/ModelCard";
import { Spinner } from "@/components/ui";

// Render per request: the API data (fetch-cached with revalidate) must never be
// frozen into a static build, e.g. the "service not reachable" banner.
export const dynamic = "force-dynamic";

export default function IdentifyPage() {
  return (
    <div className="grid gap-8 lg:grid-cols-[3fr_2fr]">
      <section>
        <h1 className="text-3xl">Which genus is this ant?</h1>
        <p className="mt-2 max-w-prose text-sm text-ink-2">
          Upload a lateral photo of a worker ant from Madagascar. A frozen BioCLIP 2 vision model embeds it;
          a small linear classifier trained on 986 AntWeb specimens proposes the three most likely genera,
          and the five most similar museum specimens are shown so you can judge for yourself.
        </p>
        <div className="mt-6"><Suspense fallback={<Spinner label="Loading…" />}><Dropzone /></Suspense></div>
      </section>
      <aside>
        <Suspense fallback={<Spinner label="Loading model card…" />}>
          <ModelCard />
        </Suspense>
      </aside>
    </div>
  );
}
