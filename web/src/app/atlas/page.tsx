import type { Metadata } from "next";
import { Suspense } from "react";
import { api, safe } from "@/lib/api";
import { ApiDown, Spinner } from "@/components/ui";
import { AtlasChart } from "./AtlasChart";

// Render per request: the API data (fetch-cached with revalidate) must never be
// frozen into a static build, e.g. the "service not reachable" banner.
export const dynamic = "force-dynamic";

export const metadata: Metadata = { title: "Atlas" };

export default async function AtlasPage() {
  const { data, error } = await safe(api.atlas());
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-3xl">Embedding atlas</h1>
        <p className="mt-1 max-w-prose text-sm text-ink-2">
          All {data?.n.toLocaleString() ?? "1,236"} specimen images, placed by a 2-D UMAP of their BioCLIP 2
          embeddings (cosine metric, {String(data?.umap.n_neighbors ?? 15)} neighbours). Nearby points look alike to the
          model; axes have no units. Hover for the specimen, click to open it on AntWeb.
        </p>
      </div>
      {error ? <ApiDown error={error} /> : (
        <Suspense fallback={<Spinner label="Loading atlas…" />}>
          <AtlasChart points={data!.points} />
        </Suspense>
      )}
    </div>
  );
}
