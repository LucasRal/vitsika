import type { Metadata } from "next";
import { api, safe } from "@/lib/api";
import { ApiDown } from "@/components/ui";
import { GeneraTable } from "./GeneraTable";

// Render per request: the API data (fetch-cached with revalidate) must never be
// frozen into a static build — e.g. the "service not reachable" banner.
export const dynamic = "force-dynamic";

export const metadata: Metadata = { title: "Genera" };

export default async function GeneraPage() {
  const { data, error } = await safe(api.genera());
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-3xl">The {data?.n ?? 27} genera the model knows</h1>
        <p className="mt-1 max-w-prose text-sm text-ink-2">
          Every Malagasy genus with at least 10 worker profile images on AntWeb. F1 is the linear probe’s score on
          the held-out test images; with 3–6 test images per small genus, a single miss moves it a lot — hence the
          reliability column. Click a row for its collection map.
        </p>
      </div>
      {error ? <ApiDown error={error} /> : <GeneraTable genera={data!.genera} />}
    </div>
  );
}
