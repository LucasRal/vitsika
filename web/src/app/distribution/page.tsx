import type { Metadata } from "next";
import { Suspense } from "react";
import { api, safe } from "@/lib/api";
import { ApiDown, Spinner } from "@/components/ui";
import { DistributionClient } from "./DistributionClient";

// Render per request: the API data (fetch-cached with revalidate) must never be
// frozen into a static build, e.g. the "service not reachable" banner.
export const dynamic = "force-dynamic";

export const metadata: Metadata = { title: "Distribution" };

export default async function DistributionPage() {
  const { data, error } = await safe(api.genera());
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-3xl">Where each genus has been collected</h1>
        <p className="mt-1 max-w-prose text-sm text-ink-2">
          Georeferenced AntWeb worker specimens from the full 4,354-specimen manifest (not only the images used
          for training), by genus. Provinces use current names (Majunga → Mahajanga, Toliary → Toliara, Diego-Suarez → Antsiranana).
        </p>
      </div>
      {error ? <ApiDown error={error} /> : (
        <Suspense fallback={<Spinner />}>
          <DistributionClient genera={data!.genera} />
        </Suspense>
      )}
    </div>
  );
}
