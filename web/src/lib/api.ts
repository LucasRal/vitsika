/** Typed client for the mg-ants FastAPI backend (api/schemas.py).
 *
 * The base URL comes from NEXT_PUBLIC_API_URL (browser + server). When the
 * public value is relative (e.g. "/api" behind nginx), server components use
 * API_URL instead, which must be absolute. */

export type GenusPrediction = { genus: string; subfamily: string; probability: number };
export type SimilarSpecimen = {
  specimen_code: string; genus: string; subfamily: string; species: string | null;
  similarity: number; image_url: string; antweb_url: string;
  photographer: string | null; license: string | null; locality: string | null;
};
export type AtlasPosition = { x: number; y: number };
export type AnalyzeResponse = {
  predictions: GenusPrediction[]; similar: SimilarSpecimen[];
  atlas_position: AtlasPosition | null; model_name: string; probe_version: string;
  embed_ms: number; filename: string | null;
};
export type GenusInfo = {
  name: string; subfamily: string; n_train: number; n_test: number; f1_probe: number;
  atlas_median: AtlasPosition;
};
export type GeneraResponse = { genera: GenusInfo[]; n: number };
export type ExampleSpecimen = {
  specimen_code: string; genus: string; subfamily: string; species: string | null;
  photographer: string | null; license: string | null; image_url: string; antweb_url: string;
};
export type ExamplesResponse = { examples: ExampleSpecimen[]; n: number; n_test_total: number; per_genus: number | null };
export type ElevationStats = { min: number | null; median: number | null; max: number | null; n: number };
export type GeoResponse = {
  genus: string; subfamily: string; n_specimens: number; n_species: number; n_unidentified: number;
  provinces: Record<string, number>; elevation: ElevationStats;
  year_min: number | null; year_max: number | null; n_with_coords: number;
  points: [number, number][]; points_capped: boolean;
};
export type AtlasPoint = {
  specimen_code: string; x: number; y: number; genus: string; subfamily: string;
  species: string | null; image_available: boolean;
};
export type AtlasResponse = { points: AtlasPoint[]; n: number; umap: Record<string, string | number> };
export type HealthResponse = {
  status: string; model_loaded: boolean; n_embeddings: number; n_train: number; n_genera: number;
  versions: Record<string, string>;
};

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export function apiBase(): string {
  const pub = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";
  if (typeof window === "undefined" && process.env.API_URL) return process.env.API_URL;
  return pub.replace(/\/$/, "");
}

export const imageUrl = (code: string) => `${apiBase()}/images/${encodeURIComponent(code)}`;
export const antwebUrl = (code: string) => `https://www.antweb.org/specimen/${encodeURIComponent(code)}`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${apiBase()}${path}`, init);
  } catch (e) {
    throw new ApiError(0, `cannot reach the API at ${apiBase()} (${(e as Error).message})`);
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try { message = (await res.json()).message ?? message; } catch { /* non-JSON body */ }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

const revalidate = (seconds: number): RequestInit => ({ next: { revalidate: seconds } } as RequestInit);

export const api = {
  health: () => request<HealthResponse>("/health", { cache: "no-store" }),
  genera: () => request<GeneraResponse>("/genera", revalidate(300)),
  geo: (genus: string) => request<GeoResponse>(`/geo/${encodeURIComponent(genus)}`, revalidate(300)),
  atlas: () => request<AtlasResponse>("/atlas", revalidate(3600)),
  examples: (perGenus = 2) => request<ExamplesResponse>(`/examples?per_genus=${perGenus}`, revalidate(3600)),
  analyze: (file: File) => {
    const body = new FormData();
    body.append("file", file, file.name);
    return request<AnalyzeResponse>("/analyze", { method: "POST", body });
  },
};

/** Fetch for server components: never throws, returns the error message instead. */
export async function safe<T>(p: Promise<T>): Promise<{ data: T; error: null } | { data: null; error: string }> {
  try { return { data: await p, error: null }; }
  catch (e) { return { data: null, error: e instanceof Error ? e.message : String(e) }; }
}
