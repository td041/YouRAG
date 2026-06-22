const QUERY_KEY = "yourag_query_perf_v1";
const INGEST_KEY = "yourag_ingest_perf_v1";
const MAX = 30;

export interface QueryPerf {
  ts: number;
  query: string;
  latency: {
    search?: number;
    graph?: number;
    rerank?: number;
    ttft?: number;
    llm?: number;
    cache?: number;
    total?: number;
  };
  cached: boolean;
}

export interface IngestPerf {
  ts: number;
  title: string;
  collection: string;
  latency: {
    extract_s?: number;
    chunk_s?: number;
    graph_s?: number;
    embed_s?: number;
    load_s?: number;
    total_s?: number;
  };
  chunks: number;
}

function read<T>(key: string): T[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(key) ?? "[]"); } catch { return []; }
}

function write<T>(key: string, items: T[]): void {
  try { localStorage.setItem(key, JSON.stringify(items.slice(0, MAX))); } catch {}
}

export function recordQuery(entry: Omit<QueryPerf, "ts">): void {
  const items = read<QueryPerf>(QUERY_KEY);
  items.unshift({ ...entry, ts: Date.now() });
  write(QUERY_KEY, items);
}

export function recordIngest(entry: Omit<IngestPerf, "ts">): void {
  const items = read<IngestPerf>(INGEST_KEY);
  items.unshift({ ...entry, ts: Date.now() });
  write(INGEST_KEY, items);
}

export function getQueryPerf(): QueryPerf[] { return read<QueryPerf>(QUERY_KEY); }
export function getIngestPerf(): IngestPerf[] { return read<IngestPerf>(INGEST_KEY); }
export function clearPerf(): void {
  try { localStorage.removeItem(QUERY_KEY); localStorage.removeItem(INGEST_KEY); } catch {}
}
