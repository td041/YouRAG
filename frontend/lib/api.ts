const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  return API_KEY ? { "X-API-Key": API_KEY, ...extra } : { ...extra };
}

export async function fetchCollections() {
  const r = await fetch(`${BASE}/collections`, { cache: "no-store" });
  if (!r.ok) throw new Error("API offline");
  return r.json();
}

export async function ingestVideo(
  url: string,
  useContextual: boolean,
  useLateChunking = false,
  useVisualRag = false,
  onProgress?: (status: string) => void,
) {
  // Kick off async job
  const r = await fetch(`${BASE}/ingest`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      url,
      use_contextual: useContextual,
      use_late_chunking: useLateChunking,
      use_visual_rag: useVisualRag,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  const { job_id } = await r.json();

  // Poll until done or error
  while (true) {
    await new Promise((res) => setTimeout(res, 2000));
    const s = await fetch(`${BASE}/ingest/status/${job_id}`);
    if (!s.ok) throw new Error("Không thể kiểm tra trạng thái ingest");
    const job = await s.json();
    onProgress?.(job.status);
    if (job.status === "done") {
      const result = job.result;
      if (result?.latency) {
        const { recordIngest } = await import("./perf-store");
        recordIngest({
          title: result.title ?? "",
          collection: result.collection_name ?? "",
          latency: result.latency,
          chunks: result.chunks_added ?? 0,
        });
      }
      return result;
    }
    if (job.status === "error") throw new Error(job.error ?? "Ingest thất bại");
  }
}

export async function deleteCollection(name: string) {
  const r = await fetch(`${BASE}/collections/${encodeURIComponent(name)}`, { method: "DELETE", headers: authHeaders() });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchBenchmarkReport() {
  const r = await fetch(`${BASE}/benchmark/report`, { cache: "no-store", headers: authHeaders() });
  if (!r.ok) return null;
  return r.json();
}

export async function fetchSuggestions(collection: string): Promise<string[]> {
  const r = await fetch(`${BASE}/suggestions/${encodeURIComponent(collection)}`, {
    headers: authHeaders(),
    cache: "no-store",
  });
  if (!r.ok) return [];
  const data = await r.json();
  return data.suggestions ?? [];
}

const PROGRESS_PREFIX = "__PROGRESS__";
const META_SENTINEL = "\n\n__META__";

export function streamChat(
  query: string,
  collections: string[],
  sessionId: string | undefined,
  onChunk: (text: string) => void,
  onSources: (sources: unknown[]) => void,
  onSuggestions: (suggestions: string[]) => void,
  onSessionId: (id: string) => void,
  onDone: () => void,
  onError: (e: string) => void,
  onProgress?: (msg: string) => void,
  onLatency?: (latency: Record<string, number>, cached: boolean) => void,
) {
  const ctrl = new AbortController();

  fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ query, collections, session_id: sessionId }),
    signal: ctrl.signal,
  })
    .then(async (r) => {
      if (!r.ok) { onError(await r.text()); return; }
      const reader = r.body!.getReader();
      const dec = new TextDecoder();
      let buf = "";        // full accumulated buffer (for meta detection at end)
      let textContent = ""; // accumulated display text (excludes PROGRESS lines)
      let lineBuf = "";    // incomplete line fragment (for PROGRESS line detection)
      let progressDone = false; // true after first non-PROGRESS content arrives

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const raw = dec.decode(value, { stream: true });
        buf += raw;

        if (!progressDone) {
          // PROGRESS phase: scan line-by-line until first LLM text arrives
          lineBuf += raw;
          let nlIdx: number;
          while ((nlIdx = lineBuf.indexOf("\n")) !== -1) {
            const line = lineBuf.slice(0, nlIdx + 1); // includes \n
            lineBuf = lineBuf.slice(nlIdx + 1);
            if (line.startsWith(PROGRESS_PREFIX)) {
              onProgress?.(line.slice(PROGRESS_PREFIX.length).trim());
            } else {
              // First non-PROGRESS complete line → switch to pass-through mode
              progressDone = true;
              onChunk(line);
              textContent += line;
              break;
            }
          }
          // Flush any remaining lineBuf as LLM content (now in pass-through).
          // Even when META is already present in buf, flush everything before
          // the sentinel so we don't silently drop the first LLM sentence.
          if (progressDone && lineBuf) {
            const metaPos = buf.indexOf(META_SENTINEL);
            const bufOffset = buf.length - lineBuf.length; // where lineBuf starts in buf
            const flushEnd = metaPos !== -1 ? Math.max(0, metaPos - bufOffset) : lineBuf.length;
            const toFlush = lineBuf.slice(0, flushEnd);
            if (toFlush) { onChunk(toFlush); textContent += toFlush; }
            lineBuf = "";
          }
        } else {
          // Pass-through phase: stream directly to onChunk (old behaviour)
          const metaIdx = buf.indexOf(META_SENTINEL);
          if (metaIdx === -1) {
            onChunk(raw);
            textContent += raw;
          } else if (textContent.length < metaIdx) {
            const remaining = buf.slice(textContent.length, metaIdx);
            if (remaining) { onChunk(remaining); textContent += remaining; }
          }
        }
      }

      // Parse single JSON meta frame
      const metaIdx = buf.indexOf(META_SENTINEL);
      if (metaIdx !== -1) {
        try {
          const metaStr = buf.slice(metaIdx + META_SENTINEL.length);
          const meta = JSON.parse(metaStr);
          onSources(meta.sources ?? []);
          onSuggestions(meta.suggestions ?? []);
          onSessionId(meta.session_id ?? "");
          if (meta.latency_ms && onLatency) {
            onLatency(meta.latency_ms, meta.cached ?? false);
          }
        } catch (e) {
          console.warn("Failed to parse stream meta frame", e);
        }
      }

      onDone();
    })
    .catch((e) => { if (e.name !== "AbortError") onError(String(e)); });

  return () => ctrl.abort();
}

export async function fetchQuiz(collection: string, count = 5, mode: "quiz" | "flashcard" = "quiz") {
  const r = await fetch(
    `${BASE}/quiz/${encodeURIComponent(collection)}?count=${count}&mode=${mode}`,
    { cache: "no-store", headers: authHeaders() },
  );
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function streamSummary(
  collection: string,
  onChunk: (text: string) => void,
  onDone: () => void,
  onError: (e: string) => void,
) {
  const ctrl = new AbortController();

  fetch(`${BASE}/summarize/stream/${collection}`, { signal: ctrl.signal, headers: authHeaders() })
    .then(async (r) => {
      if (!r.ok) { onError(await r.text()); return; }
      const reader = r.body!.getReader();
      const dec = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        onChunk(dec.decode(value, { stream: true }));
      }
      onDone();
    })
    .catch((e) => { if (e.name !== "AbortError") onError(String(e)); });

  return () => ctrl.abort();
}
