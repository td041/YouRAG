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
    if (job.status === "done") return job.result;
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
      const META_SENTINEL = "\n\n__META__";
      let buf = "";
      let textContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        buf += chunk;

        const metaIdx = buf.indexOf(META_SENTINEL);
        if (metaIdx === -1) {
          // No meta frame yet — stream everything as text
          onChunk(chunk);
          textContent += chunk;
        } else if (textContent.length < metaIdx) {
          // Meta frame arrived mid-chunk — flush remaining text before it
          const remaining = buf.slice(textContent.length, metaIdx);
          if (remaining) { onChunk(remaining); textContent += remaining; }
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
