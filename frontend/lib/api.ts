const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export async function fetchCollections() {
  const r = await fetch(`${BASE}/collections`, { cache: "no-store" });
  if (!r.ok) throw new Error("API offline");
  return r.json();
}

export async function ingestVideo(url: string, useContextual: boolean) {
  const r = await fetch(`${BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, use_contextual: useContextual }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function streamChat(
  query: string,
  collection: string,
  onChunk: (text: string) => void,
  onSources: (sources: string[]) => void,
  onDone: () => void,
  onError: (e: string) => void,
) {
  const ctrl = new AbortController();

  fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, collection }),
    signal: ctrl.signal,
  })
    .then(async (r) => {
      if (!r.ok) { onError(await r.text()); return; }
      const reader = r.body!.getReader();
      const dec = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        if (buf.includes("__SOURCES__::")) {
          const [text, rest] = buf.split("__SOURCES__::");
          onChunk(text);
          const srcPart = rest.split("\n\n__FACTS__::")[0];
          const sources = srcPart.split(",").map((s) => s.trim()).filter(Boolean);
          onSources(sources);
          buf = "";
        } else if (!buf.includes("__FACTS__::")) {
          onChunk(buf);
          buf = "";
        }
      }
      onDone();
    })
    .catch((e) => { if (e.name !== "AbortError") onError(String(e)); });

  return () => ctrl.abort();
}

export function streamSummary(
  collection: string,
  onChunk: (text: string) => void,
  onDone: () => void,
  onError: (e: string) => void,
) {
  const ctrl = new AbortController();

  fetch(`${BASE}/summarize/stream/${collection}`, { signal: ctrl.signal })
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
