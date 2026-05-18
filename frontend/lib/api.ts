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

export async function deleteCollection(name: string) {
  const r = await fetch(`${BASE}/collections/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function streamChat(
  query: string,
  collection: string,
  sessionId: string | undefined,
  onChunk: (text: string) => void,
  onSources: (sources: string[]) => void,
  onSuggestions: (suggestions: string[]) => void,
  onSessionId: (id: string) => void,
  onDone: () => void,
  onError: (e: string) => void,
) {
  const ctrl = new AbortController();

  fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, collection, session_id: sessionId }),
    signal: ctrl.signal,
  })
    .then(async (r) => {
      if (!r.ok) { onError(await r.text()); return; }
      const reader = r.body!.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let textContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        buf += chunk;

        if (!buf.includes("__SOURCES__::") && !buf.includes("__SESSION__::")) {
          onChunk(chunk);
          textContent += chunk;
        } else if (!textContent.includes("__SOURCES__::")) {
          const splitTag = buf.includes("__SOURCES__::") ? "__SOURCES__::" : "__SESSION__::";
          const textBeforeMeta = buf.split(splitTag)[0];
          const remainingText = textBeforeMeta.slice(textContent.length);
          if (remainingText) {
             onChunk(remainingText);
             textContent += remainingText;
          }
        }
      }

      if (buf.includes("__SESSION__::")) {
        const sessionPart = buf.split("__SESSION__::")[1].split("\n\n__")[0];
        onSessionId(sessionPart.trim());
      }

      if (buf.includes("__SOURCES__::")) {
        const metaPart = buf.split("__SOURCES__::")[1];
        const srcStr = metaPart.split("\n\n__")[0];
        const sources = srcStr.split(",").map((s) => s.trim()).filter(Boolean);
        onSources(sources);
      }
      
      if (buf.includes("__SUGGESTIONS__::")) {
        const sugPart = buf.split("__SUGGESTIONS__::")[1].split("\n\n__")[0];
        const suggestions = sugPart.split("|").map((s) => s.trim()).filter(Boolean);
        onSuggestions(suggestions);
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
