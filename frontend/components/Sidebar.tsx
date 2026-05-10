"use client";

import { useState } from "react";
import { Collection } from "@/lib/types";
import { ingestVideo } from "@/lib/api";
import {
  Plus, Library, Loader2, CheckCircle2,
  XCircle, ChevronRight, Wifi, WifiOff
} from "lucide-react";

function YouTubeLogo({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size * 0.7} viewBox="0 0 90 63" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="90" height="63" rx="13" fill="#FF0000"/>
      <path d="M36 44V19L63 31.5L36 44Z" fill="white"/>
    </svg>
  );
}

interface Props {
  collections: Collection[];
  selected: Collection | null;
  apiOnline: boolean;
  onSelect: (c: Collection) => void;
  onIngested: () => void;
}

export default function Sidebar({ collections, selected, apiOnline, onSelect, onIngested }: Props) {
  const [url, setUrl] = useState("");
  const [useCtx, setUseCtx] = useState(false);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setStatus("loading");
    setStatusMsg("");
    try {
      const d = await ingestVideo(url.trim(), useCtx);
      setStatus("ok");
      setStatusMsg(`${d.chunks_added ?? "?"} chunks · ${d.latency?.total_s?.toFixed(1) ?? "?"}s`);
      setUrl("");
      onIngested();
    } catch (err: unknown) {
      setStatus("error");
      setStatusMsg(err instanceof Error ? err.message.slice(0, 80) : String(err).slice(0, 80));
    }
  }

  return (
    <aside className="flex flex-col h-full w-[260px] shrink-0 border-r border-white/[.07] bg-[#0d1017]">

      {/* Brand */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-white/[.07]">
        <div className="shrink-0">
          <YouTubeLogo size={32} />
        </div>
        <div>
          <p className="text-sm font-semibold text-white leading-tight tracking-tight">YouRAG</p>
          <p className="text-[10px] text-slate-500 mt-0.5">YouTube Intelligence</p>
        </div>
      </div>

      {/* API status */}
      <div className="mx-4 mt-3">
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-[11px] font-medium border
          ${apiOnline
            ? "bg-emerald-500/[.07] border-emerald-500/20 text-emerald-400"
            : "bg-red-500/[.07] border-red-500/20 text-red-400"}`}>
          {apiOnline
            ? <><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-dot" /><Wifi size={11} /><span>API connected</span></>
            : <><span className="w-1.5 h-1.5 rounded-full bg-red-400" /><WifiOff size={11} /><span>API offline</span></>}
        </div>
      </div>

      {/* Ingest form */}
      <div className="px-4 mt-5">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600 mb-2.5">Add Video</p>
        <form onSubmit={handleIngest} className="space-y-2">
          <input
            type="text" value={url} onChange={e => setUrl(e.target.value)}
            placeholder="https://youtube.com/watch?v=…"
            className="w-full bg-[#12151b] border border-white/[.07] rounded-lg px-3 py-2.5 text-[13px] text-slate-200 placeholder-slate-600 outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-all"
          />
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <div
              onClick={() => setUseCtx(v => !v)}
              className={`w-8 h-4 rounded-full transition-colors relative cursor-pointer ${useCtx ? "bg-indigo-500" : "bg-slate-700"}`}
            >
              <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${useCtx ? "translate-x-4" : "translate-x-0.5"}`} />
            </div>
            <span className="text-[11px] text-slate-500">Contextual Enrichment</span>
          </label>
          <button
            type="submit" disabled={status === "loading" || !url.trim()}
            className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-[13px] font-semibold rounded-lg py-2.5 transition-colors"
          >
            {status === "loading"
              ? <><Loader2 size={13} className="spin" /><span>Ingesting…</span></>
              : <><Plus size={13} /><span>Add to Library</span></>}
          </button>

          {status === "ok" && (
            <div className="flex items-center gap-1.5 text-[11px] text-emerald-400">
              <CheckCircle2 size={12} /><span>{statusMsg}</span>
            </div>
          )}
          {status === "error" && (
            <div className="flex items-center gap-1.5 text-[11px] text-red-400">
              <XCircle size={12} /><span>{statusMsg}</span>
            </div>
          )}
        </form>
      </div>

      {/* Library */}
      <div className="mt-5 flex-1 overflow-y-auto">
        <div className="flex items-center gap-2 px-4 mb-2">
          <Library size={11} className="text-slate-600" />
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">Library</p>
          {collections.length > 0 && (
            <span className="ml-auto text-[10px] font-mono text-slate-600 bg-white/[.04] px-1.5 py-0.5 rounded">
              {collections.length}
            </span>
          )}
        </div>

        {collections.length === 0 ? (
          <p className="px-4 text-[12px] text-slate-600 leading-relaxed">
            No videos yet. Add a YouTube URL above.
          </p>
        ) : (
          <ul className="space-y-0.5 px-2">
            {collections.map(c => (
              <li key={c.name}>
                <button
                  onClick={() => onSelect(c)}
                  className={`w-full text-left flex items-center gap-2.5 px-3 py-2.5 rounded-lg transition-all group
                    ${selected?.name === c.name
                      ? "bg-indigo-500/10 border border-indigo-500/20 text-indigo-300"
                      : "hover:bg-white/[.04] border border-transparent text-slate-400 hover:text-slate-200"}`}
                >
                  <div className={`w-5 h-5 rounded shrink-0 bg-cover bg-center bg-[#1e2433] flex items-center justify-center`}>
                    {c.video_id
                      ? <img src={`https://i.ytimg.com/vi/${c.video_id}/default.jpg`} alt="" className="w-5 h-5 rounded object-cover" />
                      : <span className="text-[8px] text-slate-500">▶</span>}
                  </div>
                  <span className="text-[12px] font-medium leading-tight flex-1 truncate">
                    {c.title}
                  </span>
                  <ChevronRight size={12} className={`shrink-0 transition-opacity ${selected?.name === c.name ? "opacity-60" : "opacity-0 group-hover:opacity-30"}`} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/[.07]">
        <p className="text-[10px] text-slate-700 font-mono">bge-m3 · RRF · llama-3.3-70b</p>
      </div>
    </aside>
  );
}
