"use client";

import { useState } from "react";
import { Collection } from "@/lib/types";
import { ingestVideo, deleteCollection } from "@/lib/api";
import {
  Plus, Library, Loader2, CheckCircle2,
  XCircle, Trash2,
  Search, Sparkles, Globe
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
  onDeleted: (name: string) => void;
}

export default function Sidebar({ collections, selected, apiOnline, onSelect, onIngested, onDeleted }: Props) {
  const [url, setUrl] = useState("");
  const [useCtx, setUseCtx] = useState(false);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [deletingName, setDeletingName] = useState<string | null>(null);

  async function handleDelete(e: React.MouseEvent, name: string) {
    e.stopPropagation();
    if (!confirm(`Xóa video "${name}" khỏi thư viện?`)) return;
    setDeletingName(name);
    try {
      await deleteCollection(name);
      onDeleted(name);
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingName(null);
    }
  }

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setStatus("loading");
    setStatusMsg("");
    try {
      const d = await ingestVideo(url.trim(), useCtx);
      setStatus("ok");
      setStatusMsg(`${d.chunks_added ?? "?"} chunks indexed`);
      setUrl("");
      onIngested();
      setTimeout(() => setStatus("idle"), 3000);
    } catch (err: unknown) {
      setStatus("error");
      setStatusMsg(err instanceof Error ? err.message.slice(0, 80) : String(err).slice(0, 80));
    }
  }

  return (
    <aside className="flex flex-col h-full w-[280px] shrink-0 border-r border-white/[0.06] bg-[#050608] z-20 overflow-hidden">
      
      {/* Brand & Status */}
      <div className="flex flex-col gap-4 px-6 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center">
              <YouTubeLogo size={20} />
            </div>
            <h1 className="text-lg font-bold text-white font-display tracking-tight">YouRAG</h1>
          </div>
          <div className={`flex items-center justify-center w-5 h-5 rounded-full border transition-colors ${
            apiOnline ? "border-emerald-500/20 bg-emerald-500/10" : "border-rose-500/20 bg-rose-500/10"
          }`} title={apiOnline ? "API Online" : "API Offline"}>
            {apiOnline ? (
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
            ) : (
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
            )}
          </div>
        </div>
      </div>

      {/* Ingest Section */}
      <div className="px-5 mb-8">
        <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-4 shadow-2xl">
          <div className="flex items-center gap-2 mb-3 px-1">
            <Plus size={14} className="text-indigo-400" />
            <span className="text-[11px] font-bold uppercase tracking-widest text-slate-500">New Video</span>
          </div>
          <form onSubmit={handleIngest} className="space-y-3">
            <div className="relative group">
              <input
                type="text" value={url} onChange={e => setUrl(e.target.value)}
                placeholder="YouTube URL..."
                className="w-full bg-[#0a0c10] border border-white/[0.08] rounded-xl px-3 py-2.5 text-[13px] text-white placeholder-slate-600 outline-none focus:border-indigo-500/50 focus:ring-4 focus:ring-indigo-500/5 transition-all"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-700">
                <Search size={14} />
              </div>
            </div>

            <div 
              onClick={() => setUseCtx(v => !v)}
              className="flex items-center justify-between px-1 py-1 cursor-pointer group select-none"
            >
              <div className="flex items-center gap-2">
                <Sparkles size={12} className={useCtx ? "text-indigo-400" : "text-slate-600"} />
                <span className={`text-[11px] font-medium transition-colors ${useCtx ? "text-slate-300" : "text-slate-600"}`}>AI Contextualizer</span>
              </div>
              <div className={`w-7 h-4 rounded-full transition-all relative ${useCtx ? "bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.3)]" : "bg-slate-800"}`}>
                <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${useCtx ? "left-[14px]" : "left-0.5"}`} />
              </div>
            </div>

            <button
              type="submit" disabled={status === "loading" || !url.trim()}
              className="w-full h-10 accent-gradient text-white text-[13px] font-bold rounded-xl shadow-lg shadow-indigo-500/10 active:scale-95 transition-all disabled:opacity-30 disabled:grayscale disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {status === "loading" ? (
                <><Loader2 size={16} className="animate-spin" /><span>Indexing...</span></>
              ) : (
                <>Add to Library</>
              )}
            </button>

            {status !== "idle" && (
              <div className={`flex items-center gap-2 text-[11px] px-1 animate-fade-in ${
                status === "ok" ? "text-emerald-400" : "text-rose-400"
              }`}>
                {status === "ok" ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                <span className="truncate">{statusMsg}</span>
              </div>
            )}
          </form>
        </div>
      </div>

      {/* Library Section */}
      <div className="flex-1 flex flex-col min-h-0 px-2 overflow-hidden">
        <div className="flex items-center justify-between px-4 mb-4">
          <div className="flex items-center gap-2 text-slate-500">
            <Library size={14} />
            <span className="text-[11px] font-bold uppercase tracking-widest">Library</span>
          </div>
          {collections.length > 0 && (
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-slate-500">
              {collections.length}
            </span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar px-2 space-y-1">
          {collections.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <Globe size={32} className="mx-auto text-slate-800 mb-3" />
              <p className="text-[12px] text-slate-600">Your video library is empty</p>
            </div>
          ) : (
            collections.map(c => (
              <button
                key={c.name}
                onClick={() => onSelect(c)}
                className={`w-full group flex flex-col gap-2 p-3 rounded-2xl transition-all duration-300 border ${
                  selected?.name === c.name
                    ? "bg-white/[0.04] border-white/[0.08] shadow-[0_4px_20px_rgba(0,0,0,0.4)]"
                    : "border-transparent hover:bg-white/[0.02] text-slate-500 hover:text-slate-300"
                }`}
              >
                <div className="flex items-center gap-3 w-full">
                  <div className="w-10 h-10 rounded-xl overflow-hidden bg-slate-900 shrink-0 border border-white/[0.05] relative group-hover:border-white/[0.1] transition-all">
                    {c.video_id ? (
                      <img 
                        src={`https://i.ytimg.com/vi/${c.video_id}/default.jpg`} 
                        alt="" 
                        className={`w-full h-full object-cover transition-transform duration-500 ${selected?.name === c.name ? "scale-110" : "group-hover:scale-110"}`}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Library size={14} />
                      </div>
                    )}
                    {selected?.name === c.name && (
                      <div className="absolute inset-0 bg-indigo-500/20 backdrop-blur-[1px]" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0 text-left">
                    <p className={`text-[12.5px] font-semibold leading-snug truncate transition-colors ${
                      selected?.name === c.name ? "text-white" : "text-slate-400 group-hover:text-slate-200"
                    }`}>
                      {c.title}
                    </p>
                    <p className="text-[10px] text-slate-600 mt-0.5 truncate">
                      Source: YouTube
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, c.name)}
                    disabled={deletingName === c.name}
                    className="shrink-0 opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-rose-500/10 text-slate-600 hover:text-rose-400 transition-all"
                    title="Xóa video"
                  >
                    {deletingName === c.name
                      ? <Loader2 size={13} className="animate-spin" />
                      : <Trash2 size={13} />}
                  </button>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Modern Footer Info */}
      <div className="px-6 py-6 border-t border-white/[0.04] bg-[#030406]">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
          <span className="text-[10px] font-bold text-slate-600 uppercase tracking-tighter">System Engine</span>
        </div>
        <p className="text-[9px] font-mono text-slate-700 leading-tight">
          BGE-M3 / HYBRID-SEARCH / LLAMA-3.3-70B
        </p>
      </div>
    </aside>
  );
}

