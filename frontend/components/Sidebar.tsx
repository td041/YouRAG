"use client";

import { useState } from "react";
import { Collection } from "@/lib/types";
import { ingestVideo, deleteCollection } from "@/lib/api";
import {
  Plus, Library, Loader2, CheckCircle2,
  XCircle, Trash2, Search, Sparkles, Globe, X
} from "lucide-react";

function YouTubeLogo({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size * 0.7} viewBox="0 0 90 63" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="90" height="63" rx="13" fill="#FF0000"/>
      <path d="M36 44V19L63 31.5L36 44Z" fill="white"/>
    </svg>
  );
}

type IngestStatus = "idle" | "queued" | "running" | "ok" | "error";

const STEPS: { key: IngestStatus; label: string }[] = [
  { key: "queued",  label: "Queued" },
  { key: "running", label: "Processing" },
  { key: "ok",      label: "Done" },
];

function stepIndex(s: IngestStatus) {
  return STEPS.findIndex(x => x.key === s);
}

interface Props {
  collections: Collection[];
  selected: Collection | null;
  apiOnline: boolean;
  theme: "dark" | "light";
  onSelect: (c: Collection) => void;
  onIngested: () => void;
  onDeleted: (name: string) => void;
  onClose: () => void;
}

export default function Sidebar({ collections, selected, apiOnline, theme, onSelect, onIngested, onDeleted, onClose }: Props) {
  const [url, setUrl] = useState("");
  const [useCtx, setUseCtx] = useState(false);
  const [status, setStatus] = useState<IngestStatus>("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [deletingName, setDeletingName] = useState<string | null>(null);
  const [chunks, setChunks] = useState<number | null>(null);

  const isDark = theme === "dark";

  async function handleDelete(e: React.MouseEvent, name: string) {
    e.stopPropagation();
    if (!confirm(`Delete "${name}" from library?`)) return;
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
    setStatus("queued");
    setStatusMsg("");
    setChunks(null);
    try {
      const d = await ingestVideo(url.trim(), useCtx, false, (s) => {
        if (s === "running") setStatus("running");
      });
      setStatus("ok");
      setChunks(d?.chunks_added ?? null);
      setUrl("");
      onIngested();
      setTimeout(() => { setStatus("idle"); setChunks(null); }, 4000);
    } catch (err: unknown) {
      setStatus("error");
      setStatusMsg(err instanceof Error ? err.message.slice(0, 80) : String(err).slice(0, 80));
    }
  }

  const borderColor = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.07)";
  const cardBg = isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)";
  const inputBg = isDark ? "#0a0c10" : "#ffffff";
  const inputBorder = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.1)";
  const textMain = isDark ? "#f0f2f5" : "#0f172a";
  const textMuted = isDark ? "#94a3b8" : "#475569";
  const textDim = isDark ? "#475569" : "#94a3b8";
  const sidebarBg = isDark ? "#050608" : "#f1f3f8";
  const footerBg = isDark ? "#030406" : "#e8eaf0";
  const selectedItemBg = isDark ? "rgba(255,255,255,0.04)" : "rgba(99,102,241,0.06)";
  const selectedItemBorder = isDark ? "rgba(255,255,255,0.08)" : "rgba(99,102,241,0.2)";

  const isIngesting = status === "queued" || status === "running";
  const currentStep = stepIndex(status);

  return (
    <aside className="flex flex-col h-full w-[280px] shrink-0 z-20 overflow-hidden" style={{ background: sidebarBg, borderRight: `1px solid ${borderColor}` }}>

      {/* Brand & Status */}
      <div className="flex flex-col gap-4 px-6 py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "rgba(255,255,255,0.05)", border: `1px solid ${borderColor}` }}>
              <YouTubeLogo size={20} />
            </div>
            <h1 className="text-lg font-bold font-display tracking-tight" style={{ color: textMain }}>YouRAG</h1>
          </div>
          <div className="flex items-center gap-2">
            <div className={`flex items-center justify-center w-5 h-5 rounded-full border transition-colors ${
              apiOnline ? "border-emerald-500/20 bg-emerald-500/10" : "border-rose-500/20 bg-rose-500/10"
            }`} title={apiOnline ? "API Online" : "API Offline"}>
              {apiOnline
                ? <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
                : <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />}
            </div>
            {/* Mobile close button */}
            <button
              onClick={onClose}
              className="lg:hidden p-1.5 rounded-lg hover:bg-white/5 transition-colors"
              style={{ color: textDim }}
              aria-label="Close sidebar"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Ingest Section */}
      <div className="px-5 mb-6">
        <div className="rounded-2xl p-4 shadow-2xl" style={{ background: cardBg, border: `1px solid ${borderColor}` }}>
          <div className="flex items-center gap-2 mb-3 px-1">
            <Plus size={14} className="text-indigo-400" />
            <span className="text-[11px] font-bold uppercase tracking-widest" style={{ color: textDim }}>New Video</span>
          </div>
          <form onSubmit={handleIngest} className="space-y-3">
            <div className="relative">
              <input
                type="text" value={url} onChange={e => setUrl(e.target.value)}
                placeholder="YouTube URL..."
                className="w-full rounded-xl px-3 py-2.5 text-[13px] outline-none transition-all focus:ring-4 focus:ring-indigo-500/5"
                style={{
                  background: inputBg,
                  border: `1px solid ${inputBorder}`,
                  color: textMain,
                }}
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2" style={{ color: textDim }}>
                <Search size={14} />
              </div>
            </div>

            <div
              onClick={() => setUseCtx(v => !v)}
              className="flex items-center justify-between px-1 py-1 cursor-pointer select-none"
            >
              <div className="flex items-center gap-2">
                <Sparkles size={12} className={useCtx ? "text-indigo-400" : ""} style={!useCtx ? { color: textDim } : {}} />
                <span className="text-[11px] font-medium transition-colors" style={{ color: useCtx ? textMuted : textDim }}>
                  AI Contextualizer
                </span>
              </div>
              <div className={`w-7 h-4 rounded-full transition-all relative ${useCtx ? "bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.3)]" : ""}`}
                   style={!useCtx ? { background: isDark ? "#1e293b" : "#cbd5e1" } : {}}>
                <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${useCtx ? "left-[14px]" : "left-0.5"}`} />
              </div>
            </div>

            <button
              type="submit" disabled={isIngesting || !url.trim()}
              className="w-full h-10 accent-gradient text-white text-[13px] font-bold rounded-xl shadow-lg shadow-indigo-500/10 active:scale-95 transition-all disabled:opacity-30 disabled:grayscale disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isIngesting
                ? <><Loader2 size={16} className="animate-spin" /><span>Indexing...</span></>
                : <>Add to Library</>
              }
            </button>

            {/* Progress bar */}
            {isIngesting && (
              <div className="space-y-1.5 animate-fade-in">
                <div className="flex items-center justify-between px-0.5">
                  <span className="text-[10px] font-medium text-indigo-400 progress-pulse">
                    {status === "queued" ? "Queued..." : "Processing..."}
                  </span>
                  <span className="text-[10px]" style={{ color: textDim }}>
                    {status === "queued" ? "25%" : "70%"}
                  </span>
                </div>
                <div className="w-full rounded-full h-1 overflow-hidden" style={{ background: isDark ? "#1e293b" : "#e2e8f0" }}>
                  <div
                    className="h-full accent-gradient rounded-full transition-all duration-700"
                    style={{ width: status === "queued" ? "25%" : "70%" }}
                  />
                </div>
              </div>
            )}

            {/* Done / Error feedback */}
            {status === "ok" && (
              <div className="flex items-center gap-2 text-[11px] px-1 animate-fade-in text-emerald-400">
                <CheckCircle2 size={12} />
                <span>{chunks != null ? `${chunks} chunks indexed` : "Done!"}</span>
              </div>
            )}
            {status === "error" && (
              <div className="flex items-center gap-2 text-[11px] px-1 animate-fade-in text-rose-400">
                <XCircle size={12} />
                <span className="truncate">{statusMsg || "Ingest failed"}</span>
              </div>
            )}
          </form>
        </div>
      </div>

      {/* Library Section */}
      <div className="flex-1 flex flex-col min-h-0 px-2 overflow-hidden">
        <div className="flex items-center justify-between px-4 mb-3">
          <div className="flex items-center gap-2" style={{ color: textDim }}>
            <Library size={14} />
            <span className="text-[11px] font-bold uppercase tracking-widest">Library</span>
          </div>
          {collections.length > 0 && (
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full" style={{ background: cardBg, border: `1px solid ${borderColor}`, color: textDim }}>
              {collections.length}
            </span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar px-2 space-y-1 pb-2">
          {collections.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <Globe size={32} className="mx-auto mb-3" style={{ color: textDim }} />
              <p className="text-[12px]" style={{ color: textDim }}>Your video library is empty</p>
            </div>
          ) : (
            collections.map(c => (
              <button
                key={c.name}
                onClick={() => onSelect(c)}
                className="w-full group flex flex-col gap-2 p-3 rounded-2xl transition-all duration-300 border"
                style={{
                  background: selected?.name === c.name ? selectedItemBg : "transparent",
                  borderColor: selected?.name === c.name ? selectedItemBorder : "transparent",
                  boxShadow: selected?.name === c.name ? "0 4px 20px rgba(0,0,0,0.2)" : "none",
                }}
              >
                <div className="flex items-center gap-3 w-full">
                  <div className="w-10 h-10 rounded-xl overflow-hidden shrink-0 relative transition-all"
                       style={{ background: isDark ? "#0f172a" : "#e2e8f0", border: `1px solid ${borderColor}` }}>
                    {c.video_id ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={`https://i.ytimg.com/vi/${c.video_id}/default.jpg`}
                        alt=""
                        className={`w-full h-full object-cover transition-transform duration-500 ${selected?.name === c.name ? "scale-110" : "group-hover:scale-110"}`}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Library size={14} style={{ color: textDim }} />
                      </div>
                    )}
                    {selected?.name === c.name && (
                      <div className="absolute inset-0 bg-indigo-500/20 backdrop-blur-[1px]" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0 text-left">
                    <p className="text-[12.5px] font-semibold leading-snug truncate transition-colors"
                       style={{ color: selected?.name === c.name ? textMain : textMuted }}>
                      {c.title}
                    </p>
                    <p className="text-[10px] mt-0.5 truncate" style={{ color: textDim }}>Source: YouTube</p>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, c.name)}
                    disabled={deletingName === c.name}
                    className="shrink-0 opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-rose-500/10 transition-all"
                    style={{ color: textDim }}
                    title="Delete video"
                  >
                    {deletingName === c.name
                      ? <Loader2 size={13} className="animate-spin" />
                      : <Trash2 size={13} className="hover:text-rose-400" />}
                  </button>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-4 border-t" style={{ borderColor, background: footerBg }}>
        <div className="flex items-center gap-2 mb-1.5">
          <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
          <span className="text-[10px] font-bold uppercase tracking-tighter" style={{ color: textDim }}>System Engine</span>
        </div>
        <p className="text-[9px] font-mono leading-tight" style={{ color: textDim }}>
          BGE-M3 / HYBRID-SEARCH / LLAMA-3.3-70B
        </p>
      </div>
    </aside>
  );
}
