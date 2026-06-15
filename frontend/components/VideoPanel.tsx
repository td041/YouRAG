"use client";

import { useState } from "react";
import { Collection } from "@/lib/types";
import { streamSummary } from "@/lib/api";
import {
  Play, Sparkles, Loader2,
  ExternalLink, Clock, BookOpen, Info
} from "lucide-react";

function YouTubeLogo({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size * 0.7} viewBox="0 0 90 63" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="90" height="63" rx="13" fill="#FF0000"/>
      <path d="M36 44V19L63 31.5L36 44Z" fill="white"/>
    </svg>
  );
}

interface Props {
  collection: Collection | null;
  seek?: { time: number; seq: number };
  theme: "dark" | "light";
}

export default function VideoPanel({ collection, seek, theme }: Props) {
  const [summary, setSummary] = useState("");
  const [streaming, setStreaming] = useState(false);

  const isDark = theme === "dark";
  const borderColor = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.06)";
  const textMain = isDark ? "#f0f2f5" : "#0f172a";
  const textMuted = isDark ? "#94a3b8" : "#475569";
  const textDim = isDark ? "#475569" : "#94a3b8";
  const cardBg = isDark ? "#0d1117" : "#ffffff";
  const cardBorder = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.07)";
  const headerBg = isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)";
  const statBg = isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)";

  function handleSummarize() {
    if (!collection || streaming) return;
    setSummary("");
    setStreaming(true);

    streamSummary(
      collection.name,
      (chunk) => setSummary(prev => prev + chunk),
      () => setStreaming(false),
      (err) => { setSummary(`Error: ${err}`); setStreaming(false); }
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: isDark ? "#050608" : "#f8f9fc", borderRight: `1px solid ${borderColor}` }}>

      {/* Header */}
      <div className="flex items-center justify-between px-6 h-14 border-b glass-light shrink-0" style={{ borderColor }}>
        <div className="flex items-center gap-3">
          <YouTubeLogo size={20} />
          <span className="text-[12px] font-bold uppercase tracking-widest font-display" style={{ color: textMuted }}>Source Viewer</span>
        </div>
        {collection && (
          <a
            href={`https://youtube.com/watch?v=${collection.video_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-[11px] font-semibold transition-colors hover:text-indigo-400"
            style={{ color: textDim }}
          >
            <span className="hidden sm:inline">Open in YouTube</span>
            <ExternalLink size={12} />
          </a>
        )}
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar">

        {/* Player */}
        <div className="p-4 sm:p-6">
          {collection?.video_id ? (
            <div className="group relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl blur opacity-10 group-hover:opacity-20 transition duration-1000" />
              <div className="relative w-full rounded-2xl overflow-hidden border shadow-2xl"
                   style={{ aspectRatio: "16/9", borderColor: cardBorder, background: "#000" }}>
                <iframe
                  key={seek ? `seek-${seek.seq}-${seek.time}` : `vid-${collection.video_id}`}
                  src={`https://www.youtube.com/embed/${collection.video_id}?rel=0&modestbranding=1&autoplay=${seek ? 1 : 0}${seek ? `&start=${Math.floor(seek.time)}` : ""}`}
                  title={collection.title}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="absolute inset-0 w-full h-full"
                />
              </div>
            </div>
          ) : (
            <div className="w-full rounded-2xl border border-dashed flex flex-col items-center justify-center gap-4 py-20"
                 style={{ borderColor: cardBorder, background: statBg }}>
              <div className="w-16 h-16 rounded-3xl flex items-center justify-center" style={{ background: statBg, border: `1px solid ${cardBorder}` }}>
                <Play size={24} className="ml-1" style={{ color: textDim }} />
              </div>
              <p className="text-[14px] font-medium" style={{ color: textDim }}>Select content to begin</p>
            </div>
          )}
        </div>

        {/* Info section */}
        {collection && (
          <div className="px-4 sm:px-6 pb-8 space-y-6 animate-fade-in">

            {/* Title + meta */}
            <div className="space-y-3">
              <h2 className="text-lg sm:text-xl font-bold font-display leading-tight tracking-tight" style={{ color: textMain }}>
                {collection.title}
              </h2>
              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-2 text-[12px]" style={{ color: textDim }}>
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  Indexed
                </div>
                <div className="flex items-center gap-2 text-[12px]" style={{ color: textDim }}>
                  <Clock size={14} style={{ color: textDim }} />
                  Synced
                </div>
              </div>
            </div>

            {/* Summarization */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2" style={{ color: textMuted }}>
                  <Sparkles size={14} className="text-indigo-400" />
                  <span className="text-[11px] font-bold uppercase tracking-widest">AI Synthesis</span>
                </div>
                {summary && !streaming && (
                  <button onClick={handleSummarize} className="text-[11px] font-bold text-indigo-400 hover:text-indigo-300 transition-colors">
                    Regenerate
                  </button>
                )}
              </div>

              {!summary && !streaming ? (
                <button onClick={handleSummarize} className="w-full group relative overflow-hidden rounded-2xl p-px">
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/50 to-purple-500/50 opacity-20 group-hover:opacity-40 transition-opacity" />
                  <div className="relative rounded-2xl px-6 py-8 flex flex-col items-center gap-3 transition-colors border"
                       style={{ background: cardBg, borderColor: cardBorder }}>
                    <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                      <BookOpen size={20} className="text-indigo-400" />
                    </div>
                    <div className="text-center">
                      <p className="text-[14px] font-bold" style={{ color: textMain }}>Generate AI Summary</p>
                      <p className="text-[12px] mt-1" style={{ color: textDim }}>Overview of key concepts and timeline.</p>
                    </div>
                  </div>
                </button>
              ) : (
                <div className="rounded-2xl border overflow-hidden" style={{ borderColor: cardBorder }}>
                  <div className="px-5 py-3 border-b flex items-center justify-between" style={{ borderColor, background: headerBg }}>
                    <div className="flex items-center gap-2">
                      <Info size={14} className="text-indigo-400" />
                      <span className="text-[12px] font-bold" style={{ color: textMain }}>Detailed Abstract</span>
                    </div>
                    {streaming && <Loader2 size={14} className="animate-spin text-indigo-500" />}
                  </div>
                  <div className="p-5 text-[14px] leading-relaxed whitespace-pre-wrap font-light" style={{ color: textMuted, background: cardBg }}>
                    {summary || "Initializing synthesis engine..."}
                    {streaming && (
                      <span className="inline-block w-1.5 h-4 bg-indigo-500 ml-1 rounded-sm blink align-middle" />
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Processing Mode", value: "Graph-Augmented" },
                { label: "Vector Space",    value: "Hybrid Dense+Sparse" },
              ].map(stat => (
                <div key={stat.label} className="p-4 rounded-2xl border" style={{ background: statBg, borderColor: cardBorder }}>
                  <p className="text-[10px] font-bold uppercase tracking-tighter mb-1" style={{ color: textDim }}>{stat.label}</p>
                  <p className="text-[13px] font-semibold" style={{ color: textMain }}>{stat.value}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
