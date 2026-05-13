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
}

export default function VideoPanel({ collection }: Props) {
  const [summary, setSummary] = useState("");
  const [streaming, setStreaming] = useState(false);

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
    <div className="flex flex-col h-full bg-[#050608] border-r border-white/[0.04] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 h-14 border-b border-white/[0.04] glass-light shrink-0">
        <div className="flex items-center gap-3">
          <YouTubeLogo size={20} />
          <span className="text-[12px] font-bold uppercase tracking-widest text-slate-400 font-display">Source Viewer</span>
        </div>
        {collection && (
          <a 
            href={`https://youtube.com/watch?v=${collection.video_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-500 hover:text-indigo-400 transition-colors"
          >
            <span className="hidden sm:inline">Open in YouTube</span>
            <ExternalLink size={12} />
          </a>
        )}
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {/* Player Container */}
        <div className="p-6">
          {collection?.video_id ? (
            <div className="group relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl blur opacity-10 group-hover:opacity-20 transition duration-1000" />
              <div className="relative w-full rounded-2xl overflow-hidden border border-white/[0.08] bg-black shadow-2xl"
                   style={{ aspectRatio: "16/9" }}>
                <iframe
                  src={`https://www.youtube.com/embed/${collection.video_id}?rel=0&modestbranding=1&autoplay=0`}
                  title={collection.title}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="absolute inset-0 w-full h-full"
                />
              </div>
            </div>
          ) : (
            <div className="w-full rounded-2xl border border-dashed border-white/[0.1] bg-white/[0.01] flex flex-col items-center justify-center gap-4 py-24">
              <div className="w-16 h-16 rounded-3xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center text-slate-700">
                <Play size={24} className="ml-1" />
              </div>
              <p className="text-[14px] font-medium text-slate-600">Select content to begin analysis</p>
            </div>
          )}
        </div>

        {/* Info & Intelligence Section */}
        {collection && (
          <div className="px-6 pb-10 space-y-8 animate-fade-in">
            {/* Metadata Card */}
            <div className="space-y-4">
              <h2 className="text-xl font-bold text-white font-display leading-tight tracking-tight">
                {collection.title}
              </h2>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2 text-[12px] text-slate-500">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  Successfully Indexed
                </div>
                <div className="flex items-center gap-2 text-[12px] text-slate-500">
                  <Clock size={14} className="text-slate-700" />
                  Synced
                </div>
              </div>
            </div>

            {/* Summarization Interface */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-slate-400">
                  <Sparkles size={14} className="text-indigo-400" />
                  <span className="text-[11px] font-bold uppercase tracking-widest">AI Synthesis</span>
                </div>
                {summary && !streaming && (
                  <button 
                    onClick={handleSummarize}
                    className="text-[11px] font-bold text-indigo-400 hover:text-indigo-300 transition-colors"
                  >
                    Regenerate
                  </button>
                )}
              </div>

              {!summary && !streaming ? (
                <button
                  onClick={handleSummarize}
                  className="w-full group relative overflow-hidden rounded-2xl p-px"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/50 to-purple-500/50 opacity-20 group-hover:opacity-40 transition-opacity" />
                  <div className="relative bg-[#0d1117] hover:bg-[#12161f] rounded-2xl px-6 py-10 flex flex-col items-center gap-3 transition-colors border border-white/[0.06]">
                    <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                      <BookOpen size={20} className="text-indigo-400" />
                    </div>
                    <div className="text-center">
                      <p className="text-[15px] font-bold text-slate-200">Generate Intelligence Summary</p>
                      <p className="text-[12px] text-slate-500 mt-1">Get an instant overview of the key concepts and timeline.</p>
                    </div>
                  </div>
                </button>
              ) : (
                <div className="rounded-2xl border border-white/[0.06] bg-white/[0.01] overflow-hidden">
                  <div className="px-5 py-4 border-b border-white/[0.04] flex items-center justify-between bg-white/[0.01]">
                    <div className="flex items-center gap-2">
                      <Info size={14} className="text-indigo-400" />
                      <span className="text-[12px] font-bold text-slate-300">Detailed Abstract</span>
                    </div>
                    {streaming && <Loader2 size={14} className="animate-spin text-indigo-500" />}
                  </div>
                  <div className="p-5 text-[14.5px] text-slate-400 leading-relaxed whitespace-pre-wrap font-light">
                    {summary || "Initializing synthesis engine..."}
                    {streaming && (
                      <span className="inline-block w-1.5 h-4 bg-indigo-500 ml-1 rounded-sm blink align-middle" />
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Quick Stats/Metadata (Optional/Mock for now) */}
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Processing Mode", value: "Graph-Augmented" },
                { label: "Vector Space", value: "Hybrid Dense+Sparse" }
              ].map(stat => (
                <div key={stat.label} className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.04]">
                  <p className="text-[10px] font-bold text-slate-600 uppercase tracking-tighter mb-1">{stat.label}</p>
                  <p className="text-[13px] font-semibold text-slate-300">{stat.value}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

