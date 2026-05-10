"use client";

import { useState } from "react";
import { Collection } from "@/lib/types";
import { streamSummary } from "@/lib/api";
import { Play, Sparkles, Loader2, ChevronDown, ChevronUp } from "lucide-react";

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
  const [showSummary, setShowSummary] = useState(false);

  function handleSummarize() {
    if (!collection || streaming) return;
    setSummary("");
    setStreaming(true);
    setShowSummary(true);

    streamSummary(
      collection.name,
      (chunk) => setSummary(prev => prev + chunk),
      () => setStreaming(false),
      (err) => { setSummary(`Error: ${err}`); setStreaming(false); }
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-5 h-12 border-b border-white/[.07] shrink-0">
        <YouTubeLogo size={18} />
        <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">Video Player</span>
        {collection && (
          <span className="ml-auto text-[11px] text-slate-600 truncate max-w-[200px]">{collection.title}</span>
        )}
      </div>

      {/* Player */}
      <div className="p-4">
        {collection?.video_id ? (
          <div className="relative w-full rounded-xl overflow-hidden border border-white/[.07] bg-black shadow-2xl shadow-black/50"
               style={{ aspectRatio: "16/9" }}>
            <iframe
              src={`https://www.youtube.com/embed/${collection.video_id}?rel=0&modestbranding=1`}
              title={collection.title}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="absolute inset-0 w-full h-full"
            />
          </div>
        ) : (
          <div className="w-full rounded-xl border border-dashed border-white/[.07] bg-[#12151b] flex flex-col items-center justify-center gap-3 py-16">
            <div className="w-12 h-12 rounded-full bg-white/[.04] border border-white/[.07] flex items-center justify-center">
              <Play size={20} className="text-slate-600 ml-0.5" />
            </div>
            <p className="text-[13px] text-slate-600">Select a video from the library</p>
          </div>
        )}
      </div>

      {/* Summarize */}
      {collection && (
        <div className="px-4 pb-4 space-y-3">
          <button
            onClick={handleSummarize}
            disabled={streaming}
            className="w-full flex items-center justify-center gap-2 bg-white/[.04] hover:bg-white/[.07] disabled:opacity-50 border border-white/[.07] hover:border-white/[.12] text-slate-300 text-[13px] font-medium rounded-xl py-2.5 transition-all"
          >
            {streaming
              ? <><Loader2 size={13} className="spin" /><span>Summarizing…</span></>
              : <><Sparkles size={13} className="text-indigo-400" /><span>Summarize Video</span></>}
          </button>

          {(summary || streaming) && (
            <div className="rounded-xl border border-white/[.07] bg-[#12151b] overflow-hidden">
              <button
                onClick={() => setShowSummary(v => !v)}
                className="w-full flex items-center justify-between px-4 py-2.5 text-[11px] font-semibold uppercase tracking-widest text-slate-500 hover:text-slate-400 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                  <span>Summary</span>
                </div>
                {showSummary ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
              {showSummary && (
                <div className="px-4 pb-4 text-[13px] text-slate-400 leading-relaxed border-t border-white/[.05] pt-3 whitespace-pre-wrap">
                  {summary}
                  {streaming && (
                    <span className="inline-block w-0.5 h-3.5 bg-indigo-400 ml-0.5 -mb-0.5 blink" />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
