"use client";

import { useCallback, useEffect, useState } from "react";
import { Collection } from "@/lib/types";
import { fetchCollections } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import VideoPanel from "@/components/VideoPanel";
import ChatPanel from "@/components/ChatPanel";
import { Layers } from "lucide-react";

export default function Home() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState<Collection | null>(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const loadCollections = useCallback(async () => {
    try {
      const data = await fetchCollections();
      setCollections(data);
      setApiOnline(true);
      if (data.length > 0 && !selected) setSelected(data[0]);
    } catch {
      setApiOnline(false);
    }
  }, [selected]);

  useEffect(() => { loadCollections(); }, []);

  function handleSelect(c: Collection) {
    setSelected(c);
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#0d0f12]">
      {/* Sidebar */}
      {sidebarOpen && (
        <Sidebar
          collections={collections}
          selected={selected}
          apiOnline={apiOnline}
          onSelect={handleSelect}
          onIngested={loadCollections}
        />
      )}

      {/* Main */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">

        {/* Topbar */}
        <header className="flex items-center gap-3 px-5 h-12 border-b border-white/[.07] bg-[#0d0f12]/90 backdrop-blur-xl sticky top-0 z-10 shrink-0">
          <button
            onClick={() => setSidebarOpen(v => !v)}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-slate-500 hover:text-slate-300 hover:bg-white/[.05] transition-all"
            title="Toggle sidebar"
          >
            <Layers size={14} />
          </button>

          <div className="flex items-center gap-2 text-[13px] min-w-0">
            <span className="text-slate-600 font-medium shrink-0">YouRAG</span>
            <span className="text-slate-700 shrink-0">/</span>
            <span className="text-slate-400 truncate">
              {selected?.title ?? "No video selected"}
            </span>
          </div>

          <div className="ml-auto flex items-center gap-1.5">
            {["bge-m3", "HybridRRF", "CrossEncoder", "llama-3.3-70b"].map(t => (
              <span key={t} className="hidden sm:inline text-[10px] font-mono text-slate-700 bg-white/[.03] border border-white/[.05] px-2 py-0.5 rounded">
                {t}
              </span>
            ))}
          </div>
        </header>

        {/* Two-pane content */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Video pane */}
          <div className="flex flex-col w-[48%] min-w-0 border-r border-white/[.07] overflow-hidden">
            <VideoPanel collection={selected} />
          </div>

          {/* Chat pane */}
          <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
            <ChatPanel collection={selected} />
          </div>
        </div>
      </div>
    </div>
  );
}
