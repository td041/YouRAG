"use client";

import { useCallback, useEffect, useState } from "react";
import { Collection } from "@/lib/types";
import { fetchCollections } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import VideoPanel from "@/components/VideoPanel";
import ChatPanel from "@/components/ChatPanel";
import { Menu, X, LayoutDashboard, Globe } from "lucide-react";

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

  useEffect(() => { loadCollections(); }, [loadCollections]);

  return (
    <div className="flex h-screen overflow-hidden bg-[#050608] text-slate-200">
      {/* Background Glows */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-500/5 blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-purple-500/5 blur-[120px] pointer-events-none" />

      {/* Sidebar */}
      <div className={`transition-all duration-500 ease-in-out ${sidebarOpen ? "w-[280px]" : "w-0"} overflow-hidden shrink-0`}>
        <Sidebar
          collections={collections}
          selected={selected}
          apiOnline={apiOnline}
          onSelect={setSelected}
          onIngested={loadCollections}
        />
      </div>

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 min-w-0 relative">
        
        {/* Glass Topbar */}
        <header className="flex items-center justify-between px-6 h-14 border-b border-white/[0.04] glass z-30">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-xl hover:bg-white/5 border border-transparent hover:border-white/10 transition-all text-slate-400 hover:text-white"
            >
              {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
            
            <div className="flex items-center gap-3">
              <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.03] border border-white/[0.06] text-[11px] font-bold text-slate-500 uppercase tracking-widest">
                <LayoutDashboard size={12} /> Workspace
              </div>
              <span className="text-slate-700">/</span>
              <h2 className="text-[13px] font-semibold text-slate-300 truncate max-w-[300px]">
                {selected?.title ?? "Awaiting Selection"}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-3 pr-4 border-r border-white/5">
              {["GraphRAG", "v2.4"].map(tag => (
                <span key={tag} className="text-[10px] font-bold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  {tag}
                </span>
              ))}
            </div>
            
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${apiOnline ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-rose-500"}`} />
              <span className="text-[11px] font-bold text-slate-500 uppercase tracking-tighter">
                {apiOnline ? "Core Active" : "Core Standby"}
              </span>
            </div>
          </div>
        </header>

        {/* Dynamic Split View */}
        <main className="flex flex-1 min-h-0 relative z-10">
          {!selected ? (
            <div className="flex-1 flex flex-col items-center justify-center animate-fade-in">
              <div className="w-20 h-20 rounded-3xl bg-white/[0.02] border border-white/[0.06] flex items-center justify-center mb-6">
                <Globe size={40} className="text-slate-800" />
              </div>
              <h3 className="text-2xl font-bold font-display text-white mb-2">Initialize Intelligent Analysis</h3>
              <p className="text-slate-500 max-w-sm text-center text-[14px] leading-relaxed">
                Connect a YouTube source from your library to begin deep context extraction and conversational synthesis.
              </p>
            </div>
          ) : (
            <>
              {/* Left Pane: Video Insight */}
              <div className="hidden lg:flex flex-col w-[45%] xl:w-[40%] border-r border-white/[0.04]">
                <VideoPanel collection={selected} />
              </div>

              {/* Right Pane: AI Chat Interface */}
              <div className="flex-1 flex flex-col min-w-0">
                <ChatPanel collection={selected} />
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

