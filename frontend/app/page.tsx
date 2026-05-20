"use client";

import { useCallback, useEffect, useState } from "react";
import { Collection } from "@/lib/types";
import { fetchCollections } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import VideoPanel from "@/components/VideoPanel";
import ChatPanel from "@/components/ChatPanel";
import { Menu, X, LayoutDashboard, Globe, Sun, Moon, Play, MessageSquare } from "lucide-react";

export default function Home() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState<Collection | null>(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [seekTime, setSeekTime] = useState<number | undefined>();
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  // Mobile: which pane is active when a video is selected
  const [mobileTab, setMobileTab] = useState<"video" | "chat">("chat");

  // Apply theme to <html>
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // Open sidebar by default on large screens
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    setSidebarOpen(mq.matches);
    const handler = (e: MediaQueryListEvent) => setSidebarOpen(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

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

  function handleSelectCollection(c: Collection) {
    setSelected(c);
    setMobileTab("chat");
    // Close drawer on mobile after selection
    if (window.innerWidth < 1024) setSidebarOpen(false);
  }

  return (
    <div className="flex h-screen overflow-hidden theme-bg theme-text relative">
      {/* Background Glows */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-500/5 blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-purple-500/5 blur-[120px] pointer-events-none" />

      {/* ── Mobile drawer overlay ── */}
      {sidebarOpen && (
        <div
          className="drawer-overlay lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ── */}
      {/* Desktop: push layout | Mobile: fixed drawer */}
      <div className={`
        shrink-0 transition-all duration-300 ease-in-out
        lg:relative lg:z-auto
        fixed inset-y-0 left-0 z-50
        ${sidebarOpen ? "w-[280px]" : "w-0"}
        overflow-hidden
      `}>
        <Sidebar
          collections={collections}
          selected={selected}
          apiOnline={apiOnline}
          theme={theme}
          onSelect={handleSelectCollection}
          onIngested={loadCollections}
          onDeleted={(name) => {
            setCollections(prev => prev.filter(c => c.name !== name));
            if (selected?.name === name) setSelected(null);
          }}
          onClose={() => setSidebarOpen(false)}
        />
      </div>

      {/* ── Main Content ── */}
      <div className="flex flex-col flex-1 min-w-0 relative">

        {/* Glass Topbar */}
        <header className="flex items-center justify-between px-4 sm:px-6 h-14 border-b glass z-30 shrink-0" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-xl hover:bg-white/5 border border-transparent hover:border-white/10 transition-all text-slate-400 hover:text-white"
              aria-label="Toggle sidebar"
            >
              {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
            </button>

            <div className="flex items-center gap-2 sm:gap-3">
              <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-widest" style={{ background: "var(--bg-hover)", border: "1px solid var(--border)", color: "var(--text-dim)" }}>
                <LayoutDashboard size={12} /> Workspace
              </div>
              <span style={{ color: "var(--border-rich)" }}>/</span>
              <h2 className="text-[13px] font-semibold truncate max-w-[160px] sm:max-w-[300px]" style={{ color: "var(--text-muted)" }}>
                {selected?.title ?? "Awaiting Selection"}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            <div className="hidden md:flex items-center gap-3 pr-4 border-r" style={{ borderColor: "var(--border)" }}>
              {["GraphRAG", "v2.4"].map(tag => (
                <span key={tag} className="text-[10px] font-bold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  {tag}
                </span>
              ))}
            </div>

            {/* Theme toggle */}
            <button
              onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
              className="p-2 rounded-xl border transition-all"
              style={{ background: "var(--bg-hover)", borderColor: "var(--border)", color: "var(--text-muted)" }}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
            </button>

            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${apiOnline ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-rose-500"}`} />
              <span className="hidden sm:inline text-[11px] font-bold uppercase tracking-tighter" style={{ color: "var(--text-dim)" }}>
                {apiOnline ? "Core Active" : "Core Standby"}
              </span>
            </div>
          </div>
        </header>

        {/* ── Main split view ── */}
        <main className="flex flex-1 min-h-0 relative z-10">
          {!selected ? (
            <div className="flex-1 flex flex-col items-center justify-center animate-fade-in px-4">
              <div className="w-20 h-20 rounded-3xl flex items-center justify-center mb-6" style={{ background: "var(--bg-hover)", border: "1px solid var(--border)" }}>
                <Globe size={40} style={{ color: "var(--text-dim)" }} />
              </div>
              <h3 className="text-xl sm:text-2xl font-bold font-display mb-2 text-center" style={{ color: "var(--text)" }}>Initialize Intelligent Analysis</h3>
              <p className="max-w-sm text-center text-[14px] leading-relaxed" style={{ color: "var(--text-dim)" }}>
                Connect a YouTube source from your library to begin deep context extraction and conversational synthesis.
              </p>
              <button
                onClick={() => setSidebarOpen(true)}
                className="mt-6 px-5 py-2.5 rounded-2xl accent-gradient text-white text-[13px] font-bold shadow-lg shadow-indigo-500/20 active:scale-95 transition-all lg:hidden"
              >
                Open Library
              </button>
            </div>
          ) : (
            <>
              {/* ── Desktop: side-by-side ── */}
              <div className="hidden lg:flex flex-col w-[45%] xl:w-[40%] border-r" style={{ borderColor: "var(--border)" }}>
                <VideoPanel collection={selected} seekTime={seekTime} theme={theme} />
              </div>
              <div className="hidden lg:flex flex-1 flex-col min-w-0">
                <ChatPanel collection={selected} onSourceClick={setSeekTime} theme={theme} />
              </div>

              {/* ── Mobile: tabbed view ── */}
              <div className="flex lg:hidden flex-col flex-1 min-w-0">
                {/* Tab bar */}
                <div className="flex border-b shrink-0" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                  {([
                    { key: "chat", icon: <MessageSquare size={14} />, label: "Chat" },
                    { key: "video", icon: <Play size={14} />, label: "Video" },
                  ] as const).map(tab => (
                    <button
                      key={tab.key}
                      onClick={() => setMobileTab(tab.key)}
                      className="flex-1 flex items-center justify-center gap-2 py-3 text-[12px] font-bold transition-all"
                      style={{
                        color: mobileTab === tab.key ? "var(--accent)" : "var(--text-dim)",
                        borderBottom: mobileTab === tab.key ? "2px solid var(--accent)" : "2px solid transparent",
                      }}
                    >
                      {tab.icon} {tab.label}
                    </button>
                  ))}
                </div>

                {/* Active tab content */}
                <div className="flex-1 min-h-0">
                  {mobileTab === "chat"
                    ? <ChatPanel collection={selected} onSourceClick={(t) => { setSeekTime(t); setMobileTab("video"); }} theme={theme} />
                    : <VideoPanel collection={selected} seekTime={seekTime} theme={theme} />
                  }
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
