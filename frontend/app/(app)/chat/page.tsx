"use client";
import { useState } from "react";
import { useCollections } from "@/lib/collections-context";
import VideoPanel from "@/components/VideoPanel";
import ChatPanel from "@/components/ChatPanel";
import { Globe, Play, MessageSquare, LayoutDashboard, Menu } from "lucide-react";

export default function ChatPage() {
  const { selectedCollections, activeVideo, setActiveVideo, collections, theme, openSidebar } = useCollections();
  // Use { time, seq } so clicking same timestamp twice still triggers iframe re-render
  const [seek, setSeek] = useState<{ time: number; seq: number } | undefined>();
  const [mobileTab, setMobileTab] = useState<"video" | "chat">("chat");

  function handleSourceClick(time: number, videoId?: string) {
    if (videoId) {
      const match = collections.find(c => c.video_id === videoId);
      if (match) setActiveVideo(match);
    }
    setSeek(prev => ({ time, seq: (prev?.seq ?? 0) + 1 }));
  }

  return (
    <>
      {/* Topbar */}
      <header className="flex items-center justify-between px-4 sm:px-6 h-14 border-b glass z-30 shrink-0"
              style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-3">
          <button onClick={openSidebar}
                  className="lg:hidden p-1.5 rounded-xl hover:bg-white/5 transition-all shrink-0 -ml-1"
                  style={{ color: "var(--text-dim)" }}
                  aria-label="Open menu">
            <Menu size={18} />
          </button>
          <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-widest"
               style={{ background: "var(--bg-hover)", border: "1px solid var(--border)", color: "var(--text-dim)" }}>
            <LayoutDashboard size={12} /> Workspace
          </div>
          <span className="hidden sm:block" style={{ color: "var(--border-rich)" }}>/</span>
          <h2 className="text-[13px] font-semibold truncate max-w-[160px] sm:max-w-[300px]"
              style={{ color: "var(--text-muted)" }}>
            {selectedCollections.length > 1
              ? `${selectedCollections.length} videos selected`
              : selectedCollections[0]?.title ?? "Awaiting Selection"}
          </h2>
        </div>
        <div className="hidden md:flex items-center gap-3">
          {["GraphRAG", "v2.4"].map(tag => (
            <span key={tag} className="text-[10px] font-bold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              {tag}
            </span>
          ))}
        </div>
      </header>

      {/* Content */}
      <main className="flex flex-1 min-h-0 relative z-10">
        {selectedCollections.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center animate-fade-in px-4">
            <div className="w-20 h-20 rounded-3xl flex items-center justify-center mb-6"
                 style={{ background: "var(--bg-hover)", border: "1px solid var(--border)" }}>
              <Globe size={40} style={{ color: "var(--text-dim)" }} />
            </div>
            <h3 className="text-xl sm:text-2xl font-bold font-display mb-2 text-center" style={{ color: "var(--text)" }}>
              Initialize Intelligent Analysis
            </h3>
            <p className="max-w-sm text-center text-[14px] leading-relaxed" style={{ color: "var(--text-dim)" }}>
              Select a video from the library on the left to begin deep context extraction and conversational synthesis.
            </p>
          </div>
        ) : (
          <>
            {/* Desktop: side-by-side */}
            <div className="hidden lg:flex flex-col w-[45%] xl:w-[40%] border-r"
                 style={{ borderColor: "var(--border)" }}>
              <VideoPanel collection={activeVideo} seek={seek} onSeek={t => handleSourceClick(t)} theme={theme} />
            </div>
            <div className="hidden lg:flex flex-1 flex-col min-w-0">
              <ChatPanel collections={selectedCollections} activeVideo={activeVideo}
                         onSourceClick={handleSourceClick} theme={theme} />
            </div>

            {/* Mobile: tabbed */}
            <div className="flex lg:hidden flex-col flex-1 min-w-0">
              <div className="flex border-b shrink-0"
                   style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                {([
                  { key: "chat",  icon: <MessageSquare size={14} />, label: "Chat" },
                  { key: "video", icon: <Play size={14} />,          label: "Video" },
                ] as const).map(tab => (
                  <button key={tab.key} onClick={() => setMobileTab(tab.key)}
                          className="flex-1 flex items-center justify-center gap-2 py-3 text-[12px] font-bold transition-all"
                          style={{
                            color: mobileTab === tab.key ? "var(--accent)" : "var(--text-dim)",
                            borderBottom: mobileTab === tab.key ? "2px solid var(--accent)" : "2px solid transparent",
                          }}>
                    {tab.icon} {tab.label}
                  </button>
                ))}
              </div>
              <div className="flex-1 min-h-0">
                {mobileTab === "chat"
                  ? <ChatPanel collections={selectedCollections} activeVideo={activeVideo}
                               onSourceClick={(t, vid) => { handleSourceClick(t, vid); setMobileTab("video"); }}
                               theme={theme} />
                  : <VideoPanel collection={activeVideo} seek={seek} onSeek={t => handleSourceClick(t)} theme={theme} />}
              </div>
            </div>
          </>
        )}
      </main>
    </>
  );
}
