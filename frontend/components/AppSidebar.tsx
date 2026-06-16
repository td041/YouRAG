"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare, Library, BarChart2, Settings, Plus,
  Loader2, Trash2, Check, Globe, Sun, Moon, X, GraduationCap,
} from "lucide-react";
import type { Collection } from "@/lib/types";
import { deleteCollection } from "@/lib/api";
import { useCollections } from "@/lib/collections-context";

function YouTubeLogo({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size * 0.7} viewBox="0 0 90 63" fill="none">
      <rect width="90" height="63" rx="13" fill="#FF0000" />
      <path d="M36 44V19L63 31.5L36 44Z" fill="white" />
    </svg>
  );
}

const NAV_ITEMS = [
  { href: "/chat",      icon: MessageSquare,  label: "Chat" },
  { href: "/library",   icon: Library,        label: "Library" },
  { href: "/learn",     icon: GraduationCap,  label: "Learn" },
  { href: "/analytics", icon: BarChart2,      label: "Analytics" },
  { href: "/settings",  icon: Settings,       label: "Settings" },
] as const;

interface Props {
  onClose?: () => void;
}

export default function AppSidebar({ onClose }: Props) {
  const pathname = usePathname();
  const {
    collections, selectedCollections, loadingCollections, apiOnline, theme,
    setTheme, setSelectedCollections, setActiveVideo, onDeleted,
  } = useCollections();

  const isDark = theme === "dark";
  const border    = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.07)";
  const cardBg    = isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)";
  const textMain  = isDark ? "#f0f2f5" : "#0f172a";
  const textMuted = isDark ? "#94a3b8" : "#475569";
  const textDim   = isDark ? "#475569" : "#94a3b8";
  const bg        = isDark ? "#050608" : "#f1f3f8";
  const footerBg  = isDark ? "#030406" : "#e8eaf0";

  async function handleDelete(e: React.MouseEvent, name: string) {
    e.stopPropagation();
    if (!confirm(`Delete "${name}" from library?`)) return;
    try { await deleteCollection(name); onDeleted(name); }
    catch (err) { alert(err instanceof Error ? err.message : String(err)); }
  }

  function handleToggleSelect(c: Collection) {
    setActiveVideo(c);
    setSelectedCollections(prev =>
      prev.some(x => x.name === c.name)
        ? prev.filter(x => x.name !== c.name)
        : [...prev, c]
    );
    onClose?.();
  }

  return (
    <aside className="flex flex-col h-full w-[260px] shrink-0 overflow-hidden"
           style={{ background: bg, borderRight: `1px solid ${border}` }}>

      {/* ── Brand ── */}
      <div className="flex items-center justify-between px-5 pt-5 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
               style={{ background: "rgba(255,255,255,0.05)", border: `1px solid ${border}` }}>
            <YouTubeLogo size={20} />
          </div>
          <h1 className="text-[17px] font-bold font-display tracking-tight" style={{ color: textMain }}>
            YouRAG
          </h1>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={`w-4 h-4 rounded-full border flex items-center justify-center transition-colors ${
            apiOnline ? "border-emerald-500/20 bg-emerald-500/10" : "border-rose-500/20 bg-rose-500/10"
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${apiOnline ? "bg-emerald-500" : "bg-rose-500"}`} />
          </div>
          <button onClick={() => setTheme(isDark ? "light" : "dark")}
                  className="p-1.5 rounded-lg border transition-all"
                  style={{ borderColor: border, color: textDim }}>
            {isDark ? <Sun size={12} /> : <Moon size={12} />}
          </button>
          {onClose && (
            <button onClick={onClose} className="lg:hidden p-1.5 rounded-lg" style={{ color: textDim }}>
              <X size={15} />
            </button>
          )}
        </div>
      </div>

      {/* ── Navigation ── */}
      <nav className="px-3 mb-3 space-y-0.5">
        {NAV_ITEMS.map(({ href, icon: Icon, label }) => {
          const active = pathname === href || (href === "/chat" && pathname === "/");
          return (
            <Link key={href} href={href}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-all"
                  style={{
                    background: active ? (isDark ? "rgba(99,102,241,0.15)" : "rgba(99,102,241,0.08)") : "transparent",
                    color: active ? (isDark ? "#818cf8" : "#4f46e5") : textDim,
                    borderLeft: active ? "2px solid rgba(99,102,241,0.5)" : "2px solid transparent",
                  }}>
              <Icon size={15} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* ── Divider ── */}
      <div className="mx-4 mb-3" style={{ height: 1, background: border }} />

      {/* ── Library header ── */}
      <div className="flex items-center justify-between px-5 mb-2">
        <div className="flex items-center gap-2" style={{ color: textDim }}>
          <Library size={13} />
          <span className="text-[11px] font-bold uppercase tracking-widest">Library</span>
          {collections.length > 0 && (
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full"
                  style={{ background: cardBg, border: `1px solid ${border}`, color: textDim }}>
              {collections.length}
            </span>
          )}
        </div>
        <Link href="/library"
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-bold transition-all hover:bg-indigo-500/10"
              style={{ color: "var(--accent, #818cf8)", border: `1px solid rgba(99,102,241,0.2)` }}
              title="Add video">
          <Plus size={11} /> Add
        </Link>
      </div>

      {/* ── Video list (takes all remaining space) ── */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-3 space-y-1 pb-3">
        {loadingCollections && collections.length === 0 ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 size={20} className="animate-spin" style={{ color: textDim }} />
          </div>
        ) : collections.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <Globe size={28} className="mx-auto mb-2" style={{ color: textDim, opacity: 0.5 }} />
            <p className="text-[12px]" style={{ color: textDim }}>No videos yet</p>
            <Link href="/library"
                  className="inline-block mt-2 text-[11px] font-bold text-indigo-400 hover:text-indigo-300 transition-colors">
              + Add your first video
            </Link>
          </div>
        ) : (
          collections.map(c => {
            const isSel = selectedCollections.some(s => s.name === c.name);
            return (
              <button key={c.name} onClick={() => handleToggleSelect(c)}
                      className="w-full group flex items-center gap-3 p-2.5 rounded-xl transition-all duration-200 border"
                      style={{
                        background: isSel ? (isDark ? "rgba(99,102,241,0.1)" : "rgba(99,102,241,0.06)") : "transparent",
                        borderColor: isSel ? (isDark ? "rgba(99,102,241,0.25)" : "rgba(99,102,241,0.2)") : "transparent",
                      }}>
                {/* Thumbnail */}
                <div className="w-9 h-9 rounded-lg overflow-hidden shrink-0 relative"
                     style={{ background: isDark ? "#0f172a" : "#e2e8f0", border: `1px solid ${border}` }}>
                  {c.video_id
                    // eslint-disable-next-line @next/next/no-img-element
                    ? <img src={`https://i.ytimg.com/vi/${c.video_id}/default.jpg`} alt=""
                           className="w-full h-full object-cover" />
                    : <div className="w-full h-full flex items-center justify-center">
                        <Library size={12} style={{ color: textDim }} />
                      </div>}
                  {isSel && (
                    <div className="absolute inset-0 bg-indigo-500/30 flex items-center justify-center">
                      <Check size={12} className="text-white" />
                    </div>
                  )}
                </div>

                {/* Title */}
                <div className="flex-1 min-w-0 text-left">
                  <p className="text-[12px] font-semibold leading-snug truncate"
                     style={{ color: isSel ? (isDark ? "#c7d2fe" : "#4f46e5") : textMuted }}>
                    {c.title}
                  </p>
                  <p className="text-[10px] mt-0.5" style={{ color: textDim }}>
                    {isSel ? "In context" : "YouTube"}
                  </p>
                </div>

                {/* Delete */}
                <button onClick={(e) => handleDelete(e, c.name)}
                        className="shrink-0 opacity-0 group-hover:opacity-100 p-1 rounded-lg hover:bg-rose-500/10 transition-all"
                        style={{ color: textDim }}>
                  <Trash2 size={12} className="hover:text-rose-400" />
                </button>
              </button>
            );
          })
        )}
      </div>

      {/* ── Footer ── */}
      <div className="px-5 py-3 border-t" style={{ borderColor: border, background: footerBg }}>
        <p className="text-[9px] font-mono" style={{ color: textDim }}>
          BGE-M3 · HYBRID · LLAMA-3.3-70B
        </p>
      </div>
    </aside>
  );
}
