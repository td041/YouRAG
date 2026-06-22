"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useCollections } from "@/lib/collections-context";
import { deleteCollection, ingestVideo } from "@/lib/api";
import {
  Trash2, MessageSquare, Library, Loader2, Search,
  Plus, Sparkles, Eye, CheckCircle2, XCircle, X, Menu,
} from "lucide-react";

type IngestStatus = "idle" | "queued" | "running" | "ok" | "error";

function ConfirmDeleteModal({ name, onConfirm, onCancel }: { name: string; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "rgba(0,0,0,0.7)" }}>
      <div className="w-full max-w-sm rounded-2xl shadow-2xl overflow-hidden animate-slide-up"
           style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
        <div className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center shrink-0">
              <Trash2 size={18} className="text-rose-400" />
            </div>
            <div>
              <p className="text-[15px] font-bold" style={{ color: "var(--text)" }}>Delete video?</p>
              <p className="text-[12px] mt-0.5 truncate max-w-[220px]" style={{ color: "var(--text-dim)" }}>{name}</p>
            </div>
          </div>
          <p className="text-[13px] mb-5" style={{ color: "var(--text-dim)" }}>
            This will permanently remove the video and all its indexed data. This action cannot be undone.
          </p>
          <div className="flex gap-3">
            <button onClick={onCancel}
                    className="flex-1 h-10 rounded-xl text-[13px] font-semibold transition-all hover:bg-white/5"
                    style={{ border: "1px solid var(--border)", color: "var(--text-muted)" }}>
              Cancel
            </button>
            <button onClick={onConfirm}
                    className="flex-1 h-10 rounded-xl text-[13px] font-bold text-white transition-all active:scale-95 bg-rose-500 hover:bg-rose-600">
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function IngestModal({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [url, setUrl] = useState("");
  const [useCtx, setUseCtx] = useState(false);
  const [useVisual, setUseVisual] = useState(false);
  const [status, setStatus] = useState<IngestStatus>("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [chunks, setChunks] = useState<number | null>(null);
  const isIngesting = status === "queued" || status === "running";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setStatus("queued"); setStatusMsg(""); setChunks(null);
    try {
      const d = await ingestVideo(url.trim(), useCtx, false, useVisual, s => {
        if (s === "running") setStatus("running");
      });
      setStatus("ok"); setChunks(d?.chunks_added ?? null); setUrl("");
      onDone();
      setTimeout(() => { setStatus("idle"); onClose(); }, 2500);
    } catch (err: unknown) {
      setStatus("error");
      setStatusMsg(err instanceof Error ? err.message.slice(0, 100) : String(err).slice(0, 100));
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "rgba(0,0,0,0.6)" }}>
      <div className="w-full max-w-md rounded-2xl shadow-2xl overflow-hidden"
           style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: "var(--border)" }}>
          <h2 className="text-[15px] font-bold" style={{ color: "var(--text)" }}>Add Video</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 transition-colors"
                  style={{ color: "var(--text-dim)" }}>
            <X size={16} />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="text-[12px] font-medium mb-1.5 block" style={{ color: "var(--text-dim)" }}>
              YouTube URL
            </label>
            <input type="text" value={url} onChange={e => setUrl(e.target.value)}
                   placeholder="https://youtube.com/watch?v=..."
                   autoFocus
                   className="w-full rounded-xl px-4 py-3 text-[14px] outline-none focus:ring-2 focus:ring-indigo-500/30"
                   style={{ background: "var(--bg-hover)", border: "1px solid var(--border)", color: "var(--text)" }} />
          </div>

          <div className="space-y-2">
            {[
              { state: useCtx,    set: setUseCtx,    icon: Sparkles, label: "AI Contextualizer",  desc: "Enriches each chunk with document context" },
              { state: useVisual, set: setUseVisual, icon: Eye,       label: "Visual Frame RAG",   desc: "Indexes video frames with AI vision" },
            ].map(({ state, set, icon: Icon, label, desc }) => (
              <div key={label} onClick={() => set(v => !v)}
                   className="flex items-center justify-between p-3 rounded-xl cursor-pointer select-none transition-colors hover:bg-white/3"
                   style={{ border: "1px solid var(--border)" }}>
                <div className="flex items-center gap-3">
                  <Icon size={14} style={{ color: state ? "#818cf8" : "var(--text-dim)" }} />
                  <div>
                    <p className="text-[13px] font-medium" style={{ color: state ? "var(--text)" : "var(--text-dim)" }}>{label}</p>
                    <p className="text-[11px]" style={{ color: "var(--text-dim)" }}>{desc}</p>
                  </div>
                </div>
                <div className={`w-8 h-4.5 rounded-full relative transition-all ${state ? "bg-indigo-500" : ""}`}
                     style={{ width: 32, height: 18, ...(!state ? { background: "var(--bg-hover)" } : {}) }}>
                  <div className={`absolute top-0.5 w-3.5 h-3.5 rounded-full bg-white transition-all`}
                       style={{ width: 14, height: 14, left: state ? 15 : 2 }} />
                </div>
              </div>
            ))}
          </div>

          {isIngesting && (
            <div className="space-y-2">
              <div className="flex justify-between text-[12px]">
                <span style={{ color: "var(--text-dim)" }}>{status === "queued" ? "Queued…" : "Processing…"}</span>
                <span style={{ color: "var(--text-dim)" }}>{status === "queued" ? "25%" : "70%"}</span>
              </div>
              <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-hover)" }}>
                <div className="h-full accent-gradient rounded-full transition-all duration-700"
                     style={{ width: status === "queued" ? "25%" : "70%" }} />
              </div>
            </div>
          )}
          {status === "ok" && (
            <div className="flex items-center gap-2 text-emerald-400 text-[13px]">
              <CheckCircle2 size={15} />
              <span>{chunks != null ? `${chunks} chunks indexed successfully!` : "Done!"}</span>
            </div>
          )}
          {status === "error" && (
            <div className="flex items-center gap-2 text-rose-400 text-[13px]">
              <XCircle size={15} />
              <span className="truncate">{statusMsg || "Ingest failed"}</span>
            </div>
          )}

          <button type="submit" disabled={isIngesting || !url.trim()}
                  className="w-full h-11 accent-gradient text-white font-bold rounded-xl disabled:opacity-30 disabled:grayscale flex items-center justify-center gap-2 transition-all active:scale-[0.98]">
            {isIngesting ? <><Loader2 size={16} className="animate-spin" /> Indexing…</> : <><Plus size={16} /> Add to Library</>}
          </button>
        </form>
      </div>
    </div>
  );
}

export default function LibraryPage() {
  const { collections, loadingCollections, onDeleted, loadCollections, setSelectedCollections, setActiveVideo, openSidebar } = useCollections();
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [deletingName, setDeletingName] = useState<string | null>(null);
  const [showIngest, setShowIngest] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const filtered = collections.filter(c =>
    c.title.toLowerCase().includes(search.toLowerCase())
  );

  async function handleDelete(name: string) {
    setDeletingName(name);
    try { await deleteCollection(name); onDeleted(name); }
    catch (err) { alert(err instanceof Error ? err.message : String(err)); }
    finally { setDeletingName(null); setConfirmDelete(null); }
  }

  function handleOpenInChat(c: typeof collections[0]) {
    setSelectedCollections([c]);
    setActiveVideo(c);
    router.push("/chat");
  }

  return (
    <>
      {showIngest && (
        <IngestModal
          onClose={() => setShowIngest(false)}
          onDone={() => { loadCollections(); }}
        />
      )}
      {confirmDelete && (
        <ConfirmDeleteModal
          name={confirmDelete}
          onConfirm={() => handleDelete(confirmDelete)}
          onCancel={() => setConfirmDelete(null)}
        />
      )}

      <div className="flex flex-col h-full theme-bg theme-text">
        {/* Header */}
        <header className="flex items-center justify-between px-4 sm:px-6 lg:px-8 py-4 sm:py-5 border-b shrink-0"
                style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-3">
            <button onClick={openSidebar}
                    className="lg:hidden p-1.5 rounded-xl hover:bg-white/5 transition-all shrink-0"
                    style={{ color: "var(--text-dim)" }}
                    aria-label="Open menu">
              <Menu size={18} />
            </button>
            <div>
              <h1 className="text-2xl font-bold font-display" style={{ color: "var(--text)" }}>Video Library</h1>
              <p className="text-[13px] mt-0.5" style={{ color: "var(--text-dim)" }}>
                {collections.length} {collections.length === 1 ? "video" : "videos"} indexed
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <input type="text" value={search} onChange={e => setSearch(e.target.value)}
                     placeholder="Search…"
                     className="pl-9 pr-4 py-2 rounded-xl text-[13px] outline-none w-44 sm:w-56"
                     style={{ background: "var(--bg-hover)", border: "1px solid var(--border)", color: "var(--text)" }} />
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2"
                      style={{ color: "var(--text-dim)" }} />
            </div>
            <button onClick={() => setShowIngest(true)}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] font-bold accent-gradient text-white transition-all active:scale-95">
              <Plus size={15} /> Add Video
            </button>
          </div>
        </header>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-8">
          {loadingCollections && collections.length === 0 ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 size={28} className="animate-spin" style={{ color: "var(--text-dim)" }} />
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 gap-4">
              <Library size={44} style={{ color: "var(--text-dim)", opacity: 0.4 }} />
              <div className="text-center">
                <p className="text-[15px] font-medium mb-1" style={{ color: "var(--text)" }}>
                  {search ? "No videos match your search" : "No videos yet"}
                </p>
                <p className="text-[13px] mb-4" style={{ color: "var(--text-dim)" }}>
                  {!search && "Add a YouTube URL to get started"}
                </p>
                {!search && (
                  <button onClick={() => setShowIngest(true)}
                          className="px-5 py-2.5 rounded-xl font-bold text-[13px] accent-gradient text-white">
                    + Add Video
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filtered.map(c => (
                <article key={c.name}
                         className="group rounded-2xl overflow-hidden border transition-all hover:shadow-xl hover:-translate-y-0.5"
                         style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
                  {/* Thumbnail */}
                  <div className="aspect-video relative overflow-hidden" style={{ background: "var(--bg-hover)" }}>
                    {c.video_id ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={`https://i.ytimg.com/vi/${c.video_id}/hqdefault.jpg`} alt={c.title}
                           className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Library size={32} style={{ color: "var(--text-dim)", opacity: 0.4 }} />
                      </div>
                    )}
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <button onClick={() => handleOpenInChat(c)}
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[12px] font-bold accent-gradient text-white">
                        <MessageSquare size={13} /> Chat
                      </button>
                    </div>
                  </div>

                  {/* Info */}
                  <div className="p-4">
                    <p className="text-[13px] font-semibold leading-snug mb-3 line-clamp-2"
                       style={{ color: "var(--text)" }}>
                      {c.title}
                    </p>
                    <div className="flex items-center gap-2">
                      <button onClick={() => handleOpenInChat(c)}
                              className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-[12px] font-bold accent-gradient text-white transition-all active:scale-95">
                        <MessageSquare size={13} /> Open in Chat
                      </button>
                      <button onClick={() => setConfirmDelete(c.name)}
                              disabled={deletingName === c.name}
                              className="p-2 rounded-xl border transition-all hover:bg-rose-500/10 hover:border-rose-500/30"
                              style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
                        {deletingName === c.name
                          ? <Loader2 size={14} className="animate-spin" />
                          : <Trash2 size={14} className="hover:text-rose-400" />}
                      </button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
