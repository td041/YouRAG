"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { Message, Collection, Source } from "@/lib/types";
import { streamChat, fetchSuggestions } from "@/lib/api";
import {
  Send, Trash2, Layers, Zap,
  Cpu, Sparkles, User, Bot, Clock, Loader2, ExternalLink, Search, Eye
} from "lucide-react";

interface Props {
  collections: Collection[];
  activeVideo: Collection | null;
  theme: "dark" | "light";
  onSourceClick?: (time: number, videoId?: string) => void;
}

let idCounter = 0;
const uid = () => String(++idCounter);

function parseTimeToSeconds(timeStr: string) {
  const parts = timeStr.split(":").reverse();
  let seconds = 0;
  for (let i = 0; i < parts.length; i++) {
    const val = parseInt(parts[i]);
    if (!isNaN(val)) seconds += val * Math.pow(60, i);
  }
  return seconds;
}

function superscriptNumber(n: number): string {
  const map: Record<string, string> = { "1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","0":"⁰" };
  return String(n).split("").map(d => map[d] ?? d).join("");
}

function MessageContent({ content, onSourceClick }: { content: string; onSourceClick?: (t: number) => void }) {
  // Match [mm:ss] or [mm:ss - mm:ss] (range) — capture only the start time
  const TS_RE = /\[(\d{1,2}:\d{2})(?:\s*[-–]\s*\d{1,2}:\d{2})?\]/g;

  // Pre-compute timestamp footnotes in document order (stable indices across renders)
  const seenLabels: string[] = [];
  const footnoteMap = new Map<string, number>();
  let m: RegExpExecArray | null;
  while ((m = TS_RE.exec(content)) !== null) {
    const label = m[1];
    if (!footnoteMap.has(label)) {
      footnoteMap.set(label, seenLabels.length);
      seenLabels.push(label);
    }
  }

  // Replace timestamp patterns with a safe HTML tag that rehype-raw will pass through
  const processed = content.replace(TS_RE, '<ts data-ts="$1"></ts>');

  return (
    <div className="prose prose-sm max-w-none leading-relaxed
      prose-p:my-1 prose-p:last:mb-0
      prose-ul:my-1 prose-ul:pl-4
      prose-ol:my-1 prose-ol:pl-4
      prose-li:my-0
      prose-strong:font-semibold
      prose-code:text-xs prose-code:px-1 prose-code:py-0.5 prose-code:rounded
      prose-pre:text-xs prose-pre:p-3 prose-pre:rounded-lg prose-pre:overflow-x-auto
      [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={({
          // Custom timestamp citation button
          ts: (props: { "data-ts"?: string } & React.HTMLAttributes<HTMLElement>) => {
            const time = props["data-ts"] ?? "";
            const idx = footnoteMap.get(time) ?? 0;
            const seconds = parseTimeToSeconds(time);
            return (
              <button
                onClick={() => onSourceClick?.(seconds)}
                className="text-indigo-400/80 hover:text-indigo-300 transition-colors cursor-pointer leading-none align-super text-[10px]"
                title={`Jump to ${time}`}
              >
                {superscriptNumber(idx + 1)}
              </button>
            );
          },
          // Inline code vs code block
          code: ({ children, className }: React.HTMLAttributes<HTMLElement>) => {
            const isBlock = typeof className === "string" && className.startsWith("language-");
            return isBlock
              ? <code className="block bg-black/20 rounded-lg p-3 text-xs font-mono overflow-x-auto">{children}</code>
              : <code className="bg-black/20 dark:bg-white/10 px-1 py-0.5 rounded text-[0.8em] font-mono">{children}</code>;
          },
        } as Parameters<typeof ReactMarkdown>[0]["components"])}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}

export default function ChatPanel({ collections, activeVideo, theme, onSourceClick }: Props) {
  const collection = collections[0] ?? null;
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [welcomeSuggestions, setWelcomeSuggestions] = useState<string[]>([]);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isDark = theme === "dark";
  const borderColor = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.06)";
  const textMain = isDark ? "#f0f2f5" : "#0f172a";
  const textMuted = isDark ? "#94a3b8" : "#475569";
  const textDim = isDark ? "#475569" : "#94a3b8";
  const msgUserBg = isDark ? "#161a22" : "#eef0f6";
  const msgBotBg = isDark ? "#0d1117" : "#ffffff";
  const msgBotBorder = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)";
  const inputBg = isDark ? "#0d1117" : "#ffffff";
  const chipBg = isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)";

  const collectionsKey = collections.map(c => c.name).join(",");

  useEffect(() => {
    setMessages([]);
    setInput("");
    setStreaming(false);
    setSessionId(undefined);
    setWelcomeSuggestions([]);
    abortRef.current?.();

    if (!collection) return;
    setLoadingSuggestions(true);
    fetchSuggestions(collection.name)
      .then(s => setWelcomeSuggestions(s))
      .finally(() => setLoadingSuggestions(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collectionsKey]);

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    // Instant scroll during streaming to avoid jank; smooth for new messages
    if (streaming) {
      el.scrollTop = el.scrollHeight;
    } else {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streaming]);

  function handleSend(overrideQuery?: string) {
    const query = overrideQuery || input.trim();
    if (!query || collections.length === 0 || streaming) return;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMsg: Message = { id: uid(), role: "user", content: query };
    const botId = uid();
    const botMsg: Message = { id: botId, role: "assistant", content: "", sources: [] };

    setMessages(prev => [...prev, userMsg, botMsg]);
    setStreaming(true);

    abortRef.current = streamChat(
      query,
      collections.map(c => c.name),
      sessionId,
      (chunk) => setMessages(prev =>
        prev.map(m => m.id === botId ? { ...m, content: m.content + chunk } : m)),
      (rawSources) => {
        const sources: Source[] = rawSources.map((s) => {
          if (typeof s === "string") {
            return { label: s, start_time: parseTimeToSeconds(s.split("–")[0].trim()), video_id: null, title: null };
          }
          return s as Source;
        });
        setMessages(prev => prev.map(m => m.id === botId ? { ...m, sources } : m));
      },
      (suggestions) => setMessages(prev =>
        prev.map(m => m.id === botId ? { ...m, suggestions } : m)),
      (id) => setSessionId(id),
      () => { setStreaming(false); abortRef.current = null; },
      (err) => {
        setMessages(prev =>
          prev.map(m => m.id === botId ? { ...m, content: `System Error: ${err}` } : m));
        setStreaming(false);
      }
    );
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  function openInYouTube(src: Source) {
    const vid = src.video_id ?? activeVideo?.video_id;
    if (!vid) return;
    window.open(`https://youtube.com/watch?v=${vid}&t=${Math.floor(src.start_time)}s`, "_blank", "noopener");
  }

  return (
    <div className="flex flex-col h-full relative" style={{ background: isDark ? "#050608" : "#f8f9fc" }}>

      {/* Header */}
      <div className="flex items-center justify-between px-6 h-14 border-b glass-light z-10 shrink-0" style={{ borderColor }}>
        <div className="flex items-center gap-3">
          <div className="flex -space-x-1">
            <div className="w-6 h-6 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
              <Sparkles size={10} className="text-indigo-400" />
            </div>
            <div className="w-6 h-6 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center translate-x-1">
              <Zap size={10} className="text-purple-400" />
            </div>
          </div>
          <span className="text-[12px] font-bold tracking-tight font-display" style={{ color: textMuted }}>Intelligence Interface</span>
        </div>

        {messages.length > 0 && (
          <button
            onClick={() => { abortRef.current?.(); setMessages([]); }}
            className="group flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition-all"
          >
            <Trash2 size={12} style={{ color: textDim }} className="group-hover:text-rose-400 transition-colors" />
            <span className="text-[11px] font-semibold group-hover:text-rose-400 transition-colors" style={{ color: textDim }}>Clear</span>
          </button>
        )}
      </div>

      {/* Messages area */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 sm:py-8 custom-scrollbar">
        <div className="max-w-3xl mx-auto space-y-8 sm:space-y-10">

          {/* Welcome screen */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-center animate-slide-up">
              <div className="relative mb-8">
                <div className="w-20 h-20 rounded-[2.5rem] bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center rotate-3 animate-pulse">
                  <Bot size={32} className="text-indigo-400 -rotate-3" />
                </div>
                <div className="absolute -bottom-2 -right-2 w-10 h-10 rounded-2xl bg-purple-500 border-4 flex items-center justify-center" style={{ borderColor: isDark ? "#050608" : "#f8f9fc" }}>
                  <Zap size={16} className="text-white" />
                </div>
              </div>

              <h2 className="text-xl sm:text-2xl font-bold font-display mb-3 tracking-tight" style={{ color: textMain }}>
                {collections.length > 0 ? "How can I help you today?" : "Select a video to begin"}
              </h2>
              <p className="text-[14px] max-w-sm leading-relaxed mb-8" style={{ color: textDim }}>
                {collections.length > 1
                  ? `Searching across ${collections.length} videos simultaneously.`
                  : collections.length === 1
                  ? `Ask me anything about "${collections[0].title}".`
                  : "Connect a YouTube video from your library to start an intelligent conversation."}
              </p>

              {collections.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-md">
                  {loadingSuggestions ? (
                    Array.from({ length: 4 }).map((_, i) => (
                      <div key={i} className="px-4 py-3 rounded-2xl h-[46px] animate-shimmer"
                           style={{ background: chipBg, border: `1px solid ${msgBotBorder}` }} />
                    ))
                  ) : welcomeSuggestions.length > 0 ? (
                    welcomeSuggestions.map(hint => (
                      <button
                        key={hint}
                        onClick={() => handleSend(hint)}
                        className="px-4 py-3 rounded-2xl text-[12px] text-left transition-all group hover:border-indigo-500/40"
                        style={{ background: chipBg, border: `1px solid ${msgBotBorder}`, color: textMuted }}
                      >
                        <Sparkles size={11} className="inline mr-2 text-indigo-500/50 group-hover:text-indigo-400 transition-colors" />
                        {hint}
                      </button>
                    ))
                  ) : (
                    ["Tóm tắt nội dung chính", "Các điểm quan trọng", "Giải thích thuật ngữ", "Phác thảo nội dung"].map(hint => (
                      <button
                        key={hint}
                        onClick={() => handleSend(hint)}
                        className="px-4 py-3 rounded-2xl text-[12px] text-left transition-all group hover:border-indigo-500/40"
                        style={{ background: chipBg, border: `1px solid ${msgBotBorder}`, color: textMuted }}
                      >
                        <span className="opacity-60 group-hover:opacity-100 mr-2">✦</span>
                        {hint}
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>
          )}

          {/* Message list */}
          {messages.map((msg, i) => (
            <div key={msg.id} className="animate-slide-up group">
              <div className={`flex gap-4 sm:gap-5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>

                {/* Avatar */}
                <div className={`shrink-0 w-8 h-8 rounded-xl flex items-center justify-center border transition-all ${
                  msg.role === "user"
                    ? "group-hover:scale-110"
                    : "bg-indigo-600 border-indigo-400/30 text-white shadow-[0_0_15px_rgba(99,102,241,0.3)] group-hover:scale-110"
                }`}
                  style={msg.role === "user" ? { background: isDark ? "#1e293b" : "#e2e8f0", borderColor: msgBotBorder, color: textDim } : {}}>
                  {msg.role === "user" ? <User size={14} /> : <Layers size={14} />}
                </div>

                {/* Content bubble */}
                <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  <div className={`px-4 sm:px-5 py-3 sm:py-3.5 rounded-[22px] text-[14px] sm:text-[14.5px] leading-relaxed shadow-sm transition-all ${
                    msg.role === "user" ? "rounded-tr-none" : "rounded-tl-none"
                  }`}
                    style={{
                      background: msg.role === "user" ? msgUserBg : msgBotBg,
                      color: textMain,
                      border: `1px solid ${msg.role === "user" ? "rgba(255,255,255,0.05)" : msgBotBorder}`,
                    }}>
                    {msg.role === "assistant" ? (
                      msg.content === "" && streaming && i === messages.length - 1 ? (
                        // TTFT indicator — retrieval phase before first token arrives
                        <span className="flex items-center gap-2 text-[13px]" style={{ color: textDim }}>
                          <Search size={13} className="animate-pulse text-indigo-400" />
                          <span className="animate-pulse">Searching...</span>
                        </span>
                      ) : (
                        <>
                          <MessageContent content={msg.content} onSourceClick={onSourceClick} />
                          {streaming && i === messages.length - 1 && (
                            <span className="inline-block w-1.5 h-4 bg-indigo-500 ml-1 rounded-sm blink align-middle" />
                          )}
                        </>
                      )
                    ) : msg.content}
                  </div>

                  {/* Source chips — clickable to seek + open in YouTube */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-1">
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest"
                           style={{ background: chipBg, border: `1px solid ${msgBotBorder}`, color: textDim }}>
                        <Clock size={10} /> Sources
                      </div>
                      {msg.sources.map((src, si) => {
                        const hasVideo = !!(src.video_id ?? activeVideo?.video_id);
                        const isVisual = src.chunk_type === "visual";
                        const borderCol = isVisual
                          ? "rgba(139,92,246,0.25)"
                          : "rgba(99,102,241,0.15)";
                        const btnClass = isVisual
                          ? "px-3 py-1 bg-violet-500/5 hover:bg-violet-500/10 text-[11px] font-mono text-violet-400 transition-all"
                          : "px-3 py-1 bg-indigo-500/5 hover:bg-indigo-500/10 text-[11px] font-mono text-indigo-400 transition-all";
                        return (
                          <div key={si} className="flex items-center rounded-full overflow-hidden" style={{ border: `1px solid ${borderCol}` }}>
                            {isVisual && (
                              <span className="pl-2 pr-1 flex items-center" title="Visual frame source">
                                <Eye size={9} className="text-violet-400/70" />
                              </span>
                            )}
                            <button
                              onClick={() => onSourceClick?.(src.start_time, src.video_id ?? undefined)}
                              className={btnClass}
                              title={`Seek to ${src.label}${isVisual ? " (visual frame)" : ""}${src.title ? ` — ${src.title}` : ""}`}
                            >
                              {collections.length > 1 && src.title
                                ? `${src.title.slice(0, 15)}… ${src.label}`
                                : src.label}
                            </button>
                            {hasVideo && (
                              <button
                                onClick={() => openInYouTube(src)}
                                className="px-2 py-1 bg-indigo-500/5 hover:bg-indigo-500/15 border-l border-indigo-500/15 text-indigo-400/60 hover:text-indigo-400 transition-all"
                                title="Open in YouTube"
                              >
                                <ExternalLink size={10} />
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Suggested questions */}
                  {msg.suggestions && msg.suggestions.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {msg.suggestions.map((sug, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSend(sug)}
                          className="px-3 py-2 rounded-xl text-[12px] text-left transition-all flex items-center group/btn hover:border-indigo-500/40"
                          style={{ background: chipBg, border: `1px solid ${msgBotBorder}`, color: textMuted }}
                        >
                          <Sparkles size={12} className="text-indigo-500/50 mr-1.5 group-hover/btn:text-indigo-400 transition-colors shrink-0" />
                          {sug}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={bottomRef} className="h-4" />
        </div>
      </div>

      {/* Input area */}
      <div className="px-4 sm:px-6 pb-6 sm:pb-8 pt-4">
        <div className="max-w-3xl mx-auto relative group">
          <div className={`absolute inset-0 bg-indigo-500/5 rounded-3xl blur-2xl transition-opacity duration-500 ${input.trim() ? "opacity-100" : "opacity-0"}`} />

          <div className={`relative flex flex-col rounded-[2rem] p-2 transition-all duration-300 ${
            collections.length > 0 ? "focus-within:shadow-[0_0_30px_rgba(99,102,241,0.1)] focus-within:ring-4 focus-within:ring-indigo-500/5" : "opacity-40 grayscale pointer-events-none"
          }`}
            style={{
              background: inputBg,
              border: `1px solid ${msgBotBorder}`,
            }}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
              }}
              onKeyDown={handleKey}
              placeholder={collections.length > 0 ? "Send a message..." : "Select a video to begin"}
              className="w-full bg-transparent px-4 sm:px-5 py-4 text-[14px] sm:text-[14.5px] outline-none resize-none min-h-[56px] max-h-48 leading-relaxed scrollbar-none placeholder-slate-600"
              style={{ color: textMain }}
              rows={1}
            />

            <div className="flex items-center justify-between px-3 pb-2 pt-1">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[10px] font-bold border" style={{ background: chipBg, borderColor: msgBotBorder, color: textDim }}>
                <Cpu size={10} className="text-indigo-500" />
                <span className="hidden sm:inline">Llama 3.3 70B</span>
                <span className="sm:hidden">LLM</span>
              </div>

              <button
                onClick={() => handleSend()}
                disabled={collections.length === 0 || !input.trim() || streaming}
                className={`w-10 h-10 rounded-2xl flex items-center justify-center transition-all ${
                  input.trim() && !streaming
                    ? "accent-gradient text-white shadow-lg shadow-indigo-500/20 active:scale-90"
                    : "text-slate-700 cursor-not-allowed"
                }`}
                style={!(input.trim() && !streaming) ? { background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.05)" } : {}}
              >
                {streaming ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} className={input.trim() ? "ml-0.5" : ""} />}
              </button>
            </div>
          </div>

          <p className="text-center text-[10px] mt-3 tracking-tight font-medium uppercase hidden sm:block" style={{ color: isDark ? "#1e293b" : "#cbd5e1" }}>
            Intelligent Video Context Engine • Hybrid Retrieval Protocol
          </p>
        </div>
      </div>
    </div>
  );
}
