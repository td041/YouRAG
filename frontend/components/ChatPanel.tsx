"use client";

import { useEffect, useRef, useState } from "react";
import { Message, Collection } from "@/lib/types";
import { streamChat } from "@/lib/api";
import { MessageSquare, Send, Trash2, Layers, Zap, BarChart3, Cpu } from "lucide-react";

interface Props {
  collection: Collection | null;
}

let idCounter = 0;
const uid = () => String(++idCounter);

export default function ChatPanel({ collection }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Reset chat when collection changes
  useEffect(() => {
    setMessages([]);
    setInput("");
    setStreaming(false);
    abortRef.current?.();
  }, [collection?.name]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSend() {
    if (!input.trim() || !collection || streaming) return;
    const query = input.trim();
    setInput("");

    const userMsg: Message = { id: uid(), role: "user", content: query };
    const botId = uid();
    const botMsg: Message = { id: botId, role: "assistant", content: "", sources: [] };

    setMessages(prev => [...prev, userMsg, botMsg]);
    setStreaming(true);

    abortRef.current = streamChat(
      query,
      collection.name,
      (chunk) => setMessages(prev =>
        prev.map(m => m.id === botId ? { ...m, content: m.content + chunk } : m)),
      (sources) => setMessages(prev =>
        prev.map(m => m.id === botId ? { ...m, sources } : m)),
      () => { setStreaming(false); abortRef.current = null; },
      (err) => {
        setMessages(prev =>
          prev.map(m => m.id === botId ? { ...m, content: `Error: ${err}` } : m));
        setStreaming(false);
      }
    );
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  function handleClear() {
    abortRef.current?.();
    setMessages([]);
    setStreaming(false);
  }

  const userCount = messages.filter(m => m.role === "user").length;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-5 h-12 border-b border-white/[.07] shrink-0">
        <MessageSquare size={12} className="text-slate-500" />
        <span className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">Chat</span>
        {userCount > 0 && (
          <span className="ml-1 text-[10px] font-mono text-slate-600 bg-white/[.04] border border-white/[.06] px-1.5 py-0.5 rounded-full">
            {userCount}
          </span>
        )}
        {messages.length > 0 && (
          <button
            onClick={handleClear}
            className="ml-auto flex items-center gap-1.5 text-[11px] text-slate-600 hover:text-slate-400 transition-colors"
          >
            <Trash2 size={11} />
            <span>Clear</span>
          </button>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">

        {/* Welcome screen */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-5 text-center py-10 fade-up">
            {collection ? (
              <>
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/15 flex items-center justify-center">
                  <MessageSquare size={22} className="text-indigo-400" />
                </div>
                <div>
                  <p className="text-[15px] font-medium text-slate-300 mb-1.5">
                    Ask about this video
                  </p>
                  <p className="text-[13px] text-slate-500 max-w-xs leading-relaxed">
                    "{collection.title}"
                  </p>
                </div>
                <div className="flex flex-wrap gap-2 justify-center mt-1">
                  {[
                    { icon: <Layers size={10} />, label: "Hybrid RRF" },
                    { icon: <BarChart3 size={10} />, label: "Cross-Encoder" },
                    { icon: <Cpu size={10} />, label: "BGE-M3" },
                    { icon: <Zap size={10} />, label: "Llama 3.3 70B" },
                  ].map(t => (
                    <span key={t.label} className="flex items-center gap-1.5 text-[11px] font-mono text-slate-600 bg-white/[.03] border border-white/[.06] px-2.5 py-1 rounded-full">
                      {t.icon}{t.label}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="w-14 h-14 rounded-2xl bg-white/[.03] border border-white/[.07] flex items-center justify-center">
                  <Layers size={22} className="text-slate-600" />
                </div>
                <div>
                  <p className="text-[15px] font-medium text-slate-400 mb-1.5">No video selected</p>
                  <p className="text-[13px] text-slate-600 max-w-xs leading-relaxed">
                    Add a YouTube URL in the sidebar and select a video to start chatting.
                  </p>
                </div>
              </>
            )}
          </div>
        )}

        {/* Message list */}
        {messages.map((msg, i) => (
          <div key={msg.id} className="fade-up" style={{ animationDelay: `${i * 20}ms` }}>
            {msg.role === "user" ? (
              <div className="flex justify-end">
                <div className="max-w-[78%] bg-[#1e2235] border border-white/[.08] rounded-2xl rounded-br-md px-4 py-3 text-[14px] text-slate-200 leading-relaxed">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div className="flex gap-3 items-start">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5 shadow-md shadow-indigo-500/20">
                  <Layers size={12} className="text-white" strokeWidth={2.5} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[14px] text-slate-300 leading-relaxed">
                    {msg.content}
                    {streaming && i === messages.length - 1 && (
                      <span className="inline-block w-0.5 h-3.5 bg-indigo-400 ml-0.5 -mb-0.5 blink" />
                    )}
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {msg.sources.map((s, si) => (
                        <span key={si} className="text-[11px] font-mono text-slate-600 bg-white/[.03] border border-white/[.06] hover:border-white/[.1] hover:text-slate-400 px-2.5 py-1 rounded-md cursor-default transition-colors">
                          ⏱ {s}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 pb-2 pt-2 border-t border-white/[.07]">
        <div className={`flex items-center gap-2 bg-[#12151b] border rounded-xl px-3 py-2 transition-all ${
          collection
            ? "border-white/[.08] focus-within:border-indigo-500/40 focus-within:ring-1 focus-within:ring-indigo-500/10"
            : "border-white/[.04] opacity-50"
        }`}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px"; }}
            onKeyDown={handleKey}
            disabled={!collection || streaming}
            placeholder={collection ? "Ask about the video… (Enter to send)" : "Select a video to start…"}
            rows={1}
            className="flex-1 bg-transparent text-[13px] text-slate-200 placeholder-slate-600 outline-none resize-none leading-snug max-h-28 disabled:cursor-not-allowed py-0.5"
          />
          <button
            onClick={handleSend}
            disabled={!collection || !input.trim() || streaming}
            className="w-7 h-7 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center shrink-0 transition-colors"
          >
            <Send size={12} className="text-white ml-0.5" />
          </button>
        </div>
        <p className="text-center text-[10px] text-slate-700 mt-1.5 font-mono">
          Hybrid Dense+BM25 → Cross-Encoder → Llama 3.3 70B
        </p>
      </div>
    </div>
  );
}
