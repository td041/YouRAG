"use client";

import { useEffect, useRef, useState } from "react";
import { Message, Collection } from "@/lib/types";
import { streamChat } from "@/lib/api";
import { 
  Send, Trash2, Layers, Zap, 
  Cpu, Sparkles, User, Bot, Clock, Loader2
} from "lucide-react";

interface Props {
  collection: Collection | null;
  onSourceClick?: (time: number) => void;
}

let idCounter = 0;
const uid = () => String(++idCounter);

function parseTimeToSeconds(timeStr: string) {
  const parts = timeStr.split(':').reverse();
  let seconds = 0;
  for (let i = 0; i < parts.length; i++) {
    const val = parseInt(parts[i]);
    if (!isNaN(val)) seconds += val * Math.pow(60, i);
  }
  return seconds;
}

// Render nội dung câu trả lời: thay [mm:ss] bằng số footnote ¹²³ có thể click
function MessageContent({ content, onSourceClick }: { content: string; onSourceClick?: (t: number) => void }) {
  if (!onSourceClick) return <span>{content}</span>;

  const footnotes: { label: string; seconds: number }[] = [];
  const parts = content.split(/(\[\d{1,2}:\d{2}\])/g);

  return (
    <>
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d{1,2}:\d{2})\]$/);
        if (match) {
          const existing = footnotes.findIndex(f => f.label === match[1]);
          let idx: number;
          if (existing !== -1) {
            idx = existing;
          } else {
            footnotes.push({ label: match[1], seconds: parseTimeToSeconds(match[1]) });
            idx = footnotes.length - 1;
          }
          const num = superscriptNumber(idx + 1);
          return (
            <button
              key={i}
              onClick={() => onSourceClick(footnotes[idx].seconds)}
              className="text-indigo-400/80 hover:text-indigo-300 transition-colors cursor-pointer leading-none align-super text-[10px]"
              title={match[1]}
            >
              {num}
            </button>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function superscriptNumber(n: number): string {
  const map: Record<string, string> = { '1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','0':'⁰' };
  return String(n).split('').map(d => map[d] ?? d).join('');
}

export default function ChatPanel({ collection, onSourceClick }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setMessages([]);
    setInput("");
    setStreaming(false);
    setSessionId(undefined);
    abortRef.current?.();
  }, [collection?.name]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSend(overrideQuery?: string) {
    const query = overrideQuery || input.trim();
    if (!query || !collection || streaming) return;
    
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const userMsg: Message = { id: uid(), role: "user", content: query };
    const botId = uid();
    const botMsg: Message = { id: botId, role: "assistant", content: "", sources: [] };

    setMessages(prev => [...prev, userMsg, botMsg]);
    setStreaming(true);

    abortRef.current = streamChat(
      query,
      collection.name,
      sessionId,
      (chunk) => setMessages(prev =>
        prev.map(m => m.id === botId ? { ...m, content: m.content + chunk } : m)),
      (sources) => setMessages(prev =>
        prev.map(m => m.id === botId ? { ...m, sources } : m)),
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

  return (
    <div className="flex flex-col h-full bg-[#050608] relative">
      {/* Dynamic Header */}
      <div className="flex items-center justify-between px-6 h-14 border-b border-white/[0.04] glass-light z-10 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex -space-x-1">
            <div className="w-6 h-6 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
              <Sparkles size={10} className="text-indigo-400" />
            </div>
            <div className="w-6 h-6 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center translate-x-1">
              <Zap size={10} className="text-purple-400" />
            </div>
          </div>
          <span className="text-[12px] font-bold tracking-tight text-slate-300 font-display">Intelligence Interface</span>
        </div>
        
        {messages.length > 0 && (
          <button
            onClick={() => { abortRef.current?.(); setMessages([]); }}
            className="group flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition-all"
          >
            <Trash2 size={12} className="text-slate-500 group-hover:text-rose-400" />
            <span className="text-[11px] font-semibold text-slate-500 group-hover:text-rose-400">Clear Session</span>
          </button>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-8 custom-scrollbar">
        <div className="max-w-3xl mx-auto space-y-10">
          
          {/* Welcome screen */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center animate-slide-up">
              <div className="relative mb-8">
                <div className="w-20 h-20 rounded-[2.5rem] bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center rotate-3 animate-pulse">
                  <Bot size={32} className="text-indigo-400 -rotate-3" />
                </div>
                <div className="absolute -bottom-2 -right-2 w-10 h-10 rounded-2xl bg-purple-500 border-4 border-[#050608] flex items-center justify-center">
                  <Zap size={16} className="text-white" />
                </div>
              </div>
              
              <h2 className="text-2xl font-bold text-white font-display mb-3 tracking-tight">
                {collection ? "How can I help you today?" : "Select a video to begin"}
              </h2>
              <p className="text-slate-500 text-[14px] max-w-sm leading-relaxed mb-8">
                {collection 
                  ? `Ask me anything about "${collection.title}". I've analyzed the content and metadata for you.`
                  : "Connect a YouTube video from your library to start an intelligent conversation."}
              </p>

              {collection && (
                <div className="grid grid-cols-2 gap-3 w-full max-w-md">
                  {[
                    "Summarize main points",
                    "List key takeaways",
                    "Explain technical terms",
                    "Draft an outline"
                  ].map(hint => (
                    <button 
                      key={hint}
                      onClick={() => setInput(hint)}
                      className="px-4 py-3 rounded-2xl bg-white/[0.02] border border-white/[0.06] hover:border-indigo-500/40 hover:bg-indigo-500/[0.02] text-[12px] text-slate-400 hover:text-indigo-300 text-left transition-all group"
                    >
                      <span className="opacity-60 group-hover:opacity-100 mr-2">✦</span>
                      {hint}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Message list */}
          {messages.map((msg, i) => (
            <div key={msg.id} className="animate-slide-up group">
              <div className={`flex gap-5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                
                {/* Avatar */}
                <div className={`shrink-0 w-8 h-8 rounded-xl flex items-center justify-center border transition-all ${
                  msg.role === "user" 
                    ? "bg-slate-800 border-white/10 text-slate-400 group-hover:scale-110" 
                    : "bg-indigo-600 border-indigo-400/30 text-white shadow-[0_0_15px_rgba(99,102,241,0.3)] group-hover:scale-110"
                }`}>
                  {msg.role === "user" ? <User size={14} /> : <Layers size={14} />}
                </div>

                {/* Content */}
                <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  <div className={`px-5 py-3.5 rounded-[22px] text-[14.5px] leading-relaxed shadow-sm transition-all ${
                    msg.role === "user"
                      ? "bg-[#161a22] text-white rounded-tr-none border border-white/[0.05]"
                      : "bg-[#0d1117] text-slate-200 rounded-tl-none border border-white/[0.08] hover:border-white/[0.15]"
                  }`}>
                    {msg.role === "assistant"
                      ? <MessageContent content={msg.content} onSourceClick={onSourceClick} />
                      : msg.content}
                    {streaming && i === messages.length - 1 && (
                      <span className="inline-block w-1.5 h-4 bg-indigo-500 ml-1 rounded-sm blink align-middle" />
                    )}
                  </div>

                  {/* Sources / Metadata */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-1">
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.03] border border-white/[0.06] text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                        <Clock size={10} /> Reference Points
                      </div>
                      {msg.sources.map((s, si) => (
                        <button 
                          key={si} 
                          onClick={() => {
                            if (onSourceClick) {
                              const timePart = s.split('–')[0].trim();
                              onSourceClick(parseTimeToSeconds(timePart));
                            }
                          }}
                          className="px-3 py-1 rounded-full bg-indigo-500/5 border border-indigo-500/10 hover:bg-indigo-500/10 hover:border-indigo-500/30 text-[11px] font-mono text-indigo-400 transition-all cursor-pointer"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Suggested Questions */}
                  {msg.suggestions && msg.suggestions.length > 0 && (
                    <div className="flex flex-col gap-2 mt-3 w-full">
                      <div className="flex flex-wrap gap-2">
                        {msg.suggestions.map((sug, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSend(sug)}
                            className="px-3 py-2 rounded-xl bg-white/[0.02] border border-white/[0.06] hover:border-indigo-500/40 hover:bg-indigo-500/[0.02] text-[12px] text-slate-300 hover:text-indigo-300 transition-all text-left flex items-center group/btn"
                          >
                            <Sparkles size={12} className="text-indigo-500/50 mr-1.5 group-hover/btn:text-indigo-400 transition-colors" />
                            {sug}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={bottomRef} className="h-4" />
        </div>
      </div>

      {/* Input area - Premium Claude-style */}
      <div className="px-6 pb-8 pt-4">
        <div className="max-w-3xl mx-auto relative group">
          <div className={`absolute inset-0 bg-indigo-500/5 rounded-3xl blur-2xl transition-opacity duration-500 ${
            input.trim() ? "opacity-100" : "opacity-0"
          }`} />
          
          <div className={`relative flex flex-col bg-[#0d1117] border rounded-[2rem] p-2 transition-all duration-300 ${
            collection
              ? "border-white/[0.08] focus-within:border-indigo-500/50 focus-within:shadow-[0_0_30px_rgba(99,102,241,0.1)] focus-within:ring-4 focus-within:ring-indigo-500/5"
              : "opacity-40 grayscale pointer-events-none"
          }`}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => { 
                setInput(e.target.value); 
                e.target.style.height = "auto"; 
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px"; 
              }}
              onKeyDown={handleKey}
              placeholder={collection ? "Send a message to your AI assistant..." : "Select a video from the library to begin"}
              className="w-full bg-transparent px-5 py-4 text-[14.5px] text-white placeholder-slate-600 outline-none resize-none min-h-[60px] max-h-48 leading-relaxed scrollbar-none"
              rows={1}
            />
            
            <div className="flex items-center justify-between px-3 pb-2 pt-1">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/[0.03] text-[10px] font-bold text-slate-500 border border-white/[0.05]">
                <Cpu size={10} className="text-indigo-500" />
                Llama 3.3 70B Adaptive
              </div>
              
              <button
                onClick={() => handleSend()}
                disabled={!collection || !input.trim() || streaming}
                className={`w-10 h-10 rounded-2xl flex items-center justify-center transition-all ${
                  input.trim() && !streaming
                    ? "accent-gradient text-white shadow-lg shadow-indigo-500/20 active:scale-90"
                    : "bg-white/[0.03] text-slate-700 cursor-not-allowed"
                }`}
              >
                {streaming ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} className={input.trim() ? "ml-0.5" : ""} />}
              </button>
            </div>
          </div>
          
          <p className="text-center text-[10px] text-slate-700 mt-4 tracking-tight font-medium uppercase">
            Intelligent Video Context Engine • Hybrid Retrieval Protocol
          </p>
        </div>
      </div>
    </div>
  );
}

