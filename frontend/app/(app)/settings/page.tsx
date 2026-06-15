"use client";
import { Key, Cpu, Database, Sliders } from "lucide-react";

const SECTIONS = [
  {
    title: "LLM Configuration",
    icon: Cpu,
    items: [
      { label: "Primary Model",  value: "llama-3.3-70b-versatile (Groq)" },
      { label: "Fast Model",     value: "llama-3.1-8b-instant (Groq)" },
      { label: "Temperature",    value: "0.7" },
      { label: "Max Tokens",     value: "1500" },
    ],
  },
  {
    title: "API Keys",
    icon: Key,
    items: [
      { label: "GROQ_API_KEY",   value: "Required — set in .env" },
      { label: "JINA_API_KEY",   value: "Optional — enables Late Chunking" },
      { label: "GEMINI_API_KEY", value: "Optional — enables Visual Frame RAG" },
    ],
  },
  {
    title: "Vector Database",
    icon: Database,
    items: [
      { label: "Provider",        value: "Qdrant (local Docker)" },
      { label: "Embedding Model", value: "BAAI/bge-m3 (1024-dim)" },
      { label: "Distance Metric", value: "Cosine HNSW" },
      { label: "Reranker",        value: "mmarco-mMiniLMv2 (GPU only)" },
    ],
  },
  {
    title: "RAG Pipeline",
    icon: Sliders,
    items: [
      { label: "Retrieval Top-K",  value: "10" },
      { label: "Hybrid Alpha",     value: "0.5 (50% dense / 50% sparse)" },
      { label: "Semantic Cache",   value: "Enabled (Qdrant-backed)" },
      { label: "Graph RAG",        value: "Enabled (rule-based entities)" },
    ],
  },
] as const;

export default function SettingsPage() {
  return (
    <div className="flex flex-col h-full theme-bg theme-text">
      <header className="px-6 sm:px-8 py-5 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        <h1 className="text-2xl font-bold font-display" style={{ color: "var(--text)" }}>Settings</h1>
        <p className="text-[13px] mt-0.5" style={{ color: "var(--text-dim)" }}>
          System configuration — edit <code className="px-1 py-0.5 rounded text-xs" style={{ background: "var(--bg-hover)" }}>.env</code> to change values
        </p>
      </header>

      <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-5">
        {SECTIONS.map(({ title, icon: Icon, items }) => (
          <div key={title} className="rounded-2xl border overflow-hidden"
               style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
            <div className="flex items-center gap-3 px-5 py-4 border-b"
                 style={{ borderColor: "var(--border)" }}>
              <Icon size={16} className="text-indigo-400" />
              <h2 className="text-[14px] font-semibold" style={{ color: "var(--text)" }}>{title}</h2>
            </div>
            <div>
              {items.map(({ label, value }, i) => (
                <div key={label}
                     className={`flex items-center justify-between px-5 py-3.5 ${i < items.length - 1 ? "border-b" : ""}`}
                     style={{ borderColor: "var(--border)" }}>
                  <span className="text-[13px]" style={{ color: "var(--text-muted)" }}>{label}</span>
                  <span className="text-[12px] font-mono" style={{ color: "var(--text-dim)" }}>{value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}

        <p className="text-[12px] text-center pb-2" style={{ color: "var(--text-dim)" }}>
          After editing <code className="px-1 py-0.5 rounded text-xs" style={{ background: "var(--bg-hover)" }}>.env</code>,
          run <code className="px-1 py-0.5 rounded text-xs" style={{ background: "var(--bg-hover)" }}>docker compose up -d --build backend</code> to apply.
        </p>
      </div>
    </div>
  );
}
