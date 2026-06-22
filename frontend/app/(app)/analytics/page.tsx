"use client";
import { useEffect, useState } from "react";
import { BarChart2, Zap, Database, RefreshCw, ExternalLink, AlertCircle, Menu, MessageSquare, Upload, Trash2 } from "lucide-react";
import { useCollections } from "@/lib/collections-context";
import { fetchBenchmarkReport } from "@/lib/api";
import { getQueryPerf, getIngestPerf, clearPerf, type QueryPerf, type IngestPerf } from "@/lib/perf-store";

const METRIC_LABELS: Record<string, string> = {
  faithfulness:       "Faithfulness",
  answer_relevancy:   "Answer Relevancy",
  context_precision:  "Context Precision",
  context_recall:     "Context Recall",
  factual_correctness:"Factual Correctness",
};

const MODE_LABELS: Record<string, { label: string; color: string; bg: string }> = {
  "0_naive":    { label: "Naive",    color: "#94a3b8", bg: "rgba(148,163,184,0.15)" },
  "1_hybrid":   { label: "Hybrid",   color: "#818cf8", bg: "rgba(129,140,248,0.15)" },
  "2_advanced": { label: "Advanced", color: "#34d399", bg: "rgba(52,211,153,0.15)" },
};

interface ModeResult {
  ragas_scores: Record<string, number>;
  avg_latency_s: number;
  total_questions: number;
}

interface Report {
  available: boolean;
  generated_at?: string;
  collection?: string;
  [key: string]: unknown;
}

function ScoreBar({ value, color }: { value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="text-[12px] font-bold w-10 text-right tabular-nums" style={{ color }}>
        {pct}%
      </span>
    </div>
  );
}

function avgScore(scores: Record<string, number>): number {
  const vals = Object.values(scores);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

function avg(nums: (number | undefined)[]): number {
  const valid = nums.filter((n): n is number => n !== undefined && n > 0);
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
}

function Sparkline({ values, color = "#818cf8" }: { values: number[]; color?: string }) {
  if (values.length < 2) return null;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const w = 80, h = 28, pad = 3;
  const pts = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (w - pad * 2);
    const y = pad + ((max - v) / range) * (h - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={w} height={h} className="opacity-60">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function AnalyticsPage() {
  const { collections, openSidebar } = useCollections();
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [queryPerf, setQueryPerf] = useState<QueryPerf[]>([]);
  const [ingestPerf, setIngestPerf] = useState<IngestPerf[]>([]);

  useEffect(() => {
    fetchBenchmarkReport()
      .then(d => setReport(d ?? { available: false }))
      .catch(() => setReport({ available: false }))
      .finally(() => setLoading(false));
    setQueryPerf(getQueryPerf());
    setIngestPerf(getIngestPerf());
  }, []);

  const modes = report?.available
    ? Object.entries(MODE_LABELS)
        .map(([key]) => [key, report[key] as ModeResult | undefined] as const)
        .filter(([, v]) => v?.ragas_scores)
    : [];

  const topMode = modes.reduce<[string, ModeResult] | null>((best, [key, data]) => {
    if (!data) return best;
    const avg = avgScore(data.ragas_scores);
    if (!best) return [key, data];
    return avg > avgScore(best[1].ragas_scores) ? [key, data] : best;
  }, null);

  return (
    <div className="flex flex-col h-full theme-bg theme-text">
      <header className="flex items-start gap-3 px-4 sm:px-6 lg:px-8 py-4 sm:py-5 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        <button onClick={openSidebar}
                className="lg:hidden p-1.5 rounded-xl hover:bg-white/5 transition-all shrink-0 mt-1"
                style={{ color: "var(--text-dim)" }}
                aria-label="Open menu">
          <Menu size={18} />
        </button>
        <div>
          <h1 className="text-2xl font-bold font-display" style={{ color: "var(--text)" }}>Analytics</h1>
          <p className="text-[13px] mt-0.5" style={{ color: "var(--text-dim)" }}>
            RAGAS benchmark · system metrics · performance comparison
          </p>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6">

        {/* Top stats */}
        {(() => {
          const avgQueryMs = avg(queryPerf.filter(q => !q.cached).map(q => q.latency.total));
          const avgTtft = avg(queryPerf.filter(q => !q.cached).map(q => q.latency.ttft));
          const avgIngestS = avg(ingestPerf.map(i => i.latency.total_s));
          return (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { icon: Database,   label: "Indexed Videos",   value: String(collections.length),                             color: "text-indigo-400" },
                { icon: MessageSquare, label: "Avg Query Time", value: avgQueryMs ? `${(avgQueryMs/1000).toFixed(1)}s`  : "—", color: "text-purple-400" },
                { icon: Zap,        label: "Avg TTFT",          value: avgTtft    ? `${avgTtft.toFixed(0)}ms`           : "—", color: "text-emerald-400" },
                { icon: Upload,     label: "Avg Ingest Time",   value: avgIngestS ? `${avgIngestS.toFixed(1)}s`         : "—", color: "text-amber-400" },
              ].map(({ icon: Icon, label, value, color }) => (
                <div key={label} className="rounded-2xl p-5 border"
                     style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
                  <Icon size={18} className={`mb-3 ${color}`} />
                  <p className="text-2xl font-bold mb-1" style={{ color: "var(--text)" }}>{value}</p>
                  <p className="text-[12px]" style={{ color: "var(--text-dim)" }}>{label}</p>
                </div>
              ))}
            </div>
          );
        })()}

        {/* ── Real-time Performance ── */}
        {(queryPerf.length > 0 || ingestPerf.length > 0) && (
          <div className="rounded-2xl border overflow-hidden" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: "var(--border)" }}>
              <div className="flex items-center gap-2">
                <Zap size={15} className="text-indigo-400" />
                <h2 className="text-[14px] font-semibold" style={{ color: "var(--text)" }}>Live Performance</h2>
                <span className="text-[10px] px-2 py-0.5 rounded-full font-bold"
                      style={{ background: "var(--bg-hover)", color: "var(--text-dim)" }}>
                  last {Math.max(queryPerf.length, ingestPerf.length)} records
                </span>
              </div>
              <button onClick={() => { clearPerf(); setQueryPerf([]); setIngestPerf([]); }}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-semibold hover:bg-rose-500/10 transition-all"
                      style={{ color: "var(--text-dim)", border: "1px solid var(--border)" }}>
                <Trash2 size={11} /> Clear
              </button>
            </div>

            <div className="p-5 grid sm:grid-cols-2 gap-5">
              {/* Query latency table */}
              {queryPerf.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-[12px] font-bold uppercase tracking-widest" style={{ color: "var(--text-dim)" }}>
                      Queries ({queryPerf.length})
                    </p>
                    <Sparkline values={queryPerf.filter(q=>!q.cached).map(q=>q.latency.total??0).slice(0,12).reverse()} color="#818cf8" />
                  </div>
                  <div className="space-y-2 max-h-52 overflow-y-auto custom-scrollbar pr-1">
                    {queryPerf.slice(0, 15).map((q, i) => (
                      <div key={i} className="flex items-start gap-2 py-2 border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                        <div className="flex-1 min-w-0">
                          <p className="text-[11px] font-medium truncate" style={{ color: "var(--text-muted)" }}>
                            {q.cached && <span className="text-amber-400 mr-1">⚡</span>}
                            {q.query.slice(0, 55)}{q.query.length > 55 ? "…" : ""}
                          </p>
                          {!q.cached && (
                            <div className="flex items-center gap-2 mt-1 flex-wrap">
                              {[
                                { k: "search", label: "search", color: "#818cf8" },
                                { k: "rerank", label: "rerank", color: "#a78bfa" },
                                { k: "ttft",   label: "ttft",   color: "#34d399" },
                                { k: "llm",    label: "llm",    color: "#f59e0b" },
                              ].filter(s => q.latency[s.k as keyof typeof q.latency]).map(s => (
                                <span key={s.k} className="text-[10px] font-mono" style={{ color: s.color }}>
                                  {s.label} {q.latency[s.k as keyof typeof q.latency]}ms
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                        <span className="text-[11px] font-bold tabular-nums shrink-0 mt-0.5"
                              style={{ color: q.cached ? "#f59e0b" : q.latency.total && q.latency.total > 5000 ? "#f87171" : "#34d399" }}>
                          {q.cached ? "cached" : q.latency.total ? `${(q.latency.total/1000).toFixed(1)}s` : "—"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Ingest latency table */}
              {ingestPerf.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-[12px] font-bold uppercase tracking-widest" style={{ color: "var(--text-dim)" }}>
                      Ingests ({ingestPerf.length})
                    </p>
                    <Sparkline values={ingestPerf.map(i=>i.latency.total_s??0).slice(0,12).reverse()} color="#f59e0b" />
                  </div>
                  <div className="space-y-2 max-h-52 overflow-y-auto custom-scrollbar pr-1">
                    {ingestPerf.slice(0, 15).map((ing, i) => (
                      <div key={i} className="py-2 border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                        <div className="flex items-start justify-between gap-2">
                          <p className="text-[11px] font-medium truncate flex-1" style={{ color: "var(--text-muted)" }}>
                            {ing.title.slice(0, 50)}{ing.title.length > 50 ? "…" : ""}
                          </p>
                          <span className="text-[11px] font-bold tabular-nums shrink-0" style={{ color: "#f59e0b" }}>
                            {ing.latency.total_s?.toFixed(1)}s
                          </span>
                        </div>
                        <div className="flex items-center gap-2 mt-1 flex-wrap">
                          {[
                            { k: "extract_s", label: "fetch"  },
                            { k: "chunk_s",   label: "chunk"  },
                            { k: "embed_s",   label: "embed"  },
                            { k: "load_s",    label: "upsert" },
                          ].filter(s => ing.latency[s.k as keyof typeof ing.latency]).map(s => (
                            <span key={s.k} className="text-[10px] font-mono" style={{ color: "var(--text-dim)" }}>
                              {s.label} {ing.latency[s.k as keyof typeof ing.latency]?.toFixed(1)}s
                            </span>
                          ))}
                          <span className="text-[10px]" style={{ color: "var(--text-dim)" }}>· {ing.chunks} chunks</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* RAGAS results */}
        {loading ? (
          <div className="rounded-2xl p-8 border flex items-center justify-center gap-3"
               style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
            <RefreshCw size={18} className="animate-spin" style={{ color: "var(--text-dim)" }} />
            <span style={{ color: "var(--text-dim)" }}>Loading benchmark results…</span>
          </div>
        ) : !report?.available ? (
          <div className="rounded-2xl p-8 border text-center"
               style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
            <AlertCircle size={36} className="mx-auto mb-4" style={{ color: "var(--text-dim)", opacity: 0.4 }} />
            <p className="text-[15px] font-semibold mb-2" style={{ color: "var(--text)" }}>No benchmark report yet</p>
            <p className="text-[13px] mb-4 max-w-md mx-auto" style={{ color: "var(--text-dim)" }}>
              Run the benchmark script on a collection to generate RAGAS scores.
            </p>
            <code className="px-3 py-1.5 rounded-lg text-[12px]"
                  style={{ background: "var(--bg-hover)", color: "var(--text-muted)", border: "1px solid var(--border)" }}>
              poetry run python tests/run_benchmark.py --evaluate
            </code>
          </div>
        ) : (
          <>
            {/* Report meta */}
            <div className="flex items-center justify-between text-[12px]" style={{ color: "var(--text-dim)" }}>
              <span>
                Last run: <strong style={{ color: "var(--text-muted)" }}>
                  {new Date(report.generated_at as string).toLocaleString()}
                </strong>
              </span>
              <span className="truncate max-w-xs" title={report.collection as string}>
                Collection: <strong style={{ color: "var(--text-muted)" }}>
                  {collections.find(c => c.name === (report.collection as string))?.title
                    ?? (report.collection as string | undefined)?.replace(/-+/g, " ") ?? "–"}
                </strong>
              </span>
            </div>

            {/* Mode comparison cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {modes.map(([key, data]) => {
                if (!data) return null;
                const { label, color, bg } = MODE_LABELS[key];
                const avg = avgScore(data.ragas_scores);
                const isBest = topMode?.[0] === key;
                return (
                  <div key={key} className="rounded-2xl p-5 border relative"
                       style={{ background: "var(--bg-card)", borderColor: isBest ? color : "var(--border)" }}>
                    {isBest && (
                      <span className="absolute top-3 right-3 text-[10px] font-bold px-2 py-0.5 rounded-full"
                            style={{ background: bg, color }}>
                        Best
                      </span>
                    )}
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
                      <span className="text-[13px] font-bold" style={{ color: "var(--text)" }}>{label}</span>
                    </div>

                    {/* Overall avg */}
                    <div className="text-center mb-4 py-3 rounded-xl" style={{ background: bg }}>
                      <p className="text-3xl font-bold" style={{ color }}>
                        {Math.round(avg * 100)}%
                      </p>
                      <p className="text-[11px] mt-0.5" style={{ color: "var(--text-dim)" }}>avg score</p>
                    </div>

                    {/* Latency */}
                    <div className="flex items-center justify-between mb-4 text-[12px]">
                      <span style={{ color: "var(--text-dim)" }}>Latency</span>
                      <span className="font-bold" style={{ color: "var(--text-muted)" }}>
                        {data.avg_latency_s.toFixed(2)}s
                      </span>
                    </div>

                    {/* Per-metric bars */}
                    <div className="space-y-2.5">
                      {Object.entries(METRIC_LABELS).map(([metric, metricLabel]) => {
                        const val = data.ragas_scores[metric];
                        if (val === undefined) return null;
                        return (
                          <div key={metric}>
                            <div className="flex justify-between text-[11px] mb-1">
                              <span style={{ color: "var(--text-dim)" }}>{metricLabel}</span>
                            </div>
                            <ScoreBar value={val} color={color} />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Full comparison table */}
            <div className="rounded-2xl border overflow-hidden"
                 style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
              <div className="px-5 py-4 border-b flex items-center gap-2"
                   style={{ borderColor: "var(--border)" }}>
                <BarChart2 size={15} className="text-indigo-400" />
                <h2 className="text-[14px] font-semibold" style={{ color: "var(--text)" }}>Score Comparison</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[13px]">
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      <th className="text-left px-5 py-3 font-semibold" style={{ color: "var(--text-dim)" }}>Metric</th>
                      {modes.map(([key]) => (
                        <th key={key} className="px-5 py-3 font-semibold text-right"
                            style={{ color: MODE_LABELS[key].color }}>
                          {MODE_LABELS[key].label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(METRIC_LABELS).map(([metric, label], i) => (
                      <tr key={metric}
                          className={i % 2 === 0 ? "" : ""}
                          style={{ borderBottom: "1px solid var(--border)" }}>
                        <td className="px-5 py-3" style={{ color: "var(--text-muted)" }}>{label}</td>
                        {modes.map(([key, data]) => {
                          if (!data) return <td key={key} className="px-5 py-3 text-right">—</td>;
                          const val = data.ragas_scores[metric];
                          const { color } = MODE_LABELS[key];
                          const best = Math.max(...modes.map(([, d]) => d?.ragas_scores[metric] ?? 0));
                          return (
                            <td key={key} className="px-5 py-3 text-right font-mono font-bold"
                                style={{ color: val === best ? color : "var(--text-dim)" }}>
                              {val !== undefined ? `${(val * 100).toFixed(1)}%` : "—"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                    {/* Avg row */}
                    <tr style={{ background: "var(--bg-hover)" }}>
                      <td className="px-5 py-3 font-bold" style={{ color: "var(--text)" }}>Average</td>
                      {modes.map(([key, data]) => {
                        if (!data) return <td key={key} className="px-5 py-3 text-right">—</td>;
                        const avg = avgScore(data.ragas_scores);
                        const { color } = MODE_LABELS[key];
                        const bestAvg = Math.max(...modes.map(([, d]) => d ? avgScore(d.ragas_scores) : 0));
                        return (
                          <td key={key} className="px-5 py-3 text-right font-mono font-bold"
                              style={{ color: avg === bestAvg ? color : "var(--text-muted)" }}>
                            {`${(avg * 100).toFixed(1)}%`}
                          </td>
                        );
                      })}
                    </tr>
                    {/* Latency row */}
                    <tr style={{ borderTop: "1px solid var(--border)" }}>
                      <td className="px-5 py-3 font-medium" style={{ color: "var(--text-muted)" }}>Avg Latency</td>
                      {modes.map(([key, data]) => {
                        if (!data) return <td key={key} className="px-5 py-3 text-right">—</td>;
                        const { color } = MODE_LABELS[key];
                        const bestLatency = Math.min(...modes.map(([, d]) => d?.avg_latency_s ?? Infinity));
                        return (
                          <td key={key} className="px-5 py-3 text-right font-mono"
                              style={{ color: data.avg_latency_s === bestLatency ? color : "var(--text-dim)" }}>
                            {`${data.avg_latency_s.toFixed(2)}s`}
                          </td>
                        );
                      })}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* Grafana link */}
        <div className="rounded-2xl p-5 border flex items-center justify-between"
             style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
          <div>
            <p className="text-[14px] font-semibold mb-0.5" style={{ color: "var(--text)" }}>Live System Metrics</p>
            <p className="text-[12px]" style={{ color: "var(--text-dim)" }}>Latency · throughput · error rates — Grafana dashboard</p>
          </div>
          <a href="http://localhost:3001" target="_blank" rel="noopener noreferrer"
             className="flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] font-bold accent-gradient text-white">
            Open Grafana <ExternalLink size={13} />
          </a>
        </div>
      </div>
    </div>
  );
}
