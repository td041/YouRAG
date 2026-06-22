"use client";
import { useState } from "react";
import { useCollections } from "@/lib/collections-context";
import { fetchQuiz } from "@/lib/api";
import {
  BookOpen, Layers, RotateCcw, ChevronLeft, ChevronRight,
  CheckCircle2, XCircle, Loader2, Sparkles, GraduationCap,
  RefreshCw, X, Play, Menu,
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface QuizQuestion {
  question: string;
  options: string[];
  correct: number;
  explanation: string;
  timestamp: string;
  start_time: number;
}

interface Flashcard {
  front: string;
  back: string;
  timestamp: string;
  start_time: number;
}

type Mode = "quiz" | "flashcard";
type Phase = "setup" | "playing" | "done";

// ─── Flashcard component ───────────────────────────────────────────────────────

function FlashCard({ card, index, total, onNext, onPrev, onSeek }: {
  card: Flashcard;
  index: number;
  total: number;
  onNext: () => void;
  onPrev: () => void;
  onSeek: (startTime: number) => void;
}) {
  const [flipped, setFlipped] = useState(false);

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-2xl mx-auto">
      {/* Progress */}
      <div className="flex items-center gap-3 text-[12px]" style={{ color: "var(--text-dim)" }}>
        <span className="font-bold" style={{ color: "var(--text)" }}>{index + 1}</span>
        <div className="flex-1 h-1 rounded-full overflow-hidden w-48" style={{ background: "var(--bg-hover)" }}>
          <div className="h-full rounded-full bg-indigo-500 transition-all duration-500"
               style={{ width: `${((index + 1) / total) * 100}%` }} />
        </div>
        <span>{total}</span>
      </div>

      {/* Card */}
      <div className="w-full cursor-pointer" style={{ perspective: 1000 }} onClick={() => setFlipped(f => !f)}>
        <div className="relative w-full transition-all duration-500"
             style={{
               transformStyle: "preserve-3d",
               transform: flipped ? "rotateY(180deg)" : "rotateY(0deg)",
               minHeight: 220,
             }}>
          {/* Front — pointerEvents:none when hidden so back face gets clicks */}
          <div className="absolute inset-0 rounded-2xl border flex flex-col items-center justify-center p-8 text-center"
               style={{
                 backfaceVisibility: "hidden",
                 pointerEvents: flipped ? "none" : "auto",
                 background: "var(--bg-card)",
                 borderColor: "var(--border)",
               }}>
            <div className="text-[10px] font-bold uppercase tracking-widest mb-4" style={{ color: "var(--text-dim)" }}>
              Câu hỏi — click để lật
            </div>
            <p className="text-[18px] font-bold leading-snug" style={{ color: "var(--text)" }}>
              {card.front}
            </p>
          </div>

          {/* Back */}
          <div className="absolute inset-0 rounded-2xl border flex flex-col items-center justify-center p-8 text-center"
               style={{
                 backfaceVisibility: "hidden",
                 transform: "rotateY(180deg)",
                 pointerEvents: flipped ? "auto" : "none",
                 background: "linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08))",
                 borderColor: "rgba(99,102,241,0.3)",
               }}>
            <div className="text-[10px] font-bold uppercase tracking-widest mb-4 text-indigo-400">
              Đáp án
            </div>
            <p className="text-[15px] leading-relaxed" style={{ color: "var(--text)" }}>
              {card.back}
            </p>
            <button onClick={e => { e.stopPropagation(); onSeek(card.start_time); }}
                    className="mt-4 flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[11px] font-bold text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/10 transition-all">
              <Play size={11} /> {card.timestamp} — Xem trong video
            </button>
          </div>
        </div>
      </div>

      {/* Nav */}
      <div className="flex items-center gap-4">
        <button onClick={onPrev} disabled={index === 0}
                className="p-2.5 rounded-xl border transition-all disabled:opacity-30"
                style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
          <ChevronLeft size={18} />
        </button>
        <button onClick={() => setFlipped(false)} title="Reset card"
                className="p-2.5 rounded-xl border transition-all hover:bg-white/5"
                style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
          <RotateCcw size={16} />
        </button>
        <button onClick={() => { setFlipped(false); onNext(); }}
                disabled={index === total - 1}
                className="p-2.5 rounded-xl border transition-all disabled:opacity-30"
                style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}

// ─── Quiz component ────────────────────────────────────────────────────────────

function QuizQuestion({ q, index, total, onAnswer, onSeek }: {
  q: QuizQuestion;
  index: number;
  total: number;
  onAnswer: (correct: boolean) => void;
  onSeek: (startTime: number) => void;
}) {
  const [selected, setSelected] = useState<number | null>(null);
  const answered = selected !== null;

  function pick(i: number) {
    if (answered) return;
    setSelected(i);
    onAnswer(i === q.correct);
  }

  return (
    <div className="w-full max-w-2xl mx-auto space-y-5">
      {/* Progress */}
      <div className="flex items-center gap-3 text-[12px]" style={{ color: "var(--text-dim)" }}>
        <span className="font-bold" style={{ color: "var(--text)" }}>{index + 1}</span>
        <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: "var(--bg-hover)" }}>
          <div className="h-full rounded-full bg-indigo-500 transition-all duration-500"
               style={{ width: `${((index + 1) / total) * 100}%` }} />
        </div>
        <span>{total}</span>
      </div>

      {/* Question */}
      <div className="rounded-2xl border p-6" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
        <p className="text-[16px] font-semibold leading-snug" style={{ color: "var(--text)" }}>{q.question}</p>
      </div>

      {/* Options */}
      <div className="space-y-2.5">
        {q.options.map((opt, i) => {
          const isCorrect = i === q.correct;
          const isSelected = i === selected;
          let bg = "var(--bg-card)";
          let border = "var(--border)";
          let color = "var(--text-muted)";
          if (answered) {
            if (isCorrect) { bg = "rgba(52,211,153,0.08)"; border = "rgba(52,211,153,0.4)"; color = "#34d399"; }
            else if (isSelected) { bg = "rgba(248,113,113,0.08)"; border = "rgba(248,113,113,0.4)"; color = "#f87171"; }
          } else if (isSelected) {
            bg = "rgba(99,102,241,0.08)"; border = "rgba(99,102,241,0.4)"; color = "#818cf8";
          }

          return (
            <button key={i} onClick={() => pick(i)}
                    className="w-full flex items-center gap-4 p-4 rounded-2xl border text-left transition-all"
                    style={{ background: bg, borderColor: border, cursor: answered ? "default" : "pointer" }}>
              <span className="w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-bold shrink-0 border"
                    style={{ borderColor: border, color }}>
                {["A", "B", "C", "D"][i]}
              </span>
              <span className="text-[14px]" style={{ color }}>{opt}</span>
              {answered && isCorrect && <CheckCircle2 size={16} className="ml-auto text-emerald-400 shrink-0" />}
              {answered && isSelected && !isCorrect && <XCircle size={16} className="ml-auto text-rose-400 shrink-0" />}
            </button>
          );
        })}
      </div>

      {/* Explanation */}
      {answered && (
        <div className="rounded-2xl p-4 border space-y-2 animate-fade-in"
             style={{ background: "rgba(99,102,241,0.05)", borderColor: "rgba(99,102,241,0.2)" }}>
          <p className="text-[13px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
            <span className="font-bold text-indigo-400">Giải thích: </span>{q.explanation}
          </p>
          <button onClick={() => onSeek(q.start_time)}
                  className="flex items-center gap-1.5 text-[11px] font-bold text-indigo-400 hover:text-indigo-300 transition-colors">
            <Play size={11} /> {q.timestamp} — Xem trong video
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export default function LearnPage() {
  const { collections, selectedCollections, setSelectedCollections, openSidebar } = useCollections();
  const [mode, setMode] = useState<Mode>("quiz");
  const [count, setCount] = useState(5);
  const [phase, setPhase] = useState<Phase>("setup");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [cards, setCards] = useState<Flashcard[]>([]);
  const [qIndex, setQIndex] = useState(0);
  const [cardIndex, setCardIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [waitingNext, setWaitingNext] = useState(false);
  const [seekModal, setSeekModal] = useState<{ startTime: number } | null>(null);

  const targetCollection = selectedCollections[0] ?? collections[0] ?? null;

  async function start() {
    if (!targetCollection) return;
    setLoading(true); setError("");
    try {
      const data = await fetchQuiz(targetCollection.name, count, mode);
      if (mode === "quiz") {
        setQuestions(data.questions ?? []);
        setQIndex(0); setScore(0);
      } else {
        setCards(data.cards ?? []);
        setCardIndex(0);
      }
      setPhase("playing");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function handleAnswer(correct: boolean) {
    if (correct) setScore(s => s + 1);
    setWaitingNext(true);
  }

  function nextQuestion() {
    setWaitingNext(false);
    if (qIndex + 1 >= questions.length) setPhase("done");
    else setQIndex(i => i + 1);
  }

  function handleSeek(startTime: number) {
    if (!targetCollection) return;
    setSeekModal({ startTime });
  }

  function reset() { setPhase("setup"); setQuestions([]); setCards([]); setScore(0); }

  // ── Setup screen ──
  if (phase === "setup") return (
    <div className="flex flex-col h-full theme-bg theme-text">
      <header className="flex items-start gap-3 px-4 sm:px-6 lg:px-8 py-4 sm:py-5 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        <button onClick={openSidebar}
                className="lg:hidden p-1.5 rounded-xl hover:bg-white/5 transition-all shrink-0 mt-1"
                style={{ color: "var(--text-dim)" }}
                aria-label="Open menu">
          <Menu size={18} />
        </button>
        <div>
          <h1 className="text-2xl font-bold font-display" style={{ color: "var(--text)" }}>Learn</h1>
          <p className="text-[13px] mt-0.5" style={{ color: "var(--text-dim)" }}>
            Quiz trắc nghiệm · Flashcard — học từ video đã index
          </p>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6 sm:p-8 flex items-start justify-center">
        <div className="w-full max-w-md space-y-5">

          {/* Mode selector */}
          <div className="rounded-2xl border p-1 flex gap-1" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
            {([["quiz", BookOpen, "Quiz trắc nghiệm"], ["flashcard", Layers, "Flashcard"]] as const).map(([m, Icon, label]) => (
              <button key={m} onClick={() => setMode(m as Mode)}
                      className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-[13px] font-bold transition-all"
                      style={{
                        background: mode === m ? "var(--bg-hover)" : "transparent",
                        color: mode === m ? "var(--text)" : "var(--text-dim)",
                        boxShadow: mode === m ? "0 1px 4px rgba(0,0,0,0.15)" : "none",
                      }}>
                <Icon size={15} /> {label}
              </button>
            ))}
          </div>

          {/* Video selector */}
          <div className="rounded-2xl border p-4 space-y-3" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
            <p className="text-[12px] font-bold uppercase tracking-widest" style={{ color: "var(--text-dim)" }}>Video</p>
            {collections.length === 0 ? (
              <p className="text-[13px]" style={{ color: "var(--text-dim)" }}>Chưa có video nào — vào Library để thêm.</p>
            ) : (
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {collections.map(c => (
                  <button key={c.name} onClick={() => setSelectedCollections([c])}
                          className="w-full flex items-center gap-3 p-2.5 rounded-xl border text-left transition-all"
                          style={{
                            borderColor: targetCollection?.name === c.name ? "rgba(99,102,241,0.4)" : "var(--border)",
                            background: targetCollection?.name === c.name ? "rgba(99,102,241,0.08)" : "transparent",
                          }}>
                    {c.video_id && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={`https://i.ytimg.com/vi/${c.video_id}/default.jpg`} alt=""
                           className="w-9 h-9 rounded-lg object-cover shrink-0" />
                    )}
                    <span className="text-[12px] font-medium truncate" style={{ color: "var(--text-muted)" }}>
                      {c.title}
                    </span>
                    {targetCollection?.name === c.name && (
                      <CheckCircle2 size={14} className="ml-auto shrink-0 text-indigo-400" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Count selector */}
          <div className="rounded-2xl border p-4 space-y-3" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
            <p className="text-[12px] font-bold uppercase tracking-widest" style={{ color: "var(--text-dim)" }}>
              Số {mode === "quiz" ? "câu hỏi" : "flashcard"}
            </p>
            <div className="flex gap-2">
              {[3, 5, 7, 10].map(n => (
                <button key={n} onClick={() => setCount(n)}
                        className="flex-1 py-2 rounded-xl text-[13px] font-bold border transition-all"
                        style={{
                          borderColor: count === n ? "rgba(99,102,241,0.5)" : "var(--border)",
                          background: count === n ? "rgba(99,102,241,0.1)" : "transparent",
                          color: count === n ? "#818cf8" : "var(--text-dim)",
                        }}>
                  {n}
                </button>
              ))}
            </div>
          </div>

          {error && (
            <p className="text-[13px] text-rose-400 px-1">{error}</p>
          )}

          <button onClick={start} disabled={loading || !targetCollection}
                  className="w-full h-12 accent-gradient text-white font-bold rounded-2xl disabled:opacity-30 flex items-center justify-center gap-2 transition-all active:scale-[0.98]">
            {loading ? <><Loader2 size={18} className="animate-spin" /> Đang tạo…</> : <><Sparkles size={18} /> Bắt đầu</>}
          </button>
        </div>
      </div>
    </div>
  );

  // ── Done screen ──
  if (phase === "done") {
    if (mode === "flashcard") {
      return (
        <div className="flex flex-col h-full theme-bg theme-text">
          <header className="flex items-center gap-3 px-4 sm:px-6 lg:px-8 py-4 sm:py-5 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
            <button onClick={openSidebar} className="lg:hidden p-1.5 rounded-xl hover:bg-white/5 transition-all shrink-0" style={{ color: "var(--text-dim)" }} aria-label="Open menu"><Menu size={18} /></button>
            <h1 className="text-2xl font-bold font-display" style={{ color: "var(--text)" }}>Hoàn thành</h1>
          </header>
          <div className="flex-1 flex flex-col items-center justify-center gap-6 p-8">
            <div className="w-32 h-32 rounded-full border-4 border-indigo-500/40 flex items-center justify-center"
                 style={{ background: "rgba(99,102,241,0.08)" }}>
              <div className="text-center">
                <p className="text-4xl">🎉</p>
                <p className="text-[11px] mt-1" style={{ color: "var(--text-dim)" }}>{cards.length} cards</p>
              </div>
            </div>
            <p className="text-[18px] font-bold" style={{ color: "var(--text)" }}>
              Đã ôn xong {cards.length} flashcard!
            </p>
            <div className="flex gap-3">
              <button onClick={start}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-bold border transition-all hover:bg-white/5"
                      style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
                <RefreshCw size={15} /> Ôn lại
              </button>
              <button onClick={reset}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-bold accent-gradient text-white">
                <GraduationCap size={15} /> Thiết lập mới
              </button>
            </div>
          </div>
        </div>
      );
    }

    const pct = Math.round((score / questions.length) * 100);
    const color = pct >= 80 ? "#34d399" : pct >= 60 ? "#818cf8" : "#f87171";
    return (
      <div className="flex flex-col h-full theme-bg theme-text">
        <header className="flex items-center gap-3 px-4 sm:px-6 lg:px-8 py-4 sm:py-5 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
          <button onClick={openSidebar} className="lg:hidden p-1.5 rounded-xl hover:bg-white/5 transition-all shrink-0" style={{ color: "var(--text-dim)" }} aria-label="Open menu"><Menu size={18} /></button>
          <h1 className="text-2xl font-bold font-display" style={{ color: "var(--text)" }}>Kết quả</h1>
        </header>
        <div className="flex-1 flex flex-col items-center justify-center gap-6 p-8">
          <div className="w-32 h-32 rounded-full border-4 flex items-center justify-center"
               style={{ borderColor: color }}>
            <div className="text-center">
              <p className="text-3xl font-black" style={{ color }}>{pct}%</p>
              <p className="text-[11px]" style={{ color: "var(--text-dim)" }}>{score}/{questions.length}</p>
            </div>
          </div>
          <p className="text-[18px] font-bold" style={{ color: "var(--text)" }}>
            {pct >= 80 ? "Xuất sắc! 🎉" : pct >= 60 ? "Khá tốt 👍" : "Cần ôn thêm 📚"}
          </p>
          <div className="flex gap-3">
            <button onClick={start}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-bold border transition-all hover:bg-white/5"
                    style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}>
              <RefreshCw size={15} /> Làm lại
            </button>
            <button onClick={reset}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-bold accent-gradient text-white">
              <GraduationCap size={15} /> Thiết lập mới
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Playing screen ──
  return (
    <div className="flex flex-col h-full theme-bg theme-text relative">
      <header className="flex items-center justify-between px-4 sm:px-6 lg:px-8 py-4 border-b shrink-0"
              style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-3">
          <button onClick={openSidebar} className="lg:hidden p-1.5 rounded-xl hover:bg-white/5 transition-all shrink-0" style={{ color: "var(--text-dim)" }} aria-label="Open menu"><Menu size={18} /></button>
          {mode === "quiz" ? <BookOpen size={16} className="text-indigo-400" /> : <Layers size={16} className="text-indigo-400" />}
          <h1 className="text-[15px] font-bold" style={{ color: "var(--text)" }}>
            {mode === "quiz" ? "Quiz" : "Flashcard"}
          </h1>
          {targetCollection && (
            <span className="text-[11px] px-2 py-0.5 rounded-full" style={{ background: "var(--bg-hover)", color: "var(--text-dim)" }}>
              {targetCollection.title.slice(0, 30)}{targetCollection.title.length > 30 ? "…" : ""}
            </span>
          )}
        </div>
        <button onClick={reset} className="text-[12px] font-bold hover:text-rose-400 transition-colors"
                style={{ color: "var(--text-dim)" }}>
          Thoát
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-6 sm:p-8 flex flex-col items-center">
        {mode === "flashcard" && cards.length > 0 && (
          <FlashCard
            card={cards[cardIndex]}
            index={cardIndex}
            total={cards.length}
            onNext={() => { if (cardIndex + 1 < cards.length) setCardIndex(i => i + 1); else setPhase("done"); }}
            onPrev={() => setCardIndex(i => Math.max(0, i - 1))}
            onSeek={handleSeek}
          />
        )}

        {mode === "quiz" && questions.length > 0 && (
          <div className="w-full max-w-2xl">
            <QuizQuestion
              key={qIndex}
              q={questions[qIndex]}
              index={qIndex}
              total={questions.length}
              onAnswer={handleAnswer}
              onSeek={handleSeek}
            />
            {waitingNext && (
              <div className="mt-6 flex justify-center">
                <button onClick={nextQuestion}
                        className="px-6 py-2.5 rounded-xl font-bold text-[13px] accent-gradient text-white transition-all active:scale-95">
                  {qIndex + 1 >= questions.length ? "Xem kết quả" : "Câu tiếp theo →"}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Video seek modal ── */}
      {seekModal && targetCollection?.video_id && (
        <div className="absolute inset-0 z-50 flex items-center justify-center p-4 sm:p-8"
             style={{ background: "rgba(0,0,0,0.75)", backdropFilter: "blur(6px)" }}
             onClick={() => setSeekModal(null)}>
          <div className="w-full max-w-2xl rounded-2xl overflow-hidden shadow-2xl border"
               style={{ borderColor: "rgba(255,255,255,0.08)" }}
               onClick={e => e.stopPropagation()}>
            {/* Modal header */}
            <div className="flex items-center justify-between px-4 py-3"
                 style={{ background: "#0d1117", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="flex items-center gap-2">
                <Play size={13} className="text-indigo-400" />
                <span className="text-[12px] font-bold" style={{ color: "#94a3b8" }}>
                  {targetCollection.title.slice(0, 50)}{targetCollection.title.length > 50 ? "…" : ""}
                </span>
              </div>
              <button onClick={() => setSeekModal(null)}
                      className="p-1 rounded-lg hover:bg-white/10 transition-colors"
                      style={{ color: "#475569" }}>
                <X size={15} />
              </button>
            </div>
            {/* YouTube iframe */}
            <div className="relative w-full" style={{ aspectRatio: "16/9", background: "#000" }}>
              <iframe
                key={`modal-${seekModal.startTime}`}
                src={`https://www.youtube.com/embed/${targetCollection.video_id}?start=${Math.floor(seekModal.startTime)}&autoplay=1&rel=0&modestbranding=1`}
                title="Video preview"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="absolute inset-0 w-full h-full"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
