import { useState, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Zap, Database, GitBranch, Search, TrendingDown, Award, CheckCircle, XCircle, AlertCircle, Loader, Hash, FileText } from "lucide-react";

// Use VITE_API_URL if set, otherwise detect automatically
let API_BASE;
if (import.meta.env.VITE_API_URL) {
  API_BASE = import.meta.env.VITE_API_URL;
} else if (typeof window !== "undefined" && window.location.hostname === "localhost") {
  API_BASE = "http://localhost:8000";
} else if (typeof window !== "undefined") {
  API_BASE = window.location.origin;
} else {
  API_BASE = "http://localhost:8000";
}

// Round 3 pipeline ids match the backend: llm_only / rag / graphrag
const PIPELINES = [
  { id: "llm_only", label: "LLM-Only",         icon: Zap,      color: "#ef4444", dimColor: "rgba(239,68,68,0.12)", desc: "No retrieval · parametric only" },
  { id: "rag",      label: "Traditional RAG",  icon: Database, color: "#f97316", dimColor: "rgba(249,115,22,0.12)", desc: "TigerGraph vector search · top-k=8" },
  { id: "graphrag", label: "Optimized GraphRAG", icon: GitBranch, color: "#22c55e", dimColor: "rgba(34,197,94,0.12)", desc: "Vector + 2-hop graph · context optimizer" },
];

// SP100 SEC-filings eval questions (Round 3 dataset)
const SAMPLE_QUESTIONS = [
  { q: "Apple's total net sales for fiscal year 2025?", ref: "" },
  { q: "The company that agreed to acquire Apogee Therapeutics — who is its independent auditor?", ref: "" },
  { q: "Which had the higher fiscal-year net income: JPMorgan Chase or ExxonMobil?", ref: "" },
  { q: "The company that appointed John Ternus as CEO — what were its total net sales in FY2025?", ref: "" },
  { q: "Besides Netflix, which company authorized a $25B share-repurchase program in 2026?", ref: "" },
  { q: "Who is Amazon's Chief Executive Officer?", ref: "" },
];

const fmt = (n) => (n === null || n === undefined ? "—" : Number(n).toLocaleString());

// ─── Pipeline Answer Card ────────────────────────────────────────────────────
function PipelineCard({ pipeline, data, loading, best }) {
  const { label, color, dimColor, desc, icon: Icon } = pipeline;
  const graded = data && data.grade !== null && data.grade !== undefined;
  return (
    <div style={{
      background: "var(--card)", border: `1px solid ${best ? color : color + "33"}`,
      borderRadius: 16, padding: "1.25rem", display: "flex", flexDirection: "column", gap: ".9rem",
      boxShadow: best ? `0 0 24px ${color}30` : "none", position: "relative",
    }}>
      {best && (
        <span style={{ position: "absolute", top: -10, right: 14, background: color, color: "#04110a",
          fontSize: ".62rem", fontWeight: 800, padding: ".15rem .55rem", borderRadius: 20, letterSpacing: ".04em" }}>
          BEST
        </span>
      )}
      <div style={{ display: "flex", alignItems: "center", gap: ".6rem" }}>
        <div style={{ background: dimColor, borderRadius: 8, padding: ".45rem", display: "flex" }}>
          <Icon size={18} color={color} />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: ".95rem", color }}>{label}</div>
          <div style={{ fontSize: ".72rem", color: "var(--text-muted)" }}>{desc}</div>
        </div>
      </div>

      {data && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: ".6rem" }}>
          {[
            { label: "Inference tokens", val: fmt(data.total_inference) },
            { label: "Grade", val: graded ? `${data.grade}/3` : "—" },
            { label: "Latency", val: data.latency_ms !== undefined ? `${(data.latency_ms / 1000).toFixed(1)}s` : "—" },
          ].map(({ label: ml, val }) => (
            <div key={ml} style={{
              background: "var(--surface)", borderRadius: 9, padding: ".55rem .65rem", border: "1px solid var(--border-light)",
            }}>
              <div style={{ fontSize: ".62rem", color: "var(--text-muted)", marginBottom: ".2rem" }}>{ml}</div>
              <div style={{ fontWeight: 700, fontSize: ".9rem", color: "var(--text)" }}>{val}</div>
            </div>
          ))}
        </div>
      )}

      {/* Answer */}
      <div style={{
        background: "var(--surface)", borderRadius: 10, padding: "1rem",
        border: "1px solid var(--border-light)", minHeight: 120, flex: 1,
        fontSize: ".85rem", lineHeight: 1.65, color: "var(--text-secondary)", whiteSpace: "pre-wrap",
      }}>
        {loading ? (
          <div style={{ display: "flex", alignItems: "center", gap: ".6rem", color: "var(--text-muted)" }}>
            <Loader size={14} className="spin" /><span>Running pipeline…</span>
          </div>
        ) : data ? (
          <span style={{ color: "var(--text)" }}>{data.answer}</span>
        ) : (
          <span style={{ color: "var(--text-muted)" }}>Answer will appear here after you submit a query.</span>
        )}
      </div>

      {/* Footer chips */}
      {data && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: ".4rem", fontSize: ".7rem" }}>
          {graded && (
            <span style={{
              display: "inline-flex", alignItems: "center", gap: ".3rem", borderRadius: 6, padding: ".2rem .5rem",
              background: data.pass ? "rgba(34,197,94,.12)" : "rgba(239,68,68,.12)", color: data.pass ? "#22c55e" : "#f87171",
            }}>
              {data.pass ? <CheckCircle size={11} /> : <XCircle size={11} />}{data.pass ? "Strict pass" : "Not exact"}
            </span>
          )}
          {graded && data.numeric_match !== null && (
            <span style={{
              display: "inline-flex", alignItems: "center", gap: ".3rem", borderRadius: 6, padding: ".2rem .5rem",
              background: data.numeric_match ? "rgba(34,197,94,.12)" : "rgba(148,163,184,.12)",
              color: data.numeric_match ? "#22c55e" : "var(--text-muted)",
            }}>
              <Hash size={11} />{data.numeric_match ? "figure ✓" : "figure ✗"}
            </span>
          )}
          <span style={{ display: "inline-flex", alignItems: "center", gap: ".3rem", borderRadius: 6, padding: ".2rem .5rem",
            background: "var(--surface)", color: "var(--text-muted)" }}>
            <FileText size={11} />{(data.citations?.length || 0)} citations
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: ".3rem", borderRadius: 6, padding: ".2rem .5rem",
            background: "var(--surface)", color: "var(--text-muted)" }}>
            ctx {fmt(data.context_tokens)} tok
          </span>
          {data.evidence_quality !== undefined && (
            <span style={{ display: "inline-flex", alignItems: "center", gap: ".3rem", borderRadius: 6, padding: ".2rem .5rem",
              background: "var(--surface)", color: "var(--text-muted)" }}>
              evidence {data.evidence_quality}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Bar chart (generic) ─────────────────────────────────────────────────────
function CompareBar({ title, unit, keyName, results, decimals = 0 }) {
  if (!results) return null;
  const data = PIPELINES.map((p) => ({
    name: p.label.replace("Optimized ", "").replace("Traditional ", ""),
    val: results[p.id]?.[keyName] ?? 0, fill: p.color,
  }));
  return (
    <div style={{ background: "var(--card)", borderRadius: 14, padding: "1.25rem", border: "1px solid var(--border)" }}>
      <div style={{ fontWeight: 700, marginBottom: "1rem", fontSize: ".9rem" }}>{title}</div>
      <ResponsiveContainer width="100%" height={190}>
        <BarChart data={data} barCategoryGap="35%">
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(v) => [`${Number(v).toFixed(decimals)} ${unit}`, ""]}
            contentStyle={{ background: "#111827", border: "1px solid #1e3a5f", borderRadius: 8, fontSize: 13 }}
            cursor={{ fill: "rgba(255,255,255,.04)" }}
          />
          <Bar dataKey="val" radius={[6, 6, 0, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Badges ──────────────────────────────────────────────────────────────────
function ReductionBadge({ label, pct, sub }) {
  const good = pct > 0;
  const color = good ? "#22c55e" : "#ef4444";
  return (
    <div style={{
      background: good ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
      border: `1px solid ${color}44`, borderRadius: 10, padding: ".75rem 1rem",
      display: "flex", flexDirection: "column", gap: ".25rem",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: ".4rem", fontSize: ".72rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
        <TrendingDown size={12} color={color} />{label}
      </div>
      <div style={{ fontWeight: 800, fontSize: "1.5rem", color }}>{good ? "↓" : "↑"}{Math.abs(pct).toFixed(1)}%</div>
      <div style={{ fontSize: ".7rem", color: "var(--text-muted)" }}>{sub}</div>
    </div>
  );
}

function ValidityBadge({ ok, graded }) {
  const color = ok ? "#22c55e" : graded ? "#ef4444" : "#64748b";
  return (
    <div style={{
      background: ok ? "rgba(34,197,94,0.1)" : "rgba(148,163,184,0.08)",
      border: `1px solid ${color}44`, borderRadius: 10, padding: ".75rem 1rem",
      display: "flex", flexDirection: "column", gap: ".25rem",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: ".4rem", fontSize: ".72rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
        <Award size={12} color={color} />Validity ordering
      </div>
      <div style={{ fontWeight: 800, fontSize: "1.15rem", color }}>
        {!graded ? "add reference" : ok ? "LLM ≤ RAG ≤ GraphRAG ✓" : "ordering broken"}
      </div>
      <div style={{ fontSize: ".7rem", color: "var(--text-muted)" }}>graded /3 by evidence-blind judge</div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [question, setQuestion] = useState("");
  const [groundTruth, setGroundTruth] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [sessionStats, setSessionStats] = useState(null);

  const fetchSessionStats = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/stats/session`);
      if (r.ok) setSessionStats(await r.json());
    } catch {}
  }, []);

  const handleQuery = useCallback(async () => {
    if (!question.trim() || loading) return;
    setLoading(true); setError(null); setResults(null);
    try {
      const res = await fetch(`${API_BASE}/query/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), ground_truth: groundTruth, run_judge: true }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResults(data);
      fetchSessionStats();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [question, groundTruth, loading, fetchSessionStats]);

  const graded = results?.graded;

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
      {/* Header */}
      <header style={{
        background: "var(--surface)", borderBottom: "1px solid var(--border)",
        padding: "1rem 2rem", display: "flex", alignItems: "center", justifyContent: "space-between",
        position: "sticky", top: 0, zIndex: 100, backdropFilter: "blur(10px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: ".75rem" }}>
          <span style={{ fontSize: "1.5rem" }}>🐯</span>
          <div>
            <div style={{ fontWeight: 800, fontSize: "1.1rem", color: "var(--orange)" }}>TokenNinja — 3-Pipeline Comparison</div>
            <div style={{ fontSize: ".72rem", color: "var(--text-muted)" }}>TigerGraph GraphRAG Hackathon · Round 3 · SP100 SEC filings</div>
          </div>
        </div>
        {sessionStats && (
          <div style={{ display: "flex", gap: "1.5rem", fontSize: ".8rem" }}>
            <div style={{ textAlign: "right" }}>
              <div style={{ color: "var(--text-muted)", fontSize: ".68rem" }}>QUERIES RUN</div>
              <div style={{ fontWeight: 700, color: "var(--blue)" }}>{sessionStats.total_queries}</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ color: "var(--text-muted)", fontSize: ".68rem" }}>TOKENS SAVED</div>
              <div style={{ fontWeight: 700, color: "var(--green)" }}>{fmt(sessionStats.total_tokens_saved)}</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ color: "var(--text-muted)", fontSize: ".68rem" }}>AVG REDUCTION</div>
              <div style={{ fontWeight: 700, color: "var(--green)" }}>{sessionStats.avg_token_reduction_pct}%</div>
            </div>
          </div>
        )}
      </header>

      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "2rem" }}>
        {/* Query Input */}
        <div style={{
          background: "var(--card)", borderRadius: 16, padding: "1.5rem",
          border: "1px solid var(--border)", marginBottom: "1.5rem", boxShadow: "0 0 40px rgba(249,115,22,0.06)",
        }}>
          <div style={{ fontWeight: 700, marginBottom: "1rem", color: "var(--text)", display: "flex", alignItems: "center", gap: ".5rem" }}>
            <Search size={16} color="var(--orange)" /> Ask across all three pipelines
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: ".5rem", marginBottom: "1rem" }}>
            {SAMPLE_QUESTIONS.map((s, i) => (
              <button key={i} onClick={() => { setQuestion(s.q); setGroundTruth(s.ref || ""); }} style={{
                background: "var(--surface)", border: "1px solid var(--border-light)", borderRadius: 20,
                padding: ".3rem .8rem", fontSize: ".72rem", color: "var(--text-secondary)", cursor: "pointer", transition: "all .15s",
              }}
                onMouseEnter={(e) => (e.target.style.borderColor = "var(--orange)")}
                onMouseLeave={(e) => (e.target.style.borderColor = "var(--border-light)")}
              >{s.q.length > 58 ? s.q.substring(0, 58) + "…" : s.q}</button>
            ))}
          </div>

          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What were Apple's total net sales in FY2025?"
            rows={2}
            style={{
              width: "100%", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10,
              padding: ".9rem 1rem", color: "var(--text)", fontSize: ".9rem", resize: "vertical", lineHeight: 1.6,
              outline: "none", fontFamily: "inherit", marginBottom: ".75rem", transition: "border-color .15s",
            }}
            onFocus={(e) => (e.target.style.borderColor = "var(--orange)")}
            onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
            onKeyDown={(e) => e.key === "Enter" && e.ctrlKey && handleQuery()}
          />

          <input
            value={groundTruth}
            onChange={(e) => setGroundTruth(e.target.value)}
            placeholder="Reference answer (optional — enables graded /3 judge + numeric-match + validity ordering)"
            style={{
              width: "100%", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10,
              padding: ".65rem 1rem", color: "var(--text)", fontSize: ".82rem", outline: "none", fontFamily: "inherit",
              marginBottom: "1rem", transition: "border-color .15s",
            }}
            onFocus={(e) => (e.target.style.borderColor = "var(--blue)")}
            onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
          />

          <button
            onClick={handleQuery}
            disabled={!question.trim() || loading}
            style={{
              background: question.trim() && !loading ? "var(--orange)" : "var(--surface)",
              color: question.trim() && !loading ? "white" : "var(--text-muted)",
              border: "none", borderRadius: 10, padding: ".75rem 2rem", fontWeight: 700, fontSize: ".9rem",
              cursor: question.trim() && !loading ? "pointer" : "default", display: "flex", alignItems: "center", gap: ".5rem", transition: "all .2s",
            }}
          >
            {loading ? <><Loader size={16} className="spin" /> Running all 3 pipelines…</> : <><Zap size={16} /> Run All Pipelines</>}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: "rgba(239,68,68,0.1)", border: "1px solid #ef444444", borderRadius: 10, padding: "1rem",
            marginBottom: "1.5rem", color: "#f87171", display: "flex", alignItems: "center", gap: ".5rem",
          }}>
            <AlertCircle size={16} /> {error} — is the backend running? (<code>uvicorn backend.api.server:app --reload --port 8000</code>)
          </div>
        )}

        {/* Results */}
        {results && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
              <ReductionBadge label="Token reduction" pct={results.token_reduction_pct} sub="GraphRAG vs Traditional RAG" />
              <ReductionBadge label="Context reduction" pct={results.context_reduction_pct} sub="evidence sent to the model" />
              <ValidityBadge ok={results.validity_ordering_ok} graded={graded} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.5rem" }}>
              <CompareBar title="Total inference tokens" unit="tok" keyName="total_inference" results={results} />
              {graded
                ? <CompareBar title="Answer grade (LLM-as-judge, /3)" unit="/3" keyName="grade" results={results} decimals={0} />
                : <CompareBar title="Context tokens (evidence fed to model)" unit="tok" keyName="context_tokens" results={results} />}
            </div>
          </>
        )}

        {/* 3 Pipeline Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginBottom: "2rem" }}>
          {PIPELINES.map((p) => (
            <PipelineCard key={p.id} pipeline={p} data={results?.[p.id]} loading={loading} best={p.id === "graphrag" && !!results} />
          ))}
        </div>

        {/* Metrics Table */}
        {results && (
          <div style={{ background: "var(--card)", borderRadius: 14, padding: "1.25rem", border: "1px solid var(--border)", marginBottom: "2rem" }}>
            <div style={{ fontWeight: 700, marginBottom: "1rem", fontSize: ".9rem" }}>📊 Full metrics (this query)</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: ".83rem" }}>
                <thead>
                  <tr>
                    {["Metric", "LLM-Only", "Traditional RAG", "GraphRAG", "Graph vs RAG"].map((h) => (
                      <th key={h} style={{ textAlign: "left", padding: ".6rem .75rem", color: "var(--text-muted)", borderBottom: "1px solid var(--border)", fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: "Context tokens",   key: "context_tokens",  better: "lower" },
                    { label: "Output tokens",    key: "output_tokens",   better: "lower" },
                    { label: "Total inference",  key: "total_inference", better: "lower" },
                    { label: "Latency (ms)",     key: "latency_ms",      better: "lower", fx: (v) => v?.toFixed(0) },
                    { label: "Grade (/3)",       key: "grade",           better: "higher" },
                    { label: "Evidence quality", key: "evidence_quality", better: "higher" },
                  ].map((row) => {
                    const llm = results.llm_only[row.key];
                    const basic = results.rag[row.key];
                    const graph = results.graphrag[row.key];
                    const bn = parseFloat(basic), gn = parseFloat(graph);
                    const graphBetter = row.better === "lower" ? gn < bn : gn > bn;
                    const pctDiff = bn > 0 ? (((gn - bn) / bn) * 100).toFixed(1) : null;
                    const show = (v) => (v === null || v === undefined ? "—" : row.fx ? row.fx(v) : fmt(v));
                    return (
                      <tr key={row.label} style={{ borderBottom: "1px solid var(--border-light)" }}>
                        <td style={{ padding: ".55rem .75rem", color: "var(--text-secondary)", fontWeight: 500 }}>{row.label}</td>
                        <td style={{ padding: ".55rem .75rem", color: "#ef4444" }}>{show(llm)}</td>
                        <td style={{ padding: ".55rem .75rem", color: "#f97316" }}>{show(basic)}</td>
                        <td style={{ padding: ".55rem .75rem", color: "#22c55e", fontWeight: 700 }}>{show(graph)}</td>
                        <td style={{ padding: ".55rem .75rem" }}>
                          {pctDiff !== null && !isNaN(gn) && !isNaN(bn) && (
                            <span style={{
                              background: graphBetter ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
                              color: graphBetter ? "#22c55e" : "#ef4444", borderRadius: 6, padding: ".15rem .45rem", fontSize: ".75rem", fontWeight: 700,
                            }}>
                              {pctDiff > 0 ? "+" : ""}{pctDiff}%
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {results.reference && (
              <div style={{ marginTop: ".9rem", fontSize: ".78rem", color: "var(--text-muted)" }}>
                <strong style={{ color: "var(--text-secondary)" }}>Reference:</strong> {results.reference}
              </div>
            )}
          </div>
        )}

        <footer style={{ textAlign: "center", color: "var(--text-muted)", fontSize: ".75rem", paddingTop: "1rem", borderTop: "1px solid var(--border)" }}>
          🐯 TokenNinja · TigerGraph Savanna = graph + vector DB · deterministic context optimizer (0 extra tokens) &nbsp;·&nbsp;
          <a href="https://github.com/Dhruvpandey1476/TokenNinja-" style={{ color: "var(--orange)", textDecoration: "none" }}>github.com/Dhruvpandey1476/TokenNinja-</a>
        </footer>
      </main>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
        button:hover:not(:disabled) { opacity: .88; transform: translateY(-1px); }
      `}</style>
    </div>
  );
}
