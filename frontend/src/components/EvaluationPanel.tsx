import { useState } from "react";
import { BrainCircuit } from "lucide-react";
import { API } from "../lib/api";
import { formatScore } from "../lib/format";
import type { EvalSummary } from "../types";

function MetricRow({ label, value }: { label: string; value: number | null | undefined }) {
  return (
    <div className="eval-row">
      <span>{label}</span>
      <strong>{typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—"}</strong>
    </div>
  );
}

export function EvaluationPanel() {
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), 15000);
      const res = await fetch(`${API}/evaluation/summary`, { signal: controller.signal });
      window.clearTimeout(timer);
      if (!res.ok) throw new Error(`Evaluation endpoint returned ${res.status}.`);
      const data: EvalSummary = await res.json();
      setSummary(data);
      setLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load evaluation results.");
    } finally {
      setLoading(false);
    }
  }

  const run = summary?.latest_run ?? null;
  const comp = summary?.latest_comparison ?? null;
  const hasAnyData = Boolean(summary?.benchmark || run || comp);

  return (
    <section className="panel citations-panel">
      <div className="panel-heading">
        <div>
          <h2>Evaluation</h2>
          <p>Phase 6 metrics · Phase 7 baseline vs. improved</p>
        </div>
        <BrainCircuit size={18} />
      </div>
      {!loaded && !loading && !error && (
        <>
          <div className="empty-state">Stored results from the last evaluation run. Not recomputed live.</div>
          <button className="primary-button" onClick={load}>Load evaluation results</button>
        </>
      )}
      {loading && <div className="inline-status" role="status">Loading stored evaluation results…</div>}
      {error && (
        <>
          <div className="inline-status error" role="alert">{error}</div>
          <button className="primary-button" onClick={load}>Retry</button>
        </>
      )}
      {loaded && !error && summary && !hasAnyData && (
        <div className="empty-state">
          No evaluation results yet — run the Phase 6/7 harness first.
          {(summary.notes ?? []).map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      )}
      {loaded && !error && summary && hasAnyData && (
        <div className="trace-list">
          {summary.benchmark && (
            <div className="trace-row">
              <span>benchmark</span>
              <p>
                {summary.benchmark.name ?? "benchmark"} v{summary.benchmark.version ?? "?"} ·{" "}
                {summary.benchmark.question_count ?? "?"} questions
              </p>
            </div>
          )}
          {run ? (
            <div className="trace-row">
              <span>improved · latest run</span>
              <p>
                {run.file ?? "run"} · {run.question_count ?? "?"} questions
                {typeof run.api_available === "boolean" ? (run.api_available ? " · live API" : " · offline") : ""}
              </p>
              <div className="eval-table">
                <MetricRow label="Recall@5" value={run.retrieval?.recall_at_5} />
                <MetricRow label="Recall@10" value={run.retrieval?.recall_at_10} />
                <MetricRow label="MRR" value={run.retrieval?.mrr} />
                <MetricRow label="Judge correctness" value={run.judge?.correctness_mean} />
                <MetricRow label="Judge groundedness" value={run.judge?.groundedness_mean} />
              </div>
              {(run.judge?.available === 0 || run.unanswerable) && (
                <p>
                  Judge scored {run.judge?.available ?? 0}/{run.judge?.total ?? "?"} · unanswerable handled{" "}
                  {run.unanswerable?.correctly_handled ?? "?"}/{run.unanswerable?.count ?? "?"}
                </p>
              )}
            </div>
          ) : (
            <div className="empty-state">No evaluation results yet — no Phase 6 run file found.</div>
          )}
          {comp ? (
            <div className="trace-row">
              <span>baseline vs. improved</span>
              <p>{comp.file ?? "comparison"} · {comp.question_count ?? "?"} questions per pipeline</p>
              <div className="eval-table">
                {(["recall_at_5", "recall_at_10", "mrr"] as const).map((key) => {
                  const b = comp.baseline?.retrieval?.[key];
                  const i = comp.improved?.retrieval?.[key];
                  const d = comp.differences_improved_minus_baseline?.[key];
                  return (
                    <div className="eval-row" key={key}>
                      <span>{key.replaceAll("_", "@")}</span>
                      <strong>
                        {formatScore(b)} → {formatScore(i)}
                        {typeof d === "number" ? ` (${d >= 0 ? "+" : ""}${d.toFixed(4)})` : ""}
                      </strong>
                    </div>
                  );
                })}
              </div>
              <p>
                Unanswerable handled: baseline {comp.baseline?.unanswerable?.correctly_handled ?? "?"} vs. improved{" "}
                {comp.improved?.unanswerable?.correctly_handled ?? "?"}
                {comp.corpus_mismatch ? " · corpus changed mid-run" : ""}
              </p>
            </div>
          ) : (
            <div className="empty-state">No comparison yet — no Phase 7 comparison file found.</div>
          )}
        </div>
      )}
    </section>
  );
}
