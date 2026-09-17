import { Activity } from "lucide-react";
import { formatLatency, formatScore } from "../lib/format";
import type { Latencies, VerificationState } from "../types";

export function PipelinePanel({
  strategy,
  routerReason,
  routerFellBack,
  subqueries,
  candidateCount,
  latencies,
  reranked,
  rerankFallback,
  rerankReason,
  verification,
  insufficient,
  hasAsked,
}: {
  strategy: string | null;
  routerReason: string;
  routerFellBack: boolean;
  subqueries: string[];
  candidateCount: number | null;
  latencies: Latencies | null;
  reranked: boolean | null;
  rerankFallback: boolean | null;
  rerankReason: string;
  verification: VerificationState | null;
  insufficient: boolean | null;
  hasAsked: boolean;
}) {
  if (!hasAsked) {
    return (
      <section className="panel trace-panel">
        <div className="panel-heading">
          <div>
            <h2>Pipeline</h2>
            <p>Strategy, scores, latencies, verification</p>
          </div>
          <Activity size={18} />
        </div>
        <div className="empty-state">Ask a question to see how the pipeline behaved.</div>
      </section>
    );
  }
  const stages: Array<[string, number | undefined]> = [
    ["Routing", latencies?.routing_latency],
    ["Retrieval", latencies?.retrieval_latency],
    ["Reranking", latencies?.reranking_latency],
    ["Generation", latencies?.generation_latency],
    ["Verification", latencies?.verification_latency],
  ];
  return (
    <section className="panel trace-panel">
      <div className="panel-heading">
        <div>
          <h2>Pipeline</h2>
          <p>How the RAG pipeline behaved for this query</p>
        </div>
        <Activity size={18} />
      </div>
      <div className="trace-list">
        <div className="trace-row">
          <span>{strategy ?? "unknown strategy"}</span>
          <p>
            {routerReason || "No router reason returned."}
            {routerFellBack ? " (router fallback used — STANDARD)" : ""}
            {typeof candidateCount === "number" ? ` · ${candidateCount} candidates` : ""}
          </p>
          {strategy === "MULTI_HOP" && (
            <div className="subquery-list">
              {subqueries.length === 0 && <p>No subqueries returned.</p>}
              {subqueries.map((sub) => (
                <p key={sub}>↳ {sub}</p>
              ))}
            </div>
          )}
        </div>
        <div className="trace-row">
          <span>{rerankFallback ? "rerank fallback" : reranked ? "reranked" : "similarity"}</span>
          <p>
            {rerankFallback
              ? `Reranking fallback used — ordered by similarity. ${rerankReason || ""}`.trim()
              : reranked
                ? "Gemini reranking selected the final evidence."
                : "Reranking state unknown for this answer."}
          </p>
        </div>
        <div className="trace-row">
          <span>latencies</span>
          <div className="latency-grid">
            {stages.map(([label, value]) => (
              <div className="latency-cell" key={label}>
                <strong>{formatLatency(value ?? null)}</strong>
                <span>{label}</span>
              </div>
            ))}
            <div className="latency-cell total">
              <strong>{formatLatency(latencies?.total_latency ?? null)}</strong>
              <span>Total</span>
            </div>
          </div>
        </div>
        <div className="trace-row">
          <span>
            {verification == null || verification.unavailable
              ? "verification unavailable"
              : verification.grounded
                ? "grounded"
                : "not grounded"}
          </span>
          {verification == null || verification.unavailable ? (
            <p>{verification?.reason || "Verification did not run for this answer."}</p>
          ) : (
            <>
              <p>
                Score {formatScore(verification.score)} · {" "}
                {(verification.claims ?? []).filter((c) => c.supported).length}/
                {(verification.claims ?? []).length} claims supported
                {insufficient ? " · evidence was below the sufficiency threshold" : ""}
              </p>
              {(verification.unsupported_claims ?? []).length > 0 && (
                <div className="subquery-list">
                  {(verification.unsupported_claims ?? []).map((claim) => (
                    <p key={claim}>✕ {claim}</p>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
