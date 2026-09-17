import {
  Activity,
  Bot,
  Box,
  BrainCircuit,
  FileText,
  Link,
  Loader2,
  MessageSquare,
  Plus,
  RadioTower,
  Send,
  Sparkles,
  Upload,
  Video,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { API } from "./lib/api";
import { formatBytes, formatScore, scorePct } from "./lib/format";
import { sampleText } from "./lib/constants";
import { modalityIcon } from "./lib/modalityIcon";
import { AnswerContent } from "./components/AnswerContent";
import { EvaluationPanel } from "./components/EvaluationPanel";
import { PipelinePanel } from "./components/PipelinePanel";
import { SourceRow } from "./components/SourceRow";
import { VectorSpace } from "./components/VectorSpace";
import type { AskResponse, Latencies, Match, RackPoint, RackSource, SpaceSnapshot, VerificationState } from "./types";

export default function App() {
  const [space, setSpace] = useState<SpaceSnapshot | null>(null);
  const [tab, setTab] = useState<"text" | "url" | "file">("text");
  const [title, setTitle] = useState("Gemini Embedding 2 field note");
  const [text, setText] = useState(sampleText);
  const [url, setUrl] = useState("https://developers.googleblog.com/building-with-gemini-embedding-2/");
  const [notes, setNotes] = useState("Uploaded multimodal source for the agentic RAG workspace.");
  const [question, setQuestion] = useState("How does Gemini Embedding 2 help agentic RAG across modalities?");
  const [answer, setAnswer] = useState("");
  const [matches, setMatches] = useState<Match[]>([]);
  const [trace, setTrace] = useState<AskResponse["trace"]>([]);
  const [strategy, setStrategy] = useState<string | null>(null);
  const [routerReason, setRouterReason] = useState("");
  const [routerFellBack, setRouterFellBack] = useState(false);
  const [subqueries, setSubqueries] = useState<string[]>([]);
  const [candidateCount, setCandidateCount] = useState<number | null>(null);
  const [latencies, setLatencies] = useState<Latencies | null>(null);
  const [reranked, setReranked] = useState<boolean | null>(null);
  const [rerankFallback, setRerankFallback] = useState<boolean | null>(null);
  const [rerankReason, setRerankReason] = useState("");
  const [verification, setVerification] = useState<VerificationState | null>(null);
  const [insufficient, setInsufficient] = useState<boolean | null>(null);
  const [hasAsked, setHasAsked] = useState(false);
  const [queryPoint, setQueryPoint] = useState<RackPoint | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<RackPoint | null>(null);
  const [isAddingSource, setIsAddingSource] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [removingSourceId, setRemovingSourceId] = useState<string | null>(null);
  const [sourceStatus, setSourceStatus] = useState("");
  const [sourceError, setSourceError] = useState("");
  const [qaStatus, setQaStatus] = useState("");
  const [qaError, setQaError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const points = useMemo(() => space?.points ?? [], [space]);
  const sourceCount = space?.sources.length ?? 0;
  const pointCount = space?.points.length ?? 0;
  const provider = space?.provider ?? "loading";
  const projection = space?.projection?.method?.replace("_", " ").toUpperCase() ?? "PCA 3D";

  async function refreshSpace() {
    const res = await fetch(`${API}/space`);
    setSpace(await res.json());
  }

  useEffect(() => {
    refreshSpace().catch(() => undefined);
  }, []);

  async function addSource() {
    setIsAddingSource(true);
    setSourceError("");
    setSourceStatus(tab === "file" ? "Uploading and embedding media..." : "Embedding source...");
    try {
      let res: Response;
      if (tab === "text") {
        res = await fetch(`${API}/sources/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title, text, modality: "text" }),
        });
      } else if (tab === "url") {
        res = await fetch(`${API}/sources/url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, title: title || undefined }),
        });
      } else {
        const file = fileRef.current?.files?.[0];
        if (!file) {
          setSourceError("Choose a file first.");
          return;
        }
        const form = new FormData();
        form.append("title", title || file.name);
        form.append("file", file);
        form.append("notes", notes);
        res = await fetch(`${API}/sources/file`, { method: "POST", body: form });
      }

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Source ingestion failed.");
      setSpace(data.space);
      setSourceStatus(`${data.source?.title || "Source"} embedded into the map.`);
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : "Source ingestion failed.");
    } finally {
      setIsAddingSource(false);
    }
  }

  async function removeSource(source: RackSource) {
    const confirmed = window.confirm(`Remove "${source.title}" from this local vector space?`);
    if (!confirmed) return;

    setRemovingSourceId(source.id);
    setSourceError("");
    try {
      const res = await fetch(`${API}/sources/${source.id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed");
      const data = await res.json();
      setSpace(data.space);
      setMatches((current) => current.filter((match) => match.source_id !== source.id));
      if (selectedPoint?.source_id === source.id) setSelectedPoint(null);
      if (queryPoint) setQueryPoint(null);
      setSourceStatus(`${source.title} removed.`);
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : "Delete failed.");
    } finally {
      setRemovingSourceId(null);
    }
  }

  async function askQuestion() {
    if (!question.trim()) return;
    setIsAsking(true);
    setQaError("");
    setQaStatus("Retrieving evidence and asking the ADK coordinator...");
    setAnswer("");
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 6 }),
      });
      const data: AskResponse = await res.json();
      if (!res.ok) throw new Error((data as unknown as { detail?: string }).detail || "Question failed.");
      setAnswer(data.answer);
      setMatches(data.matches ?? []);
      setTrace(data.trace ?? []);
      setStrategy(typeof data.strategy === "string" ? data.strategy : null);
      setRouterReason(data.router_reason ?? "");
      setRouterFellBack(Boolean(data.router_fell_back));
      setSubqueries(Array.isArray(data.subqueries) ? data.subqueries : []);
      setCandidateCount(
        typeof data.candidate_count === "number"
          ? data.candidate_count
          : Array.isArray(data.candidates)
            ? data.candidates.length
            : null,
      );
      setLatencies(data.latencies ?? {
        retrieval_latency: data.retrieval_latency,
        reranking_latency: data.reranking_latency,
        generation_latency: data.generation_latency,
        verification_latency: data.verification_latency,
        total_latency: data.total_latency,
      });
      setReranked(typeof data.reranked === "boolean" ? data.reranked : null);
      setRerankFallback(typeof data.rerank_fallback === "boolean" ? data.rerank_fallback : null);
      setRerankReason(data.rerank_reason ?? "");
      setInsufficient(typeof data.insufficient_evidence === "boolean" ? data.insufficient_evidence : null);
      if (typeof data.verification_unavailable === "string") {
        setVerification({ unavailable: true, reason: data.verification_unavailable });
      } else if (typeof data.grounded === "boolean" || typeof data.score === "number") {
        setVerification({
          unavailable: false,
          grounded: data.grounded ?? null,
          score: data.score ?? null,
          claims: Array.isArray(data.claims) ? data.claims : [],
          unsupported_claims: Array.isArray(data.unsupported_claims) ? data.unsupported_claims : [],
        });
      } else {
        setVerification({ unavailable: true, reason: "Verification did not run for this answer." });
      }
      setHasAsked(true);
      setQueryPoint(data.query_point);
      setSpace(data.space);
      setQaStatus(`Retrieved ${data.matches.length} citation${data.matches.length === 1 ? "" : "s"}.`);
    } catch (error) {
      setQaError(error instanceof Error ? error.message : "Something went wrong.");
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={22} />
          </div>
          <div>
            <h1>Multi-modal Agentic RAG</h1>
            <p>Gemini Embedding 2 · Google ADK</p>
          </div>
        </div>
        <div className="status-strip">
          <span><RadioTower size={14} /> {provider}</span>
          <span><Box size={14} /> {pointCount} points</span>
          <span><Activity size={14} /> {sourceCount} sources</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="left-rail">
          <section className="panel source-list">
            <div className="panel-heading source-list-heading">
              <div>
                <h2>Indexed sources</h2>
                <p>Remove sources when they should leave the vector space.</p>
              </div>
            </div>
            {space?.sources.map((source) => (
              <SourceRow
                source={source}
                key={source.id}
                isRemoving={removingSourceId === source.id}
                onRemove={removeSource}
              />
            ))}
          </section>

          <section className="panel source-panel">
            <div className="panel-heading">
              <div>
                <h2>Add source</h2>
                <p>Embed new evidence into the shared vector space.</p>
              </div>
              <button className="icon-button" onClick={refreshSpace} aria-label="Refresh embedding space">
                <Activity size={16} />
              </button>
            </div>

            <div className="tabs" role="tablist">
              <button className={tab === "text" ? "active" : ""} onClick={() => setTab("text")}><FileText size={14} /> Text</button>
              <button className={tab === "url" ? "active" : ""} onClick={() => setTab("url")}><Link size={14} /> URL</button>
              <button className={tab === "file" ? "active" : ""} onClick={() => setTab("file")}><Upload size={14} /> File</button>
            </div>

            <label className="field-label">Title</label>
            <input value={title} onChange={(event) => setTitle(event.target.value)} aria-label="Source title" />

            {tab === "text" && (
              <>
                <label className="field-label">Source text</label>
                <textarea value={text} onChange={(event) => setText(event.target.value)} aria-label="Source text" />
              </>
            )}

            {tab === "url" && (
              <>
                <label className="field-label">URL</label>
                <input value={url} onChange={(event) => setUrl(event.target.value)} aria-label="URL source" />
              </>
            )}

            {tab === "file" && (
              <>
                <label className="field-label">File</label>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".txt,.md,.pdf,image/*,audio/*,video/*"
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;
                    setSelectedFile(file);
                    if (file && title === "Gemini Embedding 2 field note") setTitle(file.name);
                  }}
                />
                {selectedFile && (
                  <div className="file-preview">
                    <Video size={15} />
                    <span>{selectedFile.name}</span>
                    <strong>{selectedFile.type || "file"} · {formatBytes(selectedFile.size)}</strong>
                  </div>
                )}
                <label className="field-label">Notes</label>
                <textarea value={notes} onChange={(event) => setNotes(event.target.value)} aria-label="File notes" />
              </>
            )}

            <button className="primary-button" onClick={addSource} disabled={isAddingSource}>
              {isAddingSource ? <Loader2 className="spin" size={16} /> : <Plus size={16} />} Add source
            </button>
            {(sourceStatus || sourceError) && (
              <div className={`inline-status ${sourceError ? "error" : "success"}`} role="status">
                {sourceError || sourceStatus}
              </div>
            )}
          </section>
        </aside>

        <section className="space-stage">
          <div className="stage-header">
            <div>
              <h2>Embedding Space</h2>
              <p>{space?.dimensions ?? 768}D embeddings · {projection} · one point per source</p>
            </div>
            <div className="stage-tools">
              <div className="modality-key" aria-label="Modality legend">
                {(["Text", "Image", "Audio", "Video", "PDF", "Query"] as const).map((item) => (
                  <span key={item} className={`modality-key-item key-${item.toLowerCase()}`}>{item}</span>
                ))}
              </div>
              <div className="space-readout" aria-label="Embedding space status">
                <span>{sourceCount} sources</span>
                <span>{matches.length ? `${matches.length} matched` : "ready"}</span>
              </div>
            </div>
          </div>
          <VectorSpace
            points={points}
            queryPoint={queryPoint}
            matches={matches}
            selectedId={selectedPoint?.id ?? null}
            onSelect={setSelectedPoint}
          />
          {selectedPoint && (
            <div className="hover-card">
              <div className={`mini-dot ${selectedPoint.modality}`} />
              <strong>{selectedPoint.title}</strong>
              <span>{selectedPoint.modality} · {selectedPoint.id}</span>
              <p>{selectedPoint.preview}</p>
            </div>
          )}
        </section>

        <aside className="right-rail">
          <section className="panel qa-panel">
            <div className="panel-heading">
              <div>
                <h2>Q&A</h2>
                <p>Ask a question and read the grounded answer here.</p>
              </div>
              <Bot size={18} />
            </div>
            <label className="field-label">Question</label>
            <textarea className="question-box" value={question} onChange={(event) => setQuestion(event.target.value)} aria-label="Question" />
            <button className="primary-button" onClick={askQuestion} disabled={isAsking}>
              {isAsking ? <Loader2 className="spin" size={16} /> : <Send size={16} />} Ask question
            </button>
            {(qaStatus || qaError) && (
              <div className={`inline-status ${qaError ? "error" : "success"}`} role="status">
                {qaError || qaStatus}
              </div>
            )}
            <div className="answer-box prominent-answer">
              <MessageSquare size={16} />
              <AnswerContent answer={answer} />
            </div>
          </section>

          <section className="panel trace-panel">
            <div className="panel-heading">
              <div>
                <h2>Agent Trace</h2>
                <p>Google ADK tool path</p>
              </div>
              <Sparkles size={18} />
            </div>
            <div className="trace-list">
              {(trace.length ? trace : [
                { agent: "source_ingestor", status: "ready", detail: "Waiting for a question" },
                { agent: "retrieval_tool", status: "ready", detail: "Nearest-neighbor evidence will appear here" },
                { agent: "answer_synthesizer", status: "ready", detail: "Cited answer stream target" },
              ]).map((step) => (
                <div className="trace-row" key={step.agent}>
                  <span>{step.agent}</span>
                  <p>{step.detail}</p>
                </div>
              ))}
            </div>
          </section>

          <PipelinePanel
            strategy={strategy}
            routerReason={routerReason}
            routerFellBack={routerFellBack}
            subqueries={subqueries}
            candidateCount={candidateCount}
            latencies={latencies}
            reranked={reranked}
            rerankFallback={rerankFallback}
            rerankReason={rerankReason}
            verification={verification}
            insufficient={insufficient}
            hasAsked={hasAsked}
          />

          <section className="panel citations-panel">
            <div className="panel-heading">
              <div>
                <h2>Citations</h2>
                <p>Sources used for this answer</p>
              </div>
            </div>
            <div className="citation-list">
              {matches.length === 0 && <div className="empty-state">No query yet. Add a source, then ask a question.</div>}
              {matches.map((match) => {
                const Icon = modalityIcon[match.modality] || FileText;
                return (
                  <button
                    className="citation-row"
                    key={match.id}
                    onMouseEnter={() => setSelectedPoint({ ...match, color: "#3b82f6", preview: match.text })}
                    onMouseLeave={() => setSelectedPoint(null)}
                  >
                    <div className="citation-top">
                      <span><Icon size={14} /> {match.title}</span>
                      <strong>{scorePct(match.score)}</strong>
                    </div>
                    <div className="score-track" aria-hidden="true"><div style={{ width: scorePct(match.score) }} /></div>
                    <p>{match.text}</p>
                    {(typeof match.similarity === "number" || typeof match.relevance === "number") && (
                      <div className="source-meta">
                        sim {formatScore(match.similarity ?? match.score)} · rerank{" "}
                        {typeof match.relevance === "number" ? formatScore(match.relevance) : "fallback"}
                        {match.reason ? ` · ${match.reason}` : ""}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </section>

          <EvaluationPanel />
        </aside>
      </section>
    </main>
  );
}
