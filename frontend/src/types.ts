export type Modality = "text" | "url" | "pdf" | "image" | "audio" | "video" | "query";

export type RackPoint = {
  id: string;
  source_id: string;
  title: string;
  modality: Modality;
  projection: { x: number; y: number; z: number };
  color: string;
  preview?: string;
  score?: number;
};

export type RackSource = {
  id: string;
  title: string;
  modality: Modality;
  summary: string;
  chunks: number;
  created_at: number;
  metadata?: Record<string, unknown>;
};

export type SpaceSnapshot = {
  sources: RackSource[];
  points: RackPoint[];
  events: Array<Record<string, unknown>>;
  provider: string;
  dimensions: number;
  model: string;
  projection?: { method: string; basis: string };
};

export type Match = {
  id: string;
  source_id: string;
  title: string;
  modality: Modality;
  text: string;
  score: number;
  similarity?: number;
  relevance?: number | null;
  reason?: string | null;
  projection: { x: number; y: number; z: number };
};

export type Latencies = {
  routing_latency?: number;
  retrieval_latency?: number;
  reranking_latency?: number;
  generation_latency?: number;
  verification_latency?: number;
  total_latency?: number;
};

export type CandidateScore = {
  id: string;
  source_id: string;
  title: string;
  similarity: number;
};

export type VerificationState = {
  unavailable: boolean;
  reason?: string;
  grounded?: boolean | null;
  score?: number | null;
  claims?: Array<{ claim: string; supported: boolean; evidence_ids: string[] }>;
  unsupported_claims?: string[];
};

export type AskResponse = {
  answer: string;
  query?: string;
  matches: Match[];
  candidates?: CandidateScore[];
  candidate_count?: number;
  reranked?: boolean;
  rerank_fallback?: boolean;
  rerank_reason?: string;
  strategy?: string;
  router_reason?: string;
  subqueries?: string[];
  router_fell_back?: boolean;
  insufficient_evidence?: boolean;
  query_point: RackPoint;
  trace: Array<{ agent: string; status: string; detail: string }>;
  space: SpaceSnapshot;
  latencies?: Latencies;
  retrieval_latency?: number;
  reranking_latency?: number;
  generation_latency?: number;
  verification_latency?: number;
  total_latency?: number;
  grounded?: boolean;
  score?: number;
  claims?: VerificationState["claims"];
  unsupported_claims?: string[];
  verification_unavailable?: string;
};

export type EvalSummary = {
  benchmark: {
    name?: string;
    version?: string;
    question_count?: number;
    counts_by_category?: Record<string, number>;
  } | null;
  latest_run: {
    file?: string;
    timestamp?: string;
    question_count?: number;
    api_available?: boolean;
    embedding?: Record<string, unknown>;
    retrieval?: Record<string, number> | null;
    judge?: Record<string, number | null> | null;
    unanswerable?: Record<string, number> | null;
    corpus_drift?: { count?: number; questions?: string[] } | null;
  } | null;
  latest_comparison: {
    file?: string;
    timestamp?: string;
    question_count?: number;
    baseline?: Record<string, Record<string, number>> | null;
    improved?: Record<string, Record<string, number>> | null;
    differences_improved_minus_baseline?: Record<string, number | null> | null;
    corpus_mismatch?: boolean | null;
  } | null;
  notes: string[];
};
