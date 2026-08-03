export interface Source {
  n: number;
  doc_id: string;
  title?: string;
  snippet?: string;
  page?: number;
  bbox?: [number, number, number, number] | null;
  kind?: string;
  score?: number;
  file_url?: string;
  preview_url?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  citedNs?: number[];
  status?: "thinking" | "streaming" | "done" | "error";
  /** Backend UUID of the persisted assistant message (meta SSE event) — the
   *  id /feedback expects. Absent until the meta event arrives. */
  message_id?: string;
  /** Frontend-synthesized "thinking" telemetry rendered by the sonar radar
   *  drawer. The backend emits no phase events — phases are derived from the
   *  SSE timeline: `sources` ⇒ hybrid search + reranking done; first `token`
   *  ⇒ context cropped. */
  thinkingMeta?: ThinkingMeta;
}

/** RAG pipeline stage, in execution order. */
export type ThinkingPhase =
  | "query_expanded"
  | "hybrid_search"
  | "sonar_reranking"
  | "context_cropped";

/** The four pipeline steps in execution order — the radar drawer walks this
 *  list via ThinkingMeta.currentPhase. */
export const THINKING_PHASES: readonly ThinkingPhase[] = [
  "query_expanded",
  "hybrid_search",
  "sonar_reranking",
  "context_cropped",
] as const;

export interface ThinkingMeta {
  /** Ordered pipeline steps (THINKING_PHASES). */
  phases: ThinkingPhase[];
  /** Index of the phase currently executing. */
  currentPhase: number;
  /** Top-5 reranked sources, best first — rendered as ranked PDF cards. */
  topSources: Source[];
  /** Whether the drawer is expanded (per-message UI state). */
  isExpanded?: boolean;
  /** Milliseconds since the message was created; ticks while streaming. */
  elapsedMs?: number;
  /** Prompt as rewritten by the prefill agent (prefill SSE event) — what
   *  retrieval actually searched for; absent when no rewrite happened. */
  refinedPrompt?: string;
}

/** Arcade monster ingestion overlay state (PDF upload progress). */
export interface MonsterIngestState {
  filename: string;
  totalPages: number;
  currentPage: number;
  status: "crunching" | "paused" | "done" | "cancelled";
  /** Animation speed multiplier; "instant" = skip. */
  speed: FxSpeed;
}

/** Animation speed preset shared by the monster modal and the radar drawer.
 *  "instant" skips animation entirely. */
export type FxSpeed = 0.5 | 1 | 2 | 3 | 5 | "instant";

/** What the PDF viewer shows: one document plus every chunk that should be
 *  highlighted. A single document is often cited via several chunks — on the
 *  same page or across pages — and the viewer draws one box per chunk. */
export interface ViewerTarget {
  doc_id: string;
  title?: string;
  sources: Source[];
}

export interface Stats {
  documents: number;
  ready: boolean;
}
