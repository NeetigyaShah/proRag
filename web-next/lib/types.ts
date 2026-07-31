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
}

export interface Stats {
  documents: number;
  ready: boolean;
}
