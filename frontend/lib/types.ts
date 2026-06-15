export interface Collection {
  name: string;
  title: string;
  video_id: string | null;
}

export interface Source {
  label: string;                       // "00:18–03:31"
  start_time: number;
  video_id: string | null;
  title: string | null;
  chunk_type?: "text" | "visual";     // "visual" = từ frame description
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  suggestions?: string[];
}

export interface IngestResult {
  chunks_added: number;
  latency: { total_s: number };
}
