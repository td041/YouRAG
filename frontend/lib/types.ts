export interface Collection {
  name: string;
  title: string;
  video_id: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  suggestions?: string[];
}

export interface IngestResult {
  chunks_added: number;
  latency: { total_s: number };
}
