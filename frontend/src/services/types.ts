export interface Citation {
  id: string;
  source: string;
  page: number;
  text: string;
  score: number;
}

export interface References {
  images: string[];
  documents: string[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations: Citation[];
  references?: References;
  timestamp: string;
}

export interface StreamEvent {
  type: 'token' | 'citation' | 'done' | 'error';
  content?: string;
  citations?: Citation[];
  session_id?: string;
  metadata?: Record<string, unknown>;
  message?: string;
}

export interface Config {
  llm_model: string;
  embedding_model: string;
  db_type: string;
  backend_url: string;
  ingestion_url: string;
  api_keys: {
    openai: string;
    anthropic: string;
    local: string;
  };
}

export interface Memory {
  id: string;
  content: string;
  createdAt: string;
}

export interface Source {
  id: string;
  name: string;
  size: number;
  status: 'uploading' | 'indexing' | 'ready' | 'error';
  uploadedAt: string;
  pages?: number;
}

export interface ChatResponse {
  message: Message;
}

export interface ApiService {
  sendMessage: (content: string, history: Message[]) => Promise<ChatResponse>;
  getConfig: () => Promise<Config>;
  updateConfig: (config: Partial<Config>) => Promise<Config>;
  getMemories: () => Promise<Memory[]>;
  addMemory: (content: string) => Promise<Memory>;
  updateMemory: (id: string, content: string) => Promise<Memory>;
  deleteMemory: (id: string) => Promise<void>;
  getSources: () => Promise<Source[]>;
  uploadSource: (file: File) => Promise<Source>;
  deleteSource: (id: string) => Promise<void>;
}
