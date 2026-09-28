import { Message, Config, Memory, Source } from './types';

export const mockConfig: Config = {
  llm_model: 'qwen-2.5-72b',
  embedding_model: 'nomic-embed-text',
  db_type: 'qdrant',
  backend_url: 'http://localhost:8000',
  ingestion_url: 'http://localhost:8001',
  api_keys: {
    openai: '',
    anthropic: '',
    local: '',
  },
};

export const mockMessages: Message[] = [
  {
    id: 'msg_1',
    role: 'assistant',
    content: 'Welcome to Nemoire. I\'m your intelligent research assistant. Ask me anything about your uploaded documents, and I\'ll provide precise answers with citations.',
    citations: [],
    timestamp: new Date(Date.now() - 3600000).toISOString(),
  },
  {
    id: 'msg_2',
    role: 'user',
    content: 'What is the Nemoire architecture and how does it work?',
    citations: [],
    timestamp: new Date(Date.now() - 3500000).toISOString(),
  },
  {
    id: 'msg_3',
    role: 'assistant',
    content: 'The Nemoire architecture emphasizes modularity and clean separation of concerns [1]. It separates the UI layer from the reasoning engine, allowing for flexible deployment across different infrastructure setups [2]. The system uses a multi-stage retrieval pipeline that first performs semantic search, then reranks results for precision [3].',
    citations: [
      {
        id: '1',
        source: 'architecture_overview.pdf',
        page: 4,
        text: 'The Nemoire architecture emphasizes modularity, enabling developers to swap components without affecting the overall system. This design philosophy ensures long-term maintainability and adaptability.',
        score: 0.95,
      },
      {
        id: '2',
        source: 'architecture_overview.pdf',
        page: 7,
        text: 'The separation between UI and reasoning engine allows for flexible deployment. The backend can run on dedicated GPU servers while the frontend remains lightweight and responsive.',
        score: 0.92,
      },
      {
        id: '3',
        source: 'retrieval_pipeline.pdf',
        page: 12,
        text: 'The multi-stage retrieval pipeline first performs semantic search using dense embeddings, then applies a cross-encoder reranker to ensure the highest precision for the final context window.',
        score: 0.89,
      },
    ],
    timestamp: new Date(Date.now() - 3400000).toISOString(),
  },
];

export const mockMemories: Memory[] = [
  {
    id: 'mem_1',
    content: 'User is a software developer with experience in React and Python',
    createdAt: new Date(Date.now() - 86400000).toISOString(),
  },
  {
    id: 'mem_2',
    content: 'Prefers concise, code-focused explanations',
    createdAt: new Date(Date.now() - 72000000).toISOString(),
  },
  {
    id: 'mem_3',
    content: 'Working on a RAG-based research assistant project',
    createdAt: new Date(Date.now() - 43200000).toISOString(),
  },
];

export const mockSources: Source[] = [
  {
    id: 'src_1',
    name: 'architecture_overview.pdf',
    size: 2456789,
    status: 'ready',
    uploadedAt: new Date(Date.now() - 172800000).toISOString(),
    pages: 24,
  },
  {
    id: 'src_2',
    name: 'retrieval_pipeline.pdf',
    size: 1234567,
    status: 'ready',
    uploadedAt: new Date(Date.now() - 86400000).toISOString(),
    pages: 18,
  },
  {
    id: 'src_3',
    name: 'embedding_models_comparison.docx',
    size: 987654,
    status: 'indexing',
    uploadedAt: new Date(Date.now() - 3600000).toISOString(),
    pages: 12,
  },
];

export const llmModels = [
  { value: 'qwen-2.5-72b', label: 'Qwen 2.5 72B' },
  { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
  { value: 'gpt-4o', label: 'GPT-4o' },
  { value: 'claude-3-opus', label: 'Claude 3 Opus' },
  { value: 'llama-3-70b', label: 'Llama 3 70B' },
];

export const embeddingModels = [
  { value: 'nomic-embed-text', label: 'Nomic Embed Text' },
  { value: 'text-embedding-3-large', label: 'OpenAI Embedding 3 Large' },
  { value: 'bge-large-en', label: 'BGE Large EN' },
  { value: 'e5-mistral-7b', label: 'E5 Mistral 7B' },
];

export const vectorDatabases = [
  { value: 'qdrant', label: 'Qdrant' },
  { value: 'pinecone', label: 'Pinecone' },
  { value: 'weaviate', label: 'Weaviate' },
  { value: 'chroma', label: 'Chroma' },
];
