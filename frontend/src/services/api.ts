/**
 * Nemoire API Service Layer
 *
 * Communicates with the real FastAPI backend.
 */

import { Message, Config, Memory, Source, ChatResponse, ApiService, Citation, References } from './types';
import { useConfigStore } from '@/stores/configStore';

// Backend ChatResponse shape (differs from frontend ChatResponse)
interface BackendChatResponse {
  session_id: string;
  message: string;
  thinking?: string;
  chunks?: unknown[];
  files?: unknown[];
  source_pdfs?: string[];
  references?: {
    images: string[];
    documents: string[];
  };
  metadata?: Record<string, unknown>;
}

class RealApiService implements ApiService {
  private get backendUrl(): string {
    return useConfigStore.getState().config.backend_url;
  }

  private get ingestionBackendUrl(): string {
    return useConfigStore.getState().config.ingestion_url ?? 'http://localhost:8001';
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const url = `${this.backendUrl}${path}`;
    const options: RequestInit = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);
    if (!response.ok) {
      const errorBody = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}: ${errorBody}`);
    }
    return response.json() as Promise<T>;
  }

  async sendMessage(content: string, history: Message[]): Promise<ChatResponse> {
    const backendResponse = await this.request<BackendChatResponse>('POST', '/api/chat', {
      message: content,
      history,
    });

    const citations: Citation[] = [];
    const references: References = {
      images: backendResponse.references?.images ?? [],
      documents: backendResponse.references?.documents ?? (backendResponse.source_pdfs ?? []),
    };

    const message: Message = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      role: 'assistant',
      content: backendResponse.message,
      citations,
      references,
      timestamp: new Date().toISOString(),
    };

    return { message };
  }

  async getConfig(): Promise<Config> {
    return this.request<Config>('GET', '/api/config');
  }

  async updateConfig(updates: Partial<Config>): Promise<Config> {
    return this.request<Config>('PUT', '/api/config', updates);
  }

  async getMemories(): Promise<Memory[]> {
    return this.request<Memory[]>('GET', '/api/memory');
  }

  async addMemory(content: string): Promise<Memory> {
    return this.request<Memory>('POST', '/api/memory', { content });
  }

  async updateMemory(id: string, content: string): Promise<Memory> {
    return this.request<Memory>('PUT', `/api/memory/${id}`, { content });
  }

  async deleteMemory(id: string): Promise<void> {
    await this.request<void>('DELETE', `/api/memory/${id}`);
  }

  async getSources(): Promise<Source[]> {
    return this.request<Source[]>('GET', '/api/sources');
  }

  async uploadSource(file: File): Promise<Source> {
    const url = `${this.ingestionBackendUrl}/api/processing/upload`;
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      // Do NOT set Content-Type — let fetch set it with the boundary
    });

    if (!response.ok) {
      const errorBody = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}: ${errorBody}`);
    }

    return response.json() as Promise<Source>;
  }

  async deleteSource(id: string): Promise<void> {
    await this.request<void>('DELETE', `/api/sources/${id}`);
  }
}

export const api = new RealApiService();
