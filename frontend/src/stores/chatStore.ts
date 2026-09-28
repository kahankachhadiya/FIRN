import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, Citation, References, StreamEvent } from '@/services/types';
import { useConfigStore } from '@/stores/configStore';

export interface Conversation {
  id: string;          // === backend session_id (UUID)
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  isLoading: boolean;
  isStreaming: boolean;
  streamingContent: string;
  activeCitation: { messageId: string; citationId: string } | null;

  // Actions
  createConversation: () => Promise<string>;
  deleteConversation: (id: string) => void;
  setActiveConversation: (id: string) => void;
  loadSessionFromBackend: (sessionId: string) => Promise<void>;
  sendMessage: (content: string, files?: File[]) => Promise<void>;
  setActiveCitation: (messageId: string | null, citationId: string | null) => void;
  clearCurrentChat: () => void;
  getActiveConversation: () => Conversation | null;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeConversationId: null,
      isLoading: false,
      isStreaming: false,
      streamingContent: '',
      activeCitation: null,

      // ── Create conversation ──────────────────────────────────────────────
      // Calls the backend first so the UUID is shared from the start.
      createConversation: async () => {
        const backendUrl = useConfigStore.getState().config.backend_url;
        let sessionId: string;
        let title = 'New Chat';
        let createdAt = new Date().toISOString();

        try {
          const res = await fetch(`${backendUrl}/api/sessions`, { method: 'POST' });
          if (res.ok) {
            const data = await res.json();
            sessionId = data.session_id;
            title = data.title ?? title;
            createdAt = data.created_at ?? createdAt;
          } else {
            // Fallback: generate a UUID client-side if backend is unreachable
            sessionId = crypto.randomUUID();
          }
        } catch {
          sessionId = crypto.randomUUID();
        }

        const newConvo: Conversation = {
          id: sessionId,
          title,
          messages: [],
          createdAt,
          updatedAt: createdAt,
        };

        set((state) => ({
          conversations: [newConvo, ...state.conversations],
          activeConversationId: sessionId,
        }));

        return sessionId;
      },

      // ── Delete conversation ──────────────────────────────────────────────
      deleteConversation: (id: string) => {
        set((state) => {
          const filtered = state.conversations.filter((c) => c.id !== id);
          const newActiveId =
            state.activeConversationId === id
              ? filtered[0]?.id ?? null
              : state.activeConversationId;
          return { conversations: filtered, activeConversationId: newActiveId };
        });
      },

      // ── Switch active conversation ───────────────────────────────────────
      // Immediately switches the view, then refreshes messages from backend.
      setActiveConversation: (id: string) => {
        set({ activeConversationId: id, activeCitation: null });
        get().loadSessionFromBackend(id);
      },

      // ── Load session messages from backend ───────────────────────────────
      loadSessionFromBackend: async (sessionId: string) => {
        const backendUrl = useConfigStore.getState().config.backend_url;
        try {
          const res = await fetch(`${backendUrl}/api/sessions/${sessionId}`);
          if (!res.ok) return;
          const data = await res.json();

          const backendMessages: Array<{
            id: string;
            role: string;
            content: string;
            timestamp: string;
          }> = data.messages ?? [];

          const messages: Message[] = backendMessages
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .map((m) => {
              let references: References | undefined;
              if (m.role === 'assistant') {
                const images: string[] = [];
                const documents: string[] = [];
                const linkPattern = /<link>([\s\S]*?)<\/link>/gi;
                let match;
                while ((match = linkPattern.exec(m.content)) !== null) {
                  const url = match[1].trim();
                  if (/\.(png|jpg|jpeg|gif|webp)$/i.test(url)) images.push(url);
                  else if (/\.(pdf|docx?|html?|txt|md)$/i.test(url)) documents.push(url);
                }
                references = { images, documents };
              }
              return {
                id: m.id,
                role: m.role as 'user' | 'assistant',
                content: m.content,
                citations: [],
                references,
                timestamp: m.timestamp,
              };
            });

          set((state) => ({
            conversations: state.conversations.map((c) =>
              c.id === sessionId
                ? { ...c, messages, title: data.title ?? c.title, updatedAt: data.updated_at ?? c.updatedAt }
                : c
            ),
          }));
        } catch {
          // silently keep cached messages on network error
        }
      },

      // ── Send message ─────────────────────────────────────────────────────
      sendMessage: async (content: string, files?: File[]) => {
        let { activeConversationId } = get();

        // Create a new conversation if none is active
        if (!activeConversationId) {
          activeConversationId = await get().createConversation();
        }

        const convoId = activeConversationId;

        const userMessage: Message = {
          id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          role: 'user',
          content: files?.length
            ? `${content}\n\n📎 ${files.map((f) => f.name).join(', ')}`
            : content,
          citations: [],
          timestamp: new Date().toISOString(),
        };

        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === convoId
              ? {
                  ...c,
                  messages: [...c.messages, userMessage],
                  updatedAt: new Date().toISOString(),
                  title:
                    c.messages.length === 0
                      ? content.slice(0, 40) + (content.length > 40 ? '...' : '')
                      : c.title,
                }
              : c
          ),
          isLoading: true,
          isStreaming: true,
          streamingContent: '',
        }));

        const backendUrl = useConfigStore.getState().config.backend_url;

        try {
          const response = await fetch(`${backendUrl}/api/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: content,
              // convoId IS the backend session_id — always pass it
              session_id: convoId,
            }),
          });

          if (!response.ok || !response.body) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          let accumulatedContent = '';
          let pendingCitations: Citation[] = [];
          let pendingReferences: References | undefined;

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed.startsWith('data:')) continue;
              const jsonStr = trimmed.slice('data:'.length).trim();
              if (!jsonStr) continue;

              let event: StreamEvent;
              try { event = JSON.parse(jsonStr); } catch { continue; }

              if (event.type === 'token' && event.content != null) {
                accumulatedContent += event.content;
                set({ streamingContent: accumulatedContent });

              } else if (event.type === 'citation' && event.citations) {
                pendingCitations = event.citations;

              } else if (event.type === 'done') {
                const meta = event.metadata as Record<string, unknown> | undefined;
                if (meta?.references) pendingReferences = meta.references as References;

                const assistantMessage: Message = {
                  id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                  role: 'assistant',
                  content: accumulatedContent,
                  citations: pendingCitations,
                  references: pendingReferences,
                  timestamp: new Date().toISOString(),
                };

                set((state) => ({
                  conversations: state.conversations.map((c) =>
                    c.id === convoId
                      ? { ...c, messages: [...c.messages, assistantMessage], updatedAt: new Date().toISOString() }
                      : c
                  ),
                  isLoading: false,
                  isStreaming: false,
                  streamingContent: '',
                }));
                return;

              } else if (event.type === 'error') {
                const errorMessage: Message = {
                  id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                  role: 'assistant',
                  content: `Error: ${event.message ?? 'An unknown error occurred.'}`,
                  citations: [],
                  timestamp: new Date().toISOString(),
                };
                set((state) => ({
                  conversations: state.conversations.map((c) =>
                    c.id === convoId ? { ...c, messages: [...c.messages, errorMessage], updatedAt: new Date().toISOString() } : c
                  ),
                  isLoading: false,
                  isStreaming: false,
                  streamingContent: '',
                }));
                return;
              }
            }
          }

          // Stream ended without done/error — finalize
          if (accumulatedContent) {
            const assistantMessage: Message = {
              id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              role: 'assistant',
              content: accumulatedContent,
              citations: pendingCitations,
              references: pendingReferences,
              timestamp: new Date().toISOString(),
            };
            set((state) => ({
              conversations: state.conversations.map((c) =>
                c.id === convoId ? { ...c, messages: [...c.messages, assistantMessage], updatedAt: new Date().toISOString() } : c
              ),
              isLoading: false,
              isStreaming: false,
              streamingContent: '',
            }));
          } else {
            set({ isLoading: false, isStreaming: false, streamingContent: '' });
          }

        } catch (err) {
          const errorMessage: Message = {
            id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            role: 'assistant',
            content: `Error: ${err instanceof Error ? err.message : 'Failed to connect to the server.'}`,
            citations: [],
            timestamp: new Date().toISOString(),
          };
          set((state) => ({
            conversations: state.conversations.map((c) =>
              c.id === convoId ? { ...c, messages: [...c.messages, errorMessage], updatedAt: new Date().toISOString() } : c
            ),
            isLoading: false,
            isStreaming: false,
            streamingContent: '',
          }));
        }
      },

      // ── Misc ─────────────────────────────────────────────────────────────
      setActiveCitation: (messageId, citationId) => {
        if (messageId && citationId) {
          set({ activeCitation: { messageId, citationId } });
        } else {
          set({ activeCitation: null });
        }
      },

      clearCurrentChat: () => {
        const { activeConversationId } = get();
        if (!activeConversationId) return;
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === activeConversationId ? { ...c, messages: [], title: 'New Chat' } : c
          ),
          activeCitation: null,
        }));
      },

      getActiveConversation: () => {
        const { conversations, activeConversationId } = get();
        return conversations.find((c) => c.id === activeConversationId) ?? null;
      },
    }),
    { name: 'nemoire-chat' }
  )
);
