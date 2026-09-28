import { create } from 'zustand';
import type { Answer } from './types';
import { api } from './api';

export interface Turn {
  id: string;
  question: string;
  status: 'thinking' | 'streaming' | 'done' | 'error';
  stage: number;
  partial: string;
  answer?: Answer;
  error?: string;
}

interface Selection {
  turnId: string;
  n: number;
}

interface AskState {
  turns: Turn[];
  selected: Selection | null;
  busy: boolean;
  ask: (question: string) => Promise<void>;
  select: (sel: Selection | null) => void;
  reset: () => void;
}

export const useAsk = create<AskState>((set, get) => ({
  turns: [],
  selected: null,
  busy: false,

  ask: async (question) => {
    if (get().busy || !question.trim()) return;
    const id = `t-${Date.now()}`;
    const patch = (p: Partial<Turn>) =>
      set((s) => ({ turns: s.turns.map((t) => (t.id === id ? { ...t, ...p } : t)) }));

    set((s) => ({
      busy: true,
      turns: [...s.turns, { id, question, status: 'thinking', stage: 0, partial: '' }],
    }));

    try {
      const answer = await api.ask(question, {
        onStage: (stage) => patch({ stage }),
        onToken: (partial) => patch({ partial, status: 'streaming' }),
      });
      patch({ status: 'done', answer, partial: answer.text });
      // Open the first video moment (or first source) in the evidence panel.
      const first = answer.citations.find((c) => c.start != null) ?? answer.citations[0];
      if (first) set({ selected: { turnId: id, n: first.n } });
    } catch (e) {
      patch({ status: 'error', error: e instanceof Error ? e.message : String(e) });
    } finally {
      set({ busy: false });
    }
  },

  select: (selected) => set({ selected }),
  reset: () => set({ turns: [], selected: null }),
}));
