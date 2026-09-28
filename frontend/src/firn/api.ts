// FIRN API layer. Demo mode (default) serves the sample archive locally so the
// UI runs without the backend; set VITE_DEMO_MODE=false to call the FastAPI
// chat pipeline instead.

import type { Answer, AskCallbacks, ArchiveItem, Citation, ShotRecord } from './types';
import { ANSWERS, ARCHIVE, ASK_STAGES, matchAnswer, shotsFor } from './demo/data';

export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE !== 'false';
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000';

export interface FirnApi {
  ask(question: string, cb?: AskCallbacks): Promise<Answer>;
  listArchive(): Promise<ArchiveItem[]>;
  getShots(videoId: string): Promise<ShotRecord[]>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const isHindi = (s: string) => /[ऀ-ॿ]/.test(s);

class DemoApi implements FirnApi {
  async ask(question: string, cb: AskCallbacks = {}): Promise<Answer> {
    const hit = matchAnswer(question);
    const lang = isHindi(question) ? 'hi' : 'en';
    const answer: Answer = hit
      ? { ...hit, question, lang }
      : {
          id: `a-${Date.now()}`,
          question,
          lang,
          text:
            lang === 'hi'
              ? 'यह प्रोटोटाइप नमूना डेटा पर चलता है। कृपया सुझाए गए प्रश्नों में से कोई एक आज़माएँ।'
              : 'This prototype runs on a small sample archive, so I could not find a cited answer. Try one of the suggested questions.',
          citations: [],
        };

    // Walk through the same stages the real retrieval pipeline reports.
    for (let i = 0; i < ASK_STAGES.length - 1; i++) {
      cb.onStage?.(i);
      await sleep(520);
    }
    cb.onStage?.(ASK_STAGES.length - 1);

    // Stream the text word by word, as the backend's /api/chat/stream does.
    const parts = answer.text.split(/(\s+)/);
    let partial = '';
    for (const p of parts) {
      partial += p;
      cb.onToken?.(partial);
      if (p.trim()) await sleep(28);
    }
    return answer;
  }

  async listArchive(): Promise<ArchiveItem[]> {
    return ARCHIVE;
  }

  async getShots(videoId: string): Promise<ShotRecord[]> {
    return shotsFor(videoId);
  }
}

interface BackendChunk {
  chunk_id: string;
  text: string;
  document_id: string;
  document_path: string;
  page_number?: number;
  start_seconds?: number | null;
  end_seconds?: number | null;
  origin: string;
}

class RealApi implements FirnApi {
  async ask(question: string, cb: AskCallbacks = {}): Promise<Answer> {
    cb.onStage?.(0);
    const res = await fetch(`${BACKEND_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text().catch(() => '')}`);
    cb.onStage?.(ASK_STAGES.length - 1);
    const data: { message: string; chunks?: BackendChunk[] } = await res.json();
    cb.onToken?.(data.message);

    const citations: Citation[] = (data.chunks ?? []).map((c, i) => ({
      n: i + 1,
      sourceId: c.document_id,
      page: c.start_seconds == null ? c.page_number : undefined,
      start: c.start_seconds ?? undefined,
      end: c.end_seconds ?? undefined,
      excerpt: c.text,
      relation: c.origin.includes('contradicts') ? 'contradicts' : c.origin === 'context' ? 'context' : 'supports',
    }));
    return { id: `a-${Date.now()}`, question, lang: isHindi(question) ? 'hi' : 'en', text: data.message, citations };
  }

  async listArchive(): Promise<ArchiveItem[]> {
    const res = await fetch(`${BACKEND_URL}/api/sources`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const rows: { id: string; name: string; uploadedAt: string }[] = await res.json();
    return rows.map((r) => ({
      id: r.id,
      kind: /\.(mp4|mkv|mov|webm)$/i.test(r.name) ? 'video' : /\.(jpe?g|png|webp)$/i.test(r.name) ? 'photo' : 'report',
      title: r.name,
      org: 'NCPOR',
      year: r.uploadedAt.slice(0, 4),
      language: '—',
      tags: [],
      summary: '',
      scene: 'document',
      sample: false,
    }));
  }

  async getShots(): Promise<ShotRecord[]> {
    return [];
  }
}

export const api: FirnApi = DEMO_MODE ? new DemoApi() : new RealApi();
export { ANSWERS };
