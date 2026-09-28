import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { ArrowUp, FileText, Film, Image as ImageIcon, Database, Loader2, Check } from 'lucide-react';
import { AppShell, FirnMark } from '../components/AppShell';
import { CitationChip, EvidencePanel, EvidenceSummary, SourceRow } from '../components/Evidence';
import { ARCHIVE, ASK_STAGES, SUGGESTIONS } from '../demo/data';
import { useAsk, type Turn } from '../store';
import type { Citation } from '../types';
import { cn } from '@/lib/utils';

export default function Ask() {
  const { turns, selected, select, ask, busy } = useAsk();
  const scroller = useRef<HTMLDivElement>(null);
  const [params, setParams] = useSearchParams();

  // Shareable question links: /?q=<question> asks it on load.
  useEffect(() => {
    const q = params.get('q');
    if (q) {
      setParams({}, { replace: true });
      ask(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const selectedCitation =
    selected && turns.find((t) => t.id === selected.turnId)?.answer?.citations.find((c) => c.n === selected.n);

  return (
    <AppShell title="Ask" subtitle="Cited answers from reports, datasets, photos and video">
      <div className="h-full flex">
        <section className="flex-1 min-w-0 flex flex-col">
          <div ref={scroller} className="flex-1 overflow-y-auto scrollbar-thin">
            {turns.length === 0 ? (
              <Welcome onPick={ask} />
            ) : (
              <div className="max-w-[760px] mx-auto px-6 py-7 space-y-7">
                {turns.map((t) => (
                  <TurnView
                    key={t.id}
                    turn={t}
                    selectedN={selected?.turnId === t.id ? selected.n : null}
                    onSelect={(n) => select({ turnId: t.id, n })}
                  />
                ))}
              </div>
            )}
          </div>
          <Composer busy={busy} onSend={ask} compact={turns.length > 0} />
        </section>

        {turns.length > 0 && (
          <aside className="w-[420px] shrink-0 border-l border-[#DFE8EE] bg-white/70 overflow-y-auto scrollbar-thin">
            <EvidencePanel citation={selectedCitation ?? null} />
          </aside>
        )}
      </div>
    </AppShell>
  );
}

function Welcome({ onPick }: { onPick: (q: string) => void }) {
  const counts = {
    report: ARCHIVE.filter((i) => i.kind === 'report').length,
    video: ARCHIVE.filter((i) => i.kind === 'video').length,
    photo: ARCHIVE.filter((i) => i.kind === 'photo').length,
    dataset: ARCHIVE.filter((i) => i.kind === 'dataset').length,
  };
  return (
    <div className="max-w-[820px] mx-auto px-6 pt-16 pb-8">
      <div className="flex flex-col items-center text-center">
        <FirnMark size={54} />
        <h2 className="mt-5 text-[34px] font-extrabold tracking-tight text-ink">Ask India's polar archive</h2>
        <p className="mt-2 max-w-[560px] text-[15.5px] text-ink-2">
          Every answer cites its source, down to the page of a report or the exact second of an expedition video.
          Ask in English or any Indian language.
        </p>
      </div>

      <div className="mt-9 grid grid-cols-2 gap-3">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => onPick(s.question)}
            className="panel text-left px-4 py-3.5 hover:border-glacier hover:bg-[#FAFDFE] transition-colors"
          >
            <span className="label-mono text-glacier !text-[10px]">{s.lang === 'hi' ? 'हिंदी' : 'English'}</span>
            <span className={cn('block mt-1 text-[14.5px] font-semibold text-ink', s.lang === 'hi' && 'deva')}>{s.question}</span>
          </button>
        ))}
      </div>

      <div className="mt-8 flex items-center justify-center gap-6 text-[12.5px] text-ink-3">
        <Count icon={FileText} n={counts.report} label="reports" />
        <Count icon={Film} n={counts.video} label="videos" />
        <Count icon={ImageIcon} n={counts.photo} label="photos" />
        <Count icon={Database} n={counts.dataset} label="dataset" />
        <span className="text-[#B5C5D1]">·</span>
        <span>sample archive</span>
      </div>
    </div>
  );
}

function Count({ icon: Icon, n, label }: { icon: typeof FileText; n: number; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <Icon className="w-3.5 h-3.5" />
      <b className="text-ink-2">{n}</b> {label}
    </span>
  );
}

function TurnView({ turn, selectedN, onSelect }: { turn: Turn; selectedN: number | null; onSelect: (n: number) => void }) {
  const hindi = /[ऀ-ॿ]/.test(turn.question);
  const citations = turn.answer?.citations ?? [];
  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <div className={cn('max-w-[80%] rounded-2xl rounded-br-md bg-ink text-white px-4 py-2.5 text-[15px]', hindi && 'deva')}>
          {turn.question}
        </div>
      </div>

      <div className="panel p-5">
        <div className="flex items-center gap-2 mb-3">
          <FirnMark size={22} />
          <span className="text-[13px] font-bold text-ink">FIRN</span>
          {turn.status === 'done' && <span className="ml-auto"><EvidenceSummary citations={citations} /></span>}
        </div>

        {turn.status === 'thinking' ? (
          <Stages current={turn.stage} />
        ) : turn.status === 'error' ? (
          <p className="text-[14px] text-[#B8521A]">Could not reach the archive: {turn.error}</p>
        ) : (
          <AnswerText
            text={turn.partial}
            citations={citations}
            streaming={turn.status === 'streaming'}
            hindi={turn.answer?.lang === 'hi' || hindi}
            selectedN={selectedN}
            onSelect={onSelect}
          />
        )}

        {turn.status === 'done' && citations.length > 0 && (
          <div className="mt-4 pt-4 border-t border-[#EDF2F6]">
            <div className="label-mono text-ink-3 !text-[10px] mb-2.5">Sources</div>
            <div className="grid gap-2">
              {citations.map((c) => (
                <SourceRow key={c.n} citation={c} active={selectedN === c.n} onClick={() => onSelect(c.n)} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Stages({ current }: { current: number }) {
  return (
    <div className="grid grid-cols-4 gap-2">
      {ASK_STAGES.map((s, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div
            key={s.key}
            className={cn(
              'rounded-xl border px-3 py-2.5 transition-colors',
              active ? 'border-aurora-line bg-aurora-soft' : done ? 'border-[#CBE9DC] bg-verified-soft' : 'border-[#E3EBF1] bg-[#FAFCFD]',
            )}
          >
            <div className="flex items-center gap-1.5 text-[12.5px] font-bold text-ink">
              {done ? (
                <Check className="w-3.5 h-3.5 text-verified" />
              ) : active ? (
                <Loader2 className="w-3.5 h-3.5 text-aurora animate-spin" />
              ) : (
                <span className="w-3.5 h-3.5 rounded-full border border-[#C9D6DF]" />
              )}
              {s.label}
            </div>
            <div className="mt-1 text-[11.5px] text-ink-3 leading-snug">{s.detail}</div>
          </div>
        );
      })}
    </div>
  );
}

interface AnswerTextProps {
  text: string;
  citations: Citation[];
  streaming: boolean;
  hindi: boolean;
  selectedN: number | null;
  onSelect: (n: number) => void;
}

/** Renders markdown and turns [n] markers into clickable citation chips. */
function AnswerText({ text, citations, streaming, hindi, selectedN, onSelect }: AnswerTextProps) {
  const linked = text.replace(/\[(\d+)\]/g, '[$1](#cite-$1)');
  const byN = new Map(citations.map((c) => [c.n, c]));
  return (
    <div className={cn('answer text-[15.5px] text-ink', hindi && 'deva text-[16px]', streaming && 'streaming-cursor')}>
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            const n = Number(href?.replace('#cite-', ''));
            const c = byN.get(n);
            if (!c) return <sup className="font-mono text-[11px] text-ink-3">[{children as ReactNode}]</sup>;
            return <CitationChip citation={c} active={selectedN === n} onClick={() => onSelect(n)} />;
          },
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}

function Composer({ busy, onSend, compact }: { busy: boolean; onSend: (q: string) => void; compact: boolean }) {
  const [value, setValue] = useState('');
  const send = () => {
    if (!value.trim() || busy) return;
    onSend(value);
    setValue('');
  };
  return (
    <div className={cn('shrink-0 px-6 pb-6', compact ? 'pt-2' : 'pt-0')}>
      <div className="max-w-[760px] mx-auto">
        <div className="panel flex items-end gap-2 p-2 pl-4 focus-within:border-glacier">
          <textarea
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Ask about expeditions, stations, data or footage…  (English / हिंदी)"
            className="flex-1 resize-none bg-transparent py-2.5 text-[15px] text-ink placeholder:text-ink-3 outline-none max-h-40"
          />
          <button
            onClick={send}
            disabled={busy || !value.trim()}
            className="w-10 h-10 rounded-xl bg-glacier text-white flex items-center justify-center disabled:opacity-40 hover:bg-[#0a7ea3] transition-colors"
            aria-label="Send"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowUp className="w-4 h-4" />}
          </button>
        </div>
        <p className="mt-2 text-center text-[11.5px] text-ink-3">
          Answers come only from the archive and cite every source. Nothing is published without human review.
        </p>
      </div>
    </div>
  );
}
