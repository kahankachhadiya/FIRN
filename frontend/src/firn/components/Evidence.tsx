import { ExternalLink, FileText, Film, Image as ImageIcon, Database, ShieldCheck, ShieldAlert } from 'lucide-react';
import type { ArchiveItem, Citation, SourceKind } from '../types';
import { findItem, shotsFor } from '../demo/data';
import { formatTime } from '../format';
import { VideoMoment } from './VideoMoment';
import { cn } from '@/lib/utils';

export const KIND_ICON: Record<SourceKind, typeof FileText> = {
  report: FileText,
  video: Film,
  photo: ImageIcon,
  dataset: Database,
};

export const KIND_LABEL: Record<SourceKind, string> = {
  report: 'Report',
  video: 'Video',
  photo: 'Photo',
  dataset: 'Dataset',
};

export function locatorLabel(c: Citation): string | null {
  if (c.start != null) return formatTime(c.start);
  if (c.section) return c.section;
  if (c.page != null) return `p. ${c.page}`;
  return null;
}

interface ChipProps {
  citation: Citation;
  active: boolean;
  onClick: () => void;
}

/** Inline citation marker. Video moments show the timestamp. */
export function CitationChip({ citation, active, onClick }: ChipProps) {
  const isVideo = citation.start != null;
  return (
    <button
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1 align-[1px] mx-0.5 h-[22px] px-1.5 rounded-md text-[11.5px] font-bold font-mono transition-colors',
        isVideo
          ? active ? 'bg-ink text-white' : 'bg-[#E8EEF3] text-ink hover:bg-[#dbe4eb]'
          : active ? 'bg-glacier text-white' : 'bg-glacier-soft text-[#06698A] hover:bg-[#d7edf6]',
      )}
    >
      {isVideo && <span className="text-[9px]">▶</span>}
      {isVideo ? formatTime(citation.start!) : citation.n}
    </button>
  );
}

const RELATION = {
  supports: { label: 'Supports the answer', cls: 'bg-verified-soft text-verified-ink border-[#A9DCC5]' },
  context: { label: 'Adds context', cls: 'bg-[#F1F5F8] text-ink-2 border-[#DDE6EC]' },
  contradicts: { label: 'Conflicting evidence', cls: 'bg-alert-soft text-[#B8521A] border-[#F7C4A3]' },
} as const;

export function SourceRow({ citation, active, onClick }: ChipProps) {
  const item = findItem(citation.sourceId);
  if (!item) return null;
  const Icon = KIND_ICON[item.kind];
  const loc = locatorLabel(citation);
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left flex items-center gap-3 rounded-xl border px-3 py-2.5 transition-colors',
        active ? 'border-glacier bg-glacier-soft/60' : 'border-[#E3EBF1] bg-white hover:bg-[#F8FBFC]',
      )}
    >
      <span className="w-6 text-center font-mono text-[12px] font-bold text-glacier">{citation.n}</span>
      <span className="w-8 h-8 rounded-lg bg-[#EEF4F8] flex items-center justify-center shrink-0">
        <Icon className="w-4 h-4 text-ink-2" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-[13.5px] font-semibold text-ink truncate">{item.title}</span>
        <span className="block text-[12px] text-ink-3 truncate">
          {item.org} · {item.year}
          {loc && <> · {citation.start != null ? `▶ ${loc}` : loc}</>}
        </span>
      </span>
    </button>
  );
}

export function EvidenceSummary({ citations }: { citations: Citation[] }) {
  const sources = new Set(citations.map((c) => c.sourceId)).size;
  const moments = citations.filter((c) => c.start != null).length;
  const conflicts = citations.filter((c) => c.relation === 'contradicts').length;
  const ok = conflicts === 0;
  const Icon = ok ? ShieldCheck : ShieldAlert;
  return (
    <div className={cn('flex items-center gap-2 text-[12.5px] font-semibold', ok ? 'text-verified-ink' : 'text-[#B8521A]')}>
      <Icon className="w-4 h-4" />
      <span>
        {sources} {sources === 1 ? 'source' : 'sources'}
        {moments > 0 && <> · {moments} video {moments === 1 ? 'moment' : 'moments'}</>}
        {' · '}
        {ok ? 'no conflicting evidence' : `${conflicts} conflicting`}
      </span>
    </div>
  );
}

export function EvidencePanel({ citation }: { citation: Citation | null }) {
  if (!citation) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center px-8 text-ink-3">
        <ShieldCheck className="w-8 h-8 mb-3 text-[#B5C5D1]" />
        <p className="text-[13.5px] font-semibold text-ink-2">Evidence appears here</p>
        <p className="text-[12.5px] mt-1">Select a citation to see the exact passage or video moment behind it.</p>
      </div>
    );
  }
  const item = findItem(citation.sourceId);
  if (!item) return null;
  return (
    <div className="p-5 space-y-4">
      <div className="flex items-center justify-between">
        <span className="label-mono text-ink-3">Evidence [{citation.n}]</span>
        <span className={cn('chip h-6 text-[11px]', RELATION[citation.relation].cls)}>{RELATION[citation.relation].label}</span>
      </div>
      <SourceHeader item={item} citation={citation} />
      {item.kind === 'video' && citation.start != null ? (
        <VideoMoment video={item} shots={shotsFor(item.id)} start={citation.start} end={citation.end} />
      ) : (
        <blockquote className="rounded-xl border border-[#E3EBF1] bg-[#FAFCFD] p-4 text-[13.5px] leading-relaxed text-ink border-l-[3px] border-l-glacier">
          {citation.excerpt}
        </blockquote>
      )}
      {item.url && (
        <a
          href={item.url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-[13px] font-semibold text-glacier hover:underline"
        >
          Open original source <ExternalLink className="w-3.5 h-3.5" />
        </a>
      )}
    </div>
  );
}

function SourceHeader({ item, citation }: { item: ArchiveItem; citation: Citation }) {
  const Icon = KIND_ICON[item.kind];
  const loc = locatorLabel(citation);
  return (
    <div className="flex items-start gap-3">
      <span className="w-10 h-10 rounded-xl bg-glacier-soft flex items-center justify-center shrink-0">
        <Icon className="w-5 h-5 text-glacier" />
      </span>
      <div className="min-w-0">
        <div className="text-[15px] font-bold text-ink leading-snug">{item.title}</div>
        <div className="text-[12.5px] text-ink-3 mt-0.5">
          {KIND_LABEL[item.kind]} · {item.org} · {item.year}
          {loc && item.kind !== 'video' && <> · {loc}</>}
        </div>
      </div>
    </div>
  );
}
