import { useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import type { ArchiveItem, ShotRecord } from '../types';
import { formatTime } from '../format';
import { Scene } from './Scene';
import { cn } from '@/lib/utils';

interface VideoMomentProps {
  video: ArchiveItem;
  shots: ShotRecord[];
  start: number;
  end?: number;
  autoPlay?: boolean;
  compact?: boolean;
}

/**
 * Plays a video from a cited moment. Sample videos have no footage, so the
 * frame is drawn from the shot's keyframe scene; the timeline, shot record and
 * seek behaviour are the same as for real media.
 */
export function VideoMoment({ video, shots, start, end, autoPlay = true, compact = false }: VideoMomentProps) {
  const duration = video.duration ?? Math.max(...shots.map((s) => s.end), start + 30);
  const [time, setTime] = useState(start);
  const [playing, setPlaying] = useState(autoPlay);
  const last = useRef<number | null>(null);

  useEffect(() => {
    setTime(start);
    setPlaying(autoPlay);
  }, [video.id, start, autoPlay]);

  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    // A 30 fps interval (rather than requestAnimationFrame) keeps playback
    // deterministic when the page is captured frame by frame.
    const id = window.setInterval(() => {
      const now = performance.now();
      if (last.current != null) {
        const dt = (now - last.current) / 1000;
        setTime((t) => (t + dt >= duration ? duration : t + dt));
      }
      last.current = now;
    }, 1000 / 30);
    return () => window.clearInterval(id);
  }, [playing, duration]);

  const shot = useMemo(
    () => shots.find((s) => time >= s.start && time < s.end) ?? shots[shots.length - 1],
    [shots, time],
  );
  const pct = (v: number) => `${(v / duration) * 100}%`;

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-xl bg-[#0F2233] aspect-video">
        <Scene id={shot?.sceneId ?? video.scene} t={time} className="absolute inset-0 w-full h-full" />
        <div className="absolute left-3 top-3 flex items-center gap-2">
          <span className="chip h-6 border-0 bg-[#0F2233]/85 text-white text-[11px] font-mono font-medium">
            {formatTime(time)} / {formatTime(duration)}
          </span>
          {video.sample && (
            <span className="chip h-6 border-0 bg-white/85 text-ink-2 text-[11px]">Sample footage</span>
          )}
        </div>
        {shot?.onScreen && (
          <div className="absolute left-3 bottom-3 rounded-md bg-[#0F2233]/70 px-2 py-1 text-[11px] font-semibold text-white">
            {shot.onScreen}
          </div>
        )}
      </div>

      {/* timeline with shot boundaries and the cited span */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="w-9 h-9 shrink-0 rounded-full bg-ink text-white flex items-center justify-center hover:bg-[#1c3448]"
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
        </button>
        <div
          className="relative flex-1 h-8 cursor-pointer"
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            setTime(((e.clientX - r.left) / r.width) * duration);
          }}
        >
          <div className="absolute inset-x-0 top-3 h-2 rounded-full bg-[#E3EBF1]" />
          {shots.map((s) => (
            <div key={s.id} className="absolute top-2 h-4 w-px bg-[#B5C5D1]" style={{ left: pct(s.start) }} />
          ))}
          <div
            className="absolute top-3 h-2 rounded-full bg-verified/70"
            style={{ left: pct(start), width: pct((end ?? start + 20) - start) }}
          />
          <div className="absolute top-3 h-2 rounded-full bg-glacier" style={{ width: pct(time) }} />
          <div className="absolute top-1.5 w-5 h-5 -ml-2.5 rounded-full bg-white border-2 border-glacier shadow" style={{ left: pct(time) }} />
        </div>
        <button
          onClick={() => {
            setTime(start);
            setPlaying(true);
          }}
          className="chip h-8 border-[#A9DCC5] bg-verified-soft text-verified-ink hover:bg-[#d8f0e5]"
        >
          <RotateCcw className="w-3.5 h-3.5" /> {formatTime(start)}
        </button>
      </div>

      {shot && !compact && (
        <dl className="grid grid-cols-[92px_1fr] gap-y-2 text-[13px] rounded-xl border border-[#E3EBF1] bg-[#FAFCFD] p-3.5">
          <dt className="label-mono text-ink-3 !text-[10px] pt-0.5">Scene</dt>
          <dd className="text-ink">{shot.scene}</dd>
          <dt className="label-mono text-ink-3 !text-[10px] pt-0.5">Speech</dt>
          <dd className="text-ink">{shot.speech}</dd>
          <dt className="label-mono text-ink-3 !text-[10px] pt-0.5">On screen</dt>
          <dd className="text-ink">{shot.onScreen}</dd>
          <dt className="label-mono text-ink-3 !text-[10px] pt-1">Tags</dt>
          <dd className="flex flex-wrap gap-1.5">
            {shot.tags.map((t) => (
              <span key={t} className={cn('rounded-md px-2 py-0.5 text-[11.5px] font-semibold bg-glacier-soft text-[#06698A]')}>
                {t}
              </span>
            ))}
          </dd>
        </dl>
      )}
    </div>
  );
}
