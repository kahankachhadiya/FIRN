import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { MessageSquareText, Library, UploadCloud, Megaphone, GraduationCap } from 'lucide-react';
import { cn } from '@/lib/utils';
import { DEMO_MODE } from '../api';

const NAV = [
  { to: '/', label: 'Ask', icon: MessageSquareText, end: true },
  { to: '/archive', label: 'Archive', icon: Library },
  { to: '/ingest', label: 'Ingest', icon: UploadCloud },
  { to: '/studio', label: 'Outreach Studio', icon: Megaphone },
  { to: '/learn', label: 'Learn', icon: GraduationCap },
];

export function FirnMark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <defs>
        <linearGradient id="firn-core" x1="0" x2="1">
          <stop offset="0" stopColor="#9CCFE4" />
          <stop offset="0.5" stopColor="#F2FAFD" />
          <stop offset="1" stopColor="#8CC3DA" />
        </linearGradient>
      </defs>
      <rect x="8" y="3" width="16" height="26" rx="8" fill="url(#firn-core)" stroke="#0B8DB5" strokeWidth="1.4" />
      <path d="M8.8 11.5 Q16 14 23.2 11.5 M8.6 17 Q16 19.5 23.4 17 M8.8 22.5 Q16 25 23.2 22.5" stroke="#0B8DB5" strokeWidth="1.2" fill="none" opacity="0.7" />
    </svg>
  );
}

function Sidebar() {
  return (
    <aside className="w-[232px] shrink-0 h-full flex flex-col bg-white/80 backdrop-blur border-r border-[#DFE8EE]">
      <div className="h-[68px] px-5 flex items-center gap-2.5">
        <FirnMark />
        <div className="leading-tight">
          <div className="text-[19px] font-extrabold tracking-tight text-ink">FIRN</div>
          <div className="text-[11.5px] font-medium text-ink-3">India's polar memory</div>
        </div>
      </div>

      <nav className="px-3 pt-2 flex flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 h-10 px-3 rounded-xl text-[14px] font-semibold transition-colors',
                isActive ? 'bg-glacier-soft text-[#06698A]' : 'text-ink-2 hover:bg-[#F1F6F9]',
              )
            }
          >
            <Icon className="w-[18px] h-[18px]" strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto p-3">
        <div className="rounded-xl border border-[#E3EBF1] bg-[#F8FBFC] p-3.5">
          <div className="label-mono text-ink-3 !text-[10px]">Stack</div>
          <ul className="mt-2.5 space-y-2 text-[12.5px] text-ink-2">
            <Status color="bg-glacier" label="Search" value="Qdrant · ES" />
            <Status color="bg-aurora" label="Graph" value="FalkorDB" />
            <Status color="bg-aurora" label="Vision" value="Qwen3-VL" />
            <Status color="bg-glacier" label="Speech" value="Whisper" />
          </ul>
        </div>
        <div className="mt-3 px-1 text-[11.5px] text-ink-3">NCPOR · MoES · Team Aizen</div>
      </div>
    </aside>
  );
}

function Status({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className={cn('w-2 h-2 rounded-full', color)} />
      <span className="font-semibold text-ink">{label}</span>
      <span className="ml-auto text-ink-3">{value}</span>
    </li>
  );
}

interface AppShellProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}

export function AppShell({ title, subtitle, actions, children }: AppShellProps) {
  return (
    <div className="h-screen flex overflow-hidden">
      <Sidebar />
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-[68px] shrink-0 px-7 flex items-center gap-4 border-b border-[#DFE8EE] bg-white/50 backdrop-blur">
          <div className="min-w-0">
            <h1 className="text-[18px] font-extrabold tracking-tight text-ink">{title}</h1>
            {subtitle && <p className="text-[12.5px] text-ink-3 truncate">{subtitle}</p>}
          </div>
          <div className="ml-auto flex items-center gap-2.5">
            {actions}
            {DEMO_MODE && (
              <span className="chip border-[#F7C4A3] bg-alert-soft text-[#B8521A]">
                <span className="w-1.5 h-1.5 rounded-full bg-alert" />
                Prototype · sample data
              </span>
            )}
          </div>
        </header>
        <main className="flex-1 min-h-0">{children}</main>
      </div>
    </div>
  );
}
