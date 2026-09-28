// Stylised keyframes for sample media. `t` (seconds) drives subtle motion so the
// player feels alive; the drawing is deterministic for a given t.

import { useId } from 'react';
import type { SceneId } from '../types';

interface SceneProps {
  id: SceneId;
  t?: number;
  className?: string;
}

export function Scene({ id, t = 0, className }: SceneProps) {
  const uid = useId().replace(/:/g, '');
  const drift = (t * 6) % 360;
  return (
    <svg viewBox="0 0 320 180" preserveAspectRatio="xMidYMid slice" className={className} role="img" aria-label={`${id} scene`}>
      <defs>
        <linearGradient id={`sky-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#9CC9E0" />
          <stop offset="0.7" stopColor="#E3F1F8" />
          <stop offset="1" stopColor="#F6FBFD" />
        </linearGradient>
        <linearGradient id={`night-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#081A2B" />
          <stop offset="0.75" stopColor="#15324A" />
          <stop offset="1" stopColor="#23465F" />
        </linearGradient>
        <linearGradient id={`sea-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#3F7896" />
          <stop offset="1" stopColor="#2A5A75" />
        </linearGradient>
        <linearGradient id={`snow-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#FFFFFF" />
          <stop offset="1" stopColor="#DCEAF2" />
        </linearGradient>
        <filter id={`blur-${uid}`}><feGaussianBlur stdDeviation="5" /></filter>
      </defs>
      {id === 'aurora' ? (
        <Aurora uid={uid} t={t} />
      ) : id === 'ship' ? (
        <Ship uid={uid} t={t} drift={drift} />
      ) : id === 'ice-core' ? (
        <IceCore uid={uid} drift={drift} />
      ) : id === 'weather-mast' ? (
        <WeatherMast uid={uid} t={t} drift={drift} />
      ) : id === 'maitri' ? (
        <Maitri uid={uid} drift={drift} />
      ) : id === 'document' ? (
        <Doc />
      ) : (
        <Bharati uid={uid} drift={drift} />
      )}
    </svg>
  );
}

function Clouds({ drift, opacity = 0.8 }: { drift: number; opacity?: number }) {
  const x = (drift % 400) - 60;
  return (
    <g fill="#FFFFFF" opacity={opacity}>
      <ellipse cx={x} cy="30" rx="34" ry="7" />
      <ellipse cx={x + 22} cy="26" rx="20" ry="7" />
      <ellipse cx={((x + 210) % 400) - 40} cy="44" rx="28" ry="5" />
    </g>
  );
}

function Bharati({ uid, drift }: { uid: string; drift: number }) {
  return (
    <g>
      <rect width="320" height="180" fill={`url(#sky-${uid})`} />
      <Clouds drift={drift} />
      <path d="M0 102 L60 88 L120 96 L190 84 L260 94 L320 86 V120 H0Z" fill="#EEF6FA" />
      <path d="M0 120 H320 V130 H0Z" fill={`url(#sea-${uid})`} opacity="0.85" />
      <path d="M0 128 L40 116 L78 122 L120 110 L170 118 L220 108 L270 118 L320 112 V180 H0Z" fill="#7E8C97" />
      <path d="M0 146 L60 136 L130 142 L200 134 L260 140 L320 136 V180 H0Z" fill={`url(#snow-${uid})`} />
      {/* modular station on stilts */}
      <g transform="translate(150 78)">
        <rect x="-6" y="40" width="3" height="16" fill="#3C4A55" />
        <rect x="30" y="40" width="3" height="16" fill="#3C4A55" />
        <rect x="70" y="40" width="3" height="16" fill="#3C4A55" />
        <rect x="104" y="40" width="3" height="16" fill="#3C4A55" />
        <rect x="-12" y="8" width="126" height="34" rx="2" fill="#9AA8B3" />
        <rect x="-12" y="8" width="126" height="6" fill="#B9C5CE" />
        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => (
          <rect key={i} x={-6 + i * 12} y="20" width="7" height="10" rx="1" fill="#2F4C60" />
        ))}
        <rect x="40" y="-4" width="40" height="14" rx="2" fill="#8795A1" />
        <path d="M100 8 V-22" stroke="#3C4A55" strokeWidth="1.6" />
        <circle cx="100" cy="-24" r="3" fill="#EE6A1F" />
      </g>
    </g>
  );
}

function Maitri({ uid, drift }: { uid: string; drift: number }) {
  return (
    <g>
      <rect width="320" height="180" fill={`url(#sky-${uid})`} />
      <Clouds drift={drift} opacity={0.7} />
      <path d="M0 96 L80 84 L160 92 L240 80 L320 90 V130 H0Z" fill="#EEF6FA" />
      <path d="M0 120 L50 110 L110 116 L170 106 L240 114 L320 108 V180 H0Z" fill="#8C8279" />
      <ellipse cx="80" cy="140" rx="46" ry="7" fill="#5C8FAA" opacity="0.8" />
      <path d="M0 152 L80 146 L170 150 L250 144 L320 148 V180 H0Z" fill="#E4EEF4" />
      {[0, 1, 2].map((i) => (
        <g key={i} transform={`translate(${150 + i * 48} ${104 - (i % 2) * 4})`}>
          <rect width="40" height="22" rx="2" fill={i === 1 ? '#C9563A' : '#B8C4CD'} />
          <path d="M-3 0 L20 -10 L43 0Z" fill="#6B7A86" />
          <rect x="6" y="7" width="7" height="7" fill="#2F4C60" />
          <rect x="24" y="7" width="7" height="7" fill="#2F4C60" />
        </g>
      ))}
    </g>
  );
}

function IceCore({ uid, drift }: { uid: string; drift: number }) {
  return (
    <g>
      <rect width="320" height="180" fill={`url(#sky-${uid})`} />
      <Clouds drift={drift} />
      <path d="M0 110 Q160 100 320 110 V180 H0Z" fill={`url(#snow-${uid})`} />
      <path d="M0 110 Q160 100 320 110" stroke="#C9DDE8" fill="none" />
      {/* tent */}
      <path d="M36 118 L62 88 L88 118Z" fill="#F2A33A" />
      <path d="M62 88 L88 118 L72 118Z" fill="#D9861F" />
      {/* drill tower */}
      <g stroke="#3C4A55" strokeWidth="2.4" strokeLinecap="round">
        <path d="M160 128 L186 56 L212 128" fill="none" />
        <path d="M168 104 H204" />
      </g>
      <path d="M186 56 V132" stroke="#5B6B77" strokeWidth="1.4" />
      <rect x="181" y="120" width="10" height="16" rx="2" fill="#9CC7DC" stroke="#5E9CB8" />
      {/* person in expedition orange */}
      <g transform="translate(232 104)">
        <circle cx="0" cy="-2" r="5" fill="#F3D2B8" />
        <path d="M-6 -6 A6 6 0 0 1 6 -6 V-3 H-6Z" fill="#0F2233" />
        <rect x="-8" y="3" width="16" height="20" rx="4" fill="#EE6A1F" />
        <rect x="-7" y="22" width="6" height="12" fill="#2F3B45" />
        <rect x="1" y="22" width="6" height="12" fill="#2F3B45" />
      </g>
      {/* core on the logging stand */}
      <rect x="92" y="134" width="56" height="9" rx="4.5" fill="#CFE6F1" stroke="#8FBFD6" />
      <path d="M104 134 V143 M118 134 V143 M131 134 V143" stroke="#8FBFD6" />
    </g>
  );
}

function Ship({ uid, t, drift }: { uid: string; t: number; drift: number }) {
  const bob = Math.sin(t * 1.4) * 1.2;
  return (
    <g>
      <rect width="320" height="180" fill={`url(#sky-${uid})`} />
      <Clouds drift={drift} opacity={0.6} />
      <rect y="96" width="320" height="84" fill={`url(#sea-${uid})`} />
      {[
        [20, 120, 38], [80, 150, 30], [250, 128, 44], [180, 160, 26], [290, 156, 22], [120, 108, 18],
      ].map(([x, y, w], i) => (
        <path key={i} d={`M${x} ${y} l${w * 0.3} -5 l${w * 0.7} 2 l-${w * 0.2} 6 l-${w * 0.8} 1z`} fill="#F4FAFD" stroke="#CFE3EE" />
      ))}
      <g transform={`translate(96 ${76 + bob})`}>
        <path d="M0 30 H132 L120 48 H10Z" fill="#D9531E" />
        <rect x="0" y="27" width="132" height="4" fill="#F4FAFD" />
        <rect x="70" y="4" width="40" height="24" fill="#F4FAFD" />
        <rect x="76" y="10" width="28" height="5" fill="#2F4C60" />
        <rect x="84" y="-10" width="8" height="15" fill="#EE6A1F" />
        <path d="M30 28 V2 M30 6 L56 26" stroke="#3C4A55" strokeWidth="1.5" />
      </g>
    </g>
  );
}

function Aurora({ uid, t }: { uid: string; t: number }) {
  const w = Math.sin(t * 0.6) * 10;
  return (
    <g>
      <rect width="320" height="180" fill={`url(#night-${uid})`} />
      {[[20, 20], [60, 42], [110, 14], [170, 34], [230, 18], [290, 40], [260, 60], [40, 70]].map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="0.9" fill="#FFFFFF" opacity="0.8" />
      ))}
      <g filter={`url(#blur-${uid})`} opacity="0.9">
        <path d={`M-10 90 C60 ${40 + w} 120 ${70 - w} 190 40 S300 ${30 + w} 340 60`} stroke="#3BE3A0" strokeWidth="16" fill="none" />
        <path d={`M-10 100 C80 ${60 - w} 150 ${86 + w} 220 58 S300 ${48 - w} 340 80`} stroke="#6E6BF0" strokeWidth="10" fill="none" opacity="0.7" />
      </g>
      <path d="M0 140 L70 132 L150 138 L230 128 L320 136 V180 H0Z" fill="#1E3A50" />
      <path d="M0 156 L100 150 L200 154 L320 148 V180 H0Z" fill="#2B4B63" />
      <g transform="translate(176 118)">
        <rect width="70" height="22" fill="#0B1826" />
        {[0, 1, 2, 3, 4].map((i) => <rect key={i} x={6 + i * 13} y="8" width="6" height="6" fill="#F2C14E" />)}
      </g>
    </g>
  );
}

function WeatherMast({ uid, t, drift }: { uid: string; t: number; drift: number }) {
  const spin = (t * 240) % 360;
  return (
    <g>
      <rect width="320" height="180" fill={`url(#sky-${uid})`} />
      <Clouds drift={drift} />
      <path d="M0 118 Q160 108 320 118 V180 H0Z" fill={`url(#snow-${uid})`} />
      <path d="M170 136 V36" stroke="#3C4A55" strokeWidth="2.4" />
      <path d="M170 60 L150 136 M170 60 L190 136" stroke="#8595A1" strokeWidth="1" />
      <g transform={`translate(170 36) rotate(${spin})`}>
        <path d="M0 0 H14 M0 0 L-7 12 M0 0 L-7 -12" stroke="#3C4A55" strokeWidth="1.5" />
        <circle cx="14" cy="0" r="3" fill="#0B8DB5" /><circle cx="-7" cy="12" r="3" fill="#0B8DB5" /><circle cx="-7" cy="-12" r="3" fill="#0B8DB5" />
      </g>
      <rect x="176" y="80" width="30" height="18" rx="1" fill="#2F4C60" transform="rotate(-18 176 80)" />
      <rect x="158" y="104" width="24" height="18" rx="2" fill="#E9EEF2" stroke="#9AA8B3" />
      <g transform="translate(120 112)">
        <circle cx="0" cy="-2" r="5" fill="#F3D2B8" />
        <rect x="-8" y="3" width="16" height="20" rx="4" fill="#0B8DB5" />
        <rect x="-7" y="22" width="6" height="12" fill="#2F3B45" />
        <rect x="1" y="22" width="6" height="12" fill="#2F3B45" />
      </g>
    </g>
  );
}

function Doc() {
  return (
    <g>
      <rect width="320" height="180" fill="#EEF4F8" />
      <g transform="translate(118 22) rotate(-4)">
        <rect width="92" height="124" rx="6" fill="#FFFFFF" stroke="#D6E2EB" />
        <rect x="12" y="14" width="46" height="7" rx="3.5" fill="#0B8DB5" />
        {[32, 42, 52, 62, 72, 82, 92, 102].map((y, i) => (
          <rect key={y} x="12" y={y} width={i % 3 === 2 ? 44 : 68} height="4" rx="2" fill="#C9D6DF" />
        ))}
      </g>
    </g>
  );
}
