// FIRN domain types shared by the demo data and the real API adapter.

export type SourceKind = 'report' | 'dataset' | 'photo' | 'video';

export type SceneId =
  | 'bharati'
  | 'maitri'
  | 'ice-core'
  | 'ship'
  | 'aurora'
  | 'weather-mast'
  | 'document';

export interface ArchiveItem {
  id: string;
  kind: SourceKind;
  title: string;
  org: string;
  year: string;
  /** Public link for real documents; undefined for sample media. */
  url?: string;
  pages?: number;
  /** Seconds, for videos. */
  duration?: number;
  language: string;
  tags: string[];
  summary: string;
  scene: SceneId;
  /** True when the item is sample content made for this prototype. */
  sample: boolean;
}

/** One searchable record per video shot, produced by the video pipeline. */
export interface ShotRecord {
  id: string;
  videoId: string;
  start: number;
  end: number;
  scene: string;
  speech: string;
  onScreen: string;
  tags: string[];
  sceneId: SceneId;
}

export type EvidenceRelation = 'supports' | 'contradicts' | 'context';

export interface Citation {
  n: number;
  sourceId: string;
  /** Page for documents, start/end seconds for video moments. */
  page?: number;
  section?: string;
  start?: number;
  end?: number;
  excerpt: string;
  relation: EvidenceRelation;
}

export interface PipelineStage {
  key: string;
  label: string;
  detail: string;
}

export interface Answer {
  id: string;
  question: string;
  lang: 'en' | 'hi';
  text: string;
  citations: Citation[];
}

export interface AskCallbacks {
  onStage?: (index: number) => void;
  onToken?: (partial: string) => void;
}
