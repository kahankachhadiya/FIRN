// Sample archive for the FIRN prototype.
// Documents point to real public NCPOR / SCAR / Government of India pages; their
// excerpts restate facts from those pages. Videos, photos and datasets are
// sample content made for this prototype and are marked `sample: true`.

import type { Answer, ArchiveItem, PipelineStage, ShotRecord } from '../types';

export const ARCHIVE: ArchiveItem[] = [
  {
    id: 'doc-isea45',
    kind: 'report',
    title: "India's Antarctic Research Season 2025–26 Commences",
    org: 'NCPOR',
    year: '2025',
    url: 'https://ncpor.res.in/news/view/929',
    language: 'en',
    tags: ['45th ISEA', 'Maitri', 'expedition'],
    summary: 'Season launch note for the 45th Indian Scientific Expedition to Antarctica.',
    scene: 'document',
    sample: false,
  },
  {
    id: 'doc-scar-india',
    kind: 'report',
    title: 'Featured Member Country: India',
    org: 'SCAR',
    year: '2017',
    url: 'https://scar.org/scar-news/india-feature',
    language: 'en',
    tags: ['Indian Antarctic Programme', 'Maitri', 'Bharati', 'history'],
    summary: "Overview of India's Antarctic programme, stations and SCAR membership.",
    scene: 'document',
    sample: false,
  },
  {
    id: 'doc-antarctic-act',
    kind: 'report',
    title: 'The Indian Antarctic Act, 2022',
    org: 'Government of India',
    year: '2022',
    url: 'https://prsindia.org/files/bills_acts/acts_parliament/2022/The%20Indian%20Antarctic%20Act,%202022.pdf',
    language: 'en',
    tags: ['law', 'records', 'annual report'],
    summary: 'National law governing Indian activity in Antarctica.',
    scene: 'document',
    sample: false,
  },
  {
    id: 'doc-sci-reports',
    kind: 'report',
    title: 'Scientific Reports of Indian Expeditions to Antarctica',
    org: 'NCPOR',
    year: '—',
    url: 'https://ncpor.res.in/news/view/425',
    language: 'en',
    tags: ['archive', 'expedition reports'],
    summary: 'Notice on the online archive of expedition scientific reports.',
    scene: 'document',
    sample: false,
  },
  {
    id: 'vid-bharati',
    kind: 'video',
    title: 'Life at Bharati Station',
    org: 'Sample footage',
    year: '2025',
    duration: 400,
    language: 'en · hi',
    tags: ['Bharati', 'station life', 'Larsemann Hills'],
    summary: 'A walk through daily life and science at Bharati station.',
    scene: 'bharati',
    sample: true,
  },
  {
    id: 'vid-icecore',
    kind: 'video',
    title: 'Ice-core Drilling in the Field',
    org: 'Sample footage',
    year: '2025',
    duration: 235,
    language: 'en',
    tags: ['ice core', 'glaciology', 'field season'],
    summary: 'A field team drilling and logging a shallow ice core.',
    scene: 'ice-core',
    sample: true,
  },
  {
    id: 'vid-voyage',
    kind: 'video',
    title: 'Voyage Through the Southern Ocean',
    org: 'Sample footage',
    year: '2024',
    duration: 312,
    language: 'en',
    tags: ['ship', 'Southern Ocean', 'sea ice'],
    summary: 'The expedition vessel crossing into sea ice.',
    scene: 'ship',
    sample: true,
  },
  {
    id: 'ph-aurora',
    kind: 'photo',
    title: 'Aurora over Maitri',
    org: 'Sample photo',
    year: '2024',
    language: '—',
    tags: ['aurora', 'Maitri', 'night sky'],
    summary: 'Aurora australis over the station during polar night.',
    scene: 'aurora',
    sample: true,
  },
  {
    id: 'ph-maitri',
    kind: 'photo',
    title: 'Maitri Station, Schirmacher Oasis',
    org: 'Sample photo',
    year: '2023',
    language: '—',
    tags: ['Maitri', 'station'],
    summary: 'Maitri station buildings in summer.',
    scene: 'maitri',
    sample: true,
  },
  {
    id: 'ds-met',
    kind: 'dataset',
    title: 'Surface Meteorology, Maitri (sample extract)',
    org: 'Sample dataset',
    year: '2024',
    language: '—',
    tags: ['temperature', 'wind', 'NetCDF'],
    summary: 'Hourly air temperature, pressure and wind from the station mast.',
    scene: 'weather-mast',
    sample: true,
  },
];

export const SHOTS: ShotRecord[] = [
  { id: 's1', videoId: 'vid-bharati', start: 0, end: 38, sceneId: 'ship', scene: 'Approach by sea through scattered ice', speech: '“…the station comes into view after days at sea…”', onScreen: 'Southern Ocean', tags: ['ship', 'sea ice'] },
  { id: 's2', videoId: 'vid-bharati', start: 38, end: 104, sceneId: 'bharati', scene: 'Exterior of Bharati station on the rocky coast', speech: '“…built from modular container units…”', onScreen: 'Bharati · Larsemann Hills', tags: ['Bharati', 'architecture'] },
  { id: 's3', videoId: 'vid-bharati', start: 104, end: 190, sceneId: 'weather-mast', scene: 'Team checking the automatic weather mast', speech: '“…readings are logged every hour…”', onScreen: 'Met observations', tags: ['weather', 'instruments'] },
  { id: 's4', videoId: 'vid-bharati', start: 252, end: 271, sceneId: 'bharati', scene: 'Station exterior at dusk, crew returning', speech: '“…Bharati has been home to our teams since 2012…”', onScreen: 'Bharati station', tags: ['Bharati', 'station life'] },
  { id: 's5', videoId: 'vid-bharati', start: 271, end: 400, sceneId: 'aurora', scene: 'Night sky with aurora above the station', speech: '“…winter brings the aurora…”', onScreen: 'Polar night', tags: ['aurora', 'winter'] },
  { id: 's6', videoId: 'vid-icecore', start: 0, end: 61, sceneId: 'ice-core', scene: 'Setting up the drill on the ice sheet', speech: '“…first we level the drill…”', onScreen: 'Field camp', tags: ['ice core', 'drill'] },
  { id: 's7', videoId: 'vid-icecore', start: 61, end: 142, sceneId: 'ice-core', scene: 'Ice-core drilling beside the camp', speech: '“…now we lower the drill again…”', onScreen: 'Station name · field season', tags: ['ice core', 'glaciology', 'Antarctica'] },
  { id: 's8', videoId: 'vid-icecore', start: 142, end: 235, sceneId: 'ice-core', scene: 'Core sections logged and packed', speech: '“…each section is labelled by depth…”', onScreen: 'Core log', tags: ['ice core', 'samples'] },
];

/** The retrieval stages the Ask screen shows while an answer is being built. */
export const ASK_STAGES: PipelineStage[] = [
  { key: 'search', label: 'Searching', detail: 'Vector + keyword search' },
  { key: 'graph', label: 'Connecting', detail: 'Evidence graph' },
  { key: 'rerank', label: 'Ranking', detail: 'Reranker picks the best passages' },
  { key: 'write', label: 'Writing', detail: 'Cited answer' },
];

interface ScriptedAnswer extends Answer {
  /** Lower-case keywords that route a typed question to this answer. */
  match: string[];
}

export const ANSWERS: ScriptedAnswer[] = [
  {
    id: 'a-bharati',
    lang: 'hi',
    question: 'भारती स्टेशन कब शुरू हुआ?',
    match: ['भारती', 'bharati'],
    text:
      'भारती, अंटार्कटिका में भारत का तीसरा अनुसंधान स्टेशन है और यह **2012** में शुरू हुआ [1]। इससे पहले दक्षिण गंगोत्री (1983–84) और मैत्री (1989) स्थापित किए गए थे [1]।\n\nस्टेशन पर जीवन की झलक इस वीडियो में **04:12** पर देखी जा सकती है [2]।',
    citations: [
      { n: 1, sourceId: 'doc-scar-india', excerpt: 'India operates Maitri (1989) and Bharati (commissioned 2012); Dakshin Gangotri, its first station (1983–84), now serves as a supply base.', relation: 'supports' },
      { n: 2, sourceId: 'vid-bharati', start: 252, end: 271, excerpt: 'Station exterior at dusk, crew returning. “…Bharati has been home to our teams since 2012…”', relation: 'supports' },
    ],
  },
  {
    id: 'a-isea45',
    lang: 'en',
    question: 'When did the 45th expedition reach Maitri?',
    match: ['45', 'expedition', 'isea', 'maitri'],
    text:
      'The **45th Indian Scientific Expedition to Antarctica** reached Maitri station on **4 November 2025**, opening India\'s 2025–26 research season [1].\n\nIndia\'s programme began in 1981, and the first expedition reached Antarctica on 8 January 1982 [2].',
    citations: [
      { n: 1, sourceId: 'doc-isea45', excerpt: 'The 45th ISEA team reached Maitri on 4 November 2025, marking the start of the 2025–26 season.', relation: 'supports' },
      { n: 2, sourceId: 'doc-scar-india', excerpt: 'The Indian Antarctic Programme began in 1981; the first expedition reached Antarctica on 8 January 1982.', relation: 'context' },
    ],
  },
  {
    id: 'a-act',
    lang: 'en',
    question: 'What records does the Indian Antarctic Act require?',
    match: ['act', 'law', 'record', 'report'],
    text:
      'The Act asks the Committee on Antarctic Governance and Environmental Protection to **maintain records of the programmes and activities** carried out in Antarctica [1], and to submit an **annual report** giving a full account of its activities [2].\n\nFIRN can keep both up to date: every report, dataset and video is already indexed and dated.',
    citations: [
      { n: 1, sourceId: 'doc-antarctic-act', section: 's.25(d)', excerpt: 'The Committee shall maintain records pertaining to the programmes and activities conducted in Antarctica.', relation: 'supports' },
      { n: 2, sourceId: 'doc-antarctic-act', section: 's.53(2)', excerpt: 'The Committee shall prepare an annual report giving a true and full account of its activities.', relation: 'supports' },
    ],
  },
  {
    id: 'a-icecore',
    lang: 'en',
    question: 'Show me footage of ice-core drilling',
    match: ['ice', 'core', 'drill', 'footage'],
    text:
      'The clearest moment is at **01:01** in *Ice-core Drilling in the Field*, where the team lowers the drill again beside the camp [1]. The next shot shows each core section being labelled by depth [2].',
    citations: [
      { n: 1, sourceId: 'vid-icecore', start: 61, end: 142, excerpt: 'Ice-core drilling beside the camp. “…now we lower the drill again…”', relation: 'supports' },
      { n: 2, sourceId: 'vid-icecore', start: 142, end: 235, excerpt: 'Core sections logged and packed. “…each section is labelled by depth…”', relation: 'context' },
    ],
  },
];

export const SUGGESTIONS = ANSWERS.map((a) => ({ id: a.id, question: a.question, lang: a.lang }));

export function findItem(id: string): ArchiveItem | undefined {
  return ARCHIVE.find((i) => i.id === id);
}

export function shotsFor(videoId: string): ShotRecord[] {
  return SHOTS.filter((s) => s.videoId === videoId);
}

export function matchAnswer(question: string): ScriptedAnswer | undefined {
  const q = question.trim().toLowerCase();
  const exact = ANSWERS.find((a) => a.question.toLowerCase() === q);
  if (exact) return exact;
  let best: ScriptedAnswer | undefined;
  let bestScore = 0;
  for (const a of ANSWERS) {
    const score = a.match.filter((m) => q.includes(m)).length;
    if (score > bestScore) {
      best = a;
      bestScore = score;
    }
  }
  return best;
}
