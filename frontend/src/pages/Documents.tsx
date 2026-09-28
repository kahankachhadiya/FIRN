import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  Upload, FileText, FileImage, FileAudio, FileSpreadsheet,
  Presentation, Trash2, CheckCircle, Loader2, AlertCircle,
  ArrowLeft, RefreshCw, ChevronDown, ChevronUp, Database,
} from 'lucide-react';
import { useConfigStore } from '@/stores/configStore';

// ─── Types ────────────────────────────────────────────────────────────────────

interface UploadedFile {
  localName: string;
  serverPath: string;
}

interface StageInfo {
  name: string;
  display_name: string;
  model: string | null;
  status: string;
  progress: number;
  duration_seconds: number | null;
}

interface FileResult {
  filename: string;
  success: boolean;
  processing_time: number;
  error: string | null;
  stages_completed: string[];
}

interface Job {
  job_id: string;
  status: string;
  total_files: number;
  processed_files: number;
  current_file: string | null;
  current_stage: string | null;
  stages: StageInfo[];
  results: FileResult[];
  created_at: string;
  updated_at: string;
  error: string | null;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const SUPPORTED = '.pdf,.docx,.pptx,.xlsx,.mp3,.wav,.jpg,.png';

function fileIcon(name: string) {
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext)) return <FileImage className="w-4 h-4 text-blue-400" />;
  if (['mp3', 'wav'].includes(ext)) return <FileAudio className="w-4 h-4 text-purple-400" />;
  if (['xlsx', 'csv'].includes(ext)) return <FileSpreadsheet className="w-4 h-4 text-green-400" />;
  if (['pptx', 'ppt'].includes(ext)) return <Presentation className="w-4 h-4 text-orange-400" />;
  return <FileText className="w-4 h-4 text-muted-foreground" />;
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function statusColor(s: string) {
  if (s === 'completed') return 'text-primary';
  if (s === 'failed' || s === 'error') return 'text-destructive';
  if (s === 'running' || s === 'processing') return 'text-yellow-400';
  return 'text-muted-foreground';
}

function statusIcon(s: string) {
  if (s === 'completed') return <CheckCircle className="w-4 h-4 text-primary" />;
  if (s === 'failed' || s === 'error') return <AlertCircle className="w-4 h-4 text-destructive" />;
  return <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Documents() {
  const navigate = useNavigate();
  const backendUrl = useConfigStore((s) => s.config.ingestion_url ?? s.config.backend_url);

  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [jobs, setJobs] = useState<Job[]>([]);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Polling ────────────────────────────────────────────────────────────────

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/api/processing/jobs`);
      if (!res.ok) return;
      const data: Job[] = await res.json();
      setJobs(data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
    } catch { /* silent */ }
  }, [backendUrl]);

  useEffect(() => {
    fetchJobs();
    pollRef.current = setInterval(fetchJobs, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [fetchJobs]);

  // ── File selection ─────────────────────────────────────────────────────────

  const addFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const supported = SUPPORTED.split(',').map(e => e.replace('.', ''));
    const valid = Array.from(fileList).filter(f => {
      const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
      return supported.includes(ext);
    });
    setPendingFiles(prev => {
      const names = new Set(prev.map(f => f.name));
      return [...prev, ...valid.filter(f => !names.has(f.name))];
    });
  };

  const removeFile = (name: string) => setPendingFiles(prev => prev.filter(f => f.name !== name));

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    addFiles(e.dataTransfer.files);
  }, []);

  // ── Upload + start ─────────────────────────────────────────────────────────

  const handleIngest = async () => {
    if (!pendingFiles.length) return;
    setUploading(true);
    setUploadError(null);

    try {
      // 1. Upload files
      const formData = new FormData();
      pendingFiles.forEach(f => formData.append('files', f));

      const uploadRes = await fetch(`${backendUrl}/api/processing/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status}`);
      const uploadData: { uploaded_files: string[]; rejected_files: { filename: string; reason: string }[] } =
        await uploadRes.json();

      if (uploadData.rejected_files.length) {
        const reasons = uploadData.rejected_files.map(r => `${r.filename}: ${r.reason}`).join('\n');
        setUploadError(`Some files were rejected:\n${reasons}`);
      }

      if (!uploadData.uploaded_files.length) {
        setUploading(false);
        return;
      }

      // 2. Start processing
      const startRes = await fetch(`${backendUrl}/api/processing/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_paths: uploadData.uploaded_files, auto_inject: true }),
      });
      if (!startRes.ok) throw new Error(`Start failed: ${startRes.status}`);

      setPendingFiles([]);
      await fetchJobs();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setUploading(false);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  const activeJobs = jobs.filter(j => j.status === 'running' || j.status === 'processing' || j.status === 'pending');
  const doneJobs = jobs.filter(j => j.status !== 'running' && j.status !== 'processing' && j.status !== 'pending');

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="h-14 flex items-center gap-3 px-5 border-b border-border/50 surface-1">
        <button
          onClick={() => navigate('/')}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-surface-2 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-primary" />
          <h1 className="text-[15px] font-medium text-foreground">Document Ingestion</h1>
        </div>
        <button
          onClick={fetchJobs}
          className="ml-auto w-8 h-8 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-surface-2 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </header>

      <div className="flex-1 max-w-3xl w-full mx-auto px-5 py-8 space-y-8">

        {/* ── Upload zone ── */}
        <section className="space-y-4">
          <h2 className="text-[13px] font-semibold text-foreground uppercase tracking-wider">Add Documents</h2>

          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            className={`relative border-2 border-dashed rounded-xl p-10 text-center transition-all ${
              isDragging ? 'border-primary bg-primary/5' : 'border-border/50 hover:border-border'
            }`}
          >
            <input
              type="file"
              multiple
              accept={SUPPORTED}
              onChange={(e) => addFiles(e.target.files)}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Upload className={`w-8 h-8 mx-auto mb-3 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
            <p className="text-[14px] font-medium text-foreground">Drop files or click to browse</p>
            <p className="text-[12px] text-muted-foreground mt-1">PDF · DOCX · PPTX · XLSX · MP3 · WAV · JPG · PNG</p>
          </div>

          {/* Pending file list */}
          <AnimatePresence mode="popLayout">
            {pendingFiles.map((file) => (
              <motion.div
                key={file.name}
                layout
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center gap-3 p-3 rounded-lg surface-2 border border-border/40"
              >
                <div className="w-8 h-8 rounded-md bg-surface-3 flex items-center justify-center flex-shrink-0">
                  {fileIcon(file.name)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[13px] font-medium text-foreground truncate">{file.name}</p>
                  <p className="text-[11px] text-muted-foreground">{formatSize(file.size)}</p>
                </div>
                <button
                  onClick={() => removeFile(file.name)}
                  className="w-7 h-7 rounded-md hover:bg-destructive/10 flex items-center justify-center transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5 text-destructive/70" />
                </button>
              </motion.div>
            ))}
          </AnimatePresence>

          {uploadError && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-[12px] text-destructive whitespace-pre-wrap">
              {uploadError}
            </div>
          )}

          {pendingFiles.length > 0 && (
            <button
              onClick={handleIngest}
              disabled={uploading}
              className="w-full py-2.5 rounded-lg bg-primary text-primary-foreground text-[13px] font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
              {uploading ? (
                <><Loader2 className="w-4 h-4 animate-spin" /> Uploading & starting pipeline…</>
              ) : (
                <><Database className="w-4 h-4" /> Ingest {pendingFiles.length} file{pendingFiles.length > 1 ? 's' : ''}</>
              )}
            </button>
          )}
        </section>

        {/* ── Active jobs ── */}
        {activeJobs.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-[13px] font-semibold text-foreground uppercase tracking-wider">Processing</h2>
            {activeJobs.map(job => <JobCard key={job.job_id} job={job} expanded={expandedJob === job.job_id} onToggle={() => setExpandedJob(p => p === job.job_id ? null : job.job_id)} />)}
          </section>
        )}

        {/* ── Completed jobs ── */}
        {doneJobs.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-[13px] font-semibold text-foreground uppercase tracking-wider">History</h2>
            {doneJobs.map(job => <JobCard key={job.job_id} job={job} expanded={expandedJob === job.job_id} onToggle={() => setExpandedJob(p => p === job.job_id ? null : job.job_id)} />)}
          </section>
        )}

        {jobs.length === 0 && pendingFiles.length === 0 && (
          <div className="text-center py-16 text-muted-foreground">
            <Database className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p className="text-[13px]">No documents ingested yet</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── JobCard ──────────────────────────────────────────────────────────────────

function JobCard({ job, expanded, onToggle }: { job: Job; expanded: boolean; onToggle: () => void }) {
  const isActive = job.status === 'running' || job.status === 'processing' || job.status === 'pending';
  const progress = job.total_files > 0 ? (job.processed_files / job.total_files) * 100 : 0;

  return (
    <div className="rounded-xl surface-2 border border-border/40 overflow-hidden">
      {/* Summary row */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 p-3.5 hover:bg-surface-3/50 transition-colors text-left"
      >
        <div className="flex-shrink-0">{statusIcon(job.status)}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-[12px] font-semibold uppercase tracking-wide ${statusColor(job.status)}`}>
              {job.status}
            </span>
            <span className="text-[11px] text-muted-foreground">
              {job.processed_files}/{job.total_files} files
            </span>
            {job.current_file && (
              <span className="text-[11px] text-muted-foreground truncate max-w-[160px]">
                · {job.current_file.split('/').pop()}
              </span>
            )}
          </div>
          {isActive && (
            <div className="mt-1.5 h-1 rounded-full bg-surface-3 overflow-hidden">
              <motion.div
                className="h-full bg-primary rounded-full"
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>
          )}
        </div>
        <span className="text-[11px] text-muted-foreground flex-shrink-0">
          {new Date(job.created_at).toLocaleTimeString()}
        </span>
        {expanded ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />}
      </button>

      {/* Expanded detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="border-t border-border/40 p-3.5 space-y-3">
              {/* Stages */}
              {job.stages.length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">Pipeline Stages</p>
                  {job.stages.map(stage => (
                    <div key={stage.name} className="flex items-center gap-2.5">
                      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{
                        backgroundColor: stage.status === 'completed' ? 'hsl(var(--primary))' :
                          stage.status === 'running' ? '#facc15' :
                          stage.status === 'failed' ? 'hsl(var(--destructive))' : 'hsl(var(--muted-foreground))'
                      }} />
                      <span className="text-[12px] text-foreground flex-1">{stage.display_name}</span>
                      {stage.duration_seconds != null && (
                        <span className="text-[11px] text-muted-foreground">{stage.duration_seconds.toFixed(1)}s</span>
                      )}
                      <span className={`text-[11px] ${statusColor(stage.status)}`}>{stage.status}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* File results */}
              {job.results.length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">Files</p>
                  {job.results.map(r => (
                    <div key={r.filename} className="flex items-center gap-2.5">
                      {r.success
                        ? <CheckCircle className="w-3.5 h-3.5 text-primary flex-shrink-0" />
                        : <AlertCircle className="w-3.5 h-3.5 text-destructive flex-shrink-0" />}
                      <span className="text-[12px] text-foreground flex-1 truncate">{r.filename}</span>
                      <span className="text-[11px] text-muted-foreground">{r.processing_time.toFixed(1)}s</span>
                    </div>
                  ))}
                </div>
              )}

              {job.error && (
                <p className="text-[12px] text-destructive bg-destructive/10 rounded-lg p-2">{job.error}</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
