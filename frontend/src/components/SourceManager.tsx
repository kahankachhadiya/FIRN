import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Upload, Trash2, CheckCircle, Loader2, AlertCircle } from 'lucide-react';
import { api } from '@/services/api';
import { Source } from '@/services/types';

export function SourceManager() {
  const [sources, setSources] = useState<Source[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    loadSources();
  }, []);

  const loadSources = async () => {
    const data = await api.getSources();
    setSources(data);
  };

  const handleUpload = async (files: FileList | null) => {
    if (!files) return;
    for (const file of Array.from(files)) {
      const source = await api.uploadSource(file);
      setSources((prev) => [...prev, source]);
    }
  };

  const handleDelete = async (id: string) => {
    await api.deleteSource(id);
    setSources(sources.filter((s) => s.id !== id));
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleUpload(e.dataTransfer.files);
  }, []);

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getStatusIcon = (status: Source['status']) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="w-3.5 h-3.5 text-primary" />;
      case 'indexing':
        return <Loader2 className="w-3.5 h-3.5 text-muted-foreground animate-spin" />;
      case 'uploading':
        return <Loader2 className="w-3.5 h-3.5 text-muted-foreground animate-spin" />;
      case 'error':
        return <AlertCircle className="w-3.5 h-3.5 text-destructive" />;
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <div className="w-7 h-7 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
          <FileText className="w-3.5 h-3.5 text-primary" />
        </div>
        <div>
          <h3 className="text-[13px] font-medium text-foreground">Sources</h3>
          <p className="text-[11px] text-muted-foreground">{sources.length} documents</p>
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        className={`relative border border-dashed rounded-lg p-6 text-center transition-all ${
          isDragging ? 'border-primary bg-primary/5' : 'border-border/60 hover:border-border'
        }`}
      >
        <input
          type="file"
          multiple
          accept=".pdf,.doc,.docx,.txt,.md"
          onChange={(e) => handleUpload(e.target.files)}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <Upload className={`w-6 h-6 mx-auto mb-2 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
        <p className="text-[12px] text-muted-foreground">Drop files or click to upload</p>
        <p className="text-[11px] text-muted-foreground/60 mt-1">PDF, DOC, TXT, MD</p>
      </div>

      {/* Source list */}
      <div className="space-y-1.5">
        <AnimatePresence mode="popLayout">
          {sources.map((source) => (
            <motion.div
              key={source.id}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="group flex items-center gap-3 p-2.5 rounded-lg surface-2 border border-border/40 hover:border-border/60 transition-colors"
            >
              <div className="w-8 h-8 rounded-md bg-surface-3 flex items-center justify-center flex-shrink-0">
                <FileText className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[12px] font-medium text-foreground truncate">{source.name}</p>
                <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                  <span>{formatSize(source.size)}</span>
                  {source.pages && (
                    <>
                      <span>·</span>
                      <span>{source.pages}p</span>
                    </>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1.5">
                {getStatusIcon(source.status)}
                <button
                  onClick={() => handleDelete(source.id)}
                  className="w-6 h-6 rounded-md opacity-0 group-hover:opacity-100 hover:bg-destructive/10 flex items-center justify-center transition-all"
                >
                  <Trash2 className="w-3 h-3 text-destructive/70" />
                </button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {sources.length === 0 && (
          <div className="text-center py-8">
            <p className="text-[12px] text-muted-foreground">No sources uploaded</p>
          </div>
        )}
      </div>
    </div>
  );
}
