import { motion, AnimatePresence } from 'framer-motion';
import { Citation } from '@/services/types';
import { X, FileText, ArrowUpRight, Hash } from 'lucide-react';

interface CitationCardProps {
  citation: Citation;
  onClose: () => void;
}

export function CitationCard({ citation, onClose }: CitationCardProps) {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 20, scale: 0.96 }}
        transition={{ type: 'spring', stiffness: 500, damping: 35 }}
        className="fixed right-6 top-1/2 -translate-y-1/2 w-[380px] surface-2 border border-border/60 rounded-xl overflow-hidden z-50"
      >
        {/* Header */}
        <div className="flex items-start justify-between p-4 border-b border-border/40">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-surface-3 flex items-center justify-center flex-shrink-0">
              <FileText className="w-4 h-4 text-muted-foreground" />
            </div>
            <div className="min-w-0">
              <p className="text-[13px] font-medium text-foreground truncate">{citation.source}</p>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[12px] text-muted-foreground flex items-center gap-1">
                  <Hash className="w-3 h-3" />
                  Page {citation.page}
                </span>
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                  {Math.round(citation.score * 100)}% match
                </span>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-lg hover:bg-surface-3 flex items-center justify-center transition-colors"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          <p className="text-[13px] text-secondary-foreground leading-relaxed">
            "{citation.text}"
          </p>
        </div>

        {/* Footer */}
        <div className="px-4 pb-4">
          <button className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg border border-border/60 hover:border-border hover:bg-surface-3 text-[13px] font-medium text-secondary-foreground transition-all duration-150">
            <ArrowUpRight className="w-3.5 h-3.5" />
            Open source document
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
