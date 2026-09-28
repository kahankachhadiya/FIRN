import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Brain, FileText } from 'lucide-react';
import { MemoryView } from './MemoryView';
import { SourceManager } from './SourceManager';

interface KnowledgeDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

type Tab = 'memory' | 'sources';

export function KnowledgeDrawer({ isOpen, onClose }: KnowledgeDrawerProps) {
  const [activeTab, setActiveTab] = useState<Tab>('sources');

  const tabs = [
    { id: 'sources' as Tab, label: 'Sources', icon: FileText },
    { id: 'memory' as Tab, label: 'Memory', icon: Brain },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40"
          />

          <motion.div
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', stiffness: 400, damping: 40 }}
            className="fixed left-0 top-0 bottom-0 w-full max-w-[380px] surface-1 border-r border-border/50 z-50 flex flex-col"
          >
            {/* Header */}
            <div className="h-14 flex items-center justify-between px-4 border-b border-border/50">
              <span className="font-semibold text-[15px] text-foreground">Knowledge</span>
              <button
                onClick={onClose}
                className="w-7 h-7 rounded-lg hover:bg-surface-2 flex items-center justify-center transition-colors"
              >
                <X className="w-4 h-4 text-muted-foreground" />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex p-1.5 mx-4 mt-3 rounded-lg surface-2 border border-border/40">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-md text-[12px] font-medium transition-all duration-150 ${
                    activeTab === tab.id
                      ? 'bg-surface-1 text-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-secondary-foreground'
                  }`}
                >
                  <tab.icon className="w-3.5 h-3.5" />
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto scrollbar-thin p-4">
              <AnimatePresence mode="wait">
                {activeTab === 'memory' && (
                  <motion.div
                    key="memory"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    transition={{ duration: 0.15 }}
                  >
                    <MemoryView />
                  </motion.div>
                )}
                {activeTab === 'sources' && (
                  <motion.div
                    key="sources"
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.15 }}
                  >
                    <SourceManager />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
