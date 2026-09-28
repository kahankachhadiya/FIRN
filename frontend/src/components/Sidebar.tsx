import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Plus, Trash2, Sparkles, Search, FolderOpen, X, Database } from 'lucide-react';
import { useChatStore } from '@/stores/chatStore';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface SidebarProps {
  onOpenSettings: () => void;
  onOpenKnowledge: () => void;
}

export function Sidebar({ onOpenSettings, onOpenKnowledge }: SidebarProps) {
  const { 
    conversations, 
    activeConversationId, 
    createConversation, 
    deleteConversation,
    setActiveConversation 
  } = useChatStore();
  
  const [expanded, setExpanded] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const navigate = useNavigate();

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <>
      {/* Icon Rail */}
      <div className="w-[52px] flex flex-col items-center py-3 surface-1 border-r border-border/50">
        {/* Logo */}
        <button 
          onClick={() => setExpanded(!expanded)}
          className="w-9 h-9 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-4 hover:bg-primary/20 transition-colors"
        >
          <Sparkles className="w-4 h-4 text-primary" />
        </button>

        {/* Actions */}
        <div className="flex flex-col items-center gap-1">
          <button
            onClick={() => createConversation()}
            className="w-9 h-9 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-surface-2 transition-all"
            title="New chat"
          >
            <Plus className="w-[18px] h-[18px]" />
          </button>
          
          <button
            onClick={() => setExpanded(!expanded)}
            className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all ${
              expanded 
                ? 'text-foreground bg-surface-2' 
                : 'text-muted-foreground hover:text-foreground hover:bg-surface-2'
            }`}
            title="History"
          >
            <MessageSquare className="w-[18px] h-[18px]" />
          </button>

          <button
            onClick={onOpenKnowledge}
            className="w-9 h-9 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-surface-2 transition-all"
            title="Knowledge"
          >
            <FolderOpen className="w-[18px] h-[18px]" />
          </button>

          <button
            onClick={() => navigate('/documents')}
            className="w-9 h-9 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-surface-2 transition-all"
            title="Document Ingestion"
          >
            <Database className="w-[18px] h-[18px]" />
          </button>

          <button
            className="w-9 h-9 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-surface-2 transition-all"
            title="Search"
          >
            <Search className="w-[18px] h-[18px]" />
          </button>
        </div>

        {/* Bottom - Settings */}
        <div className="mt-auto">
          <button
            onClick={onOpenSettings}
            className="w-9 h-9 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-[13px] font-semibold hover:opacity-90 transition-opacity"
            title="Settings"
          >
            U
          </button>
        </div>
      </div>

      {/* Expanded Panel */}
      <AnimatePresence>
        {expanded && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              onClick={() => setExpanded(false)}
              className="fixed inset-0 z-40 bg-background/40 backdrop-blur-sm"
            />
            <motion.aside
              initial={{ x: -240, opacity: 0 }}
              animate={{ x: 52, opacity: 1 }}
              exit={{ x: -240, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 350 }}
              className="fixed left-0 top-0 bottom-0 z-50 w-[260px] flex flex-col surface-1 border-r border-border/50 shadow-xl"
            >
              {/* Header */}
              <div className="h-14 flex items-center justify-between px-4 border-b border-border/50">
                <span className="text-[13px] font-medium text-foreground">History</span>
                <button
                  onClick={() => setExpanded(false)}
                  className="w-7 h-7 rounded-lg hover:bg-surface-2 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Conversations */}
              <div className="flex-1 overflow-y-auto scrollbar-none p-2">
                <div className="space-y-0.5">
                  <AnimatePresence mode="popLayout">
                    {conversations.map((convo) => (
                      <motion.div
                        key={convo.id}
                        layout
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onMouseEnter={() => setHoveredId(convo.id)}
                        onMouseLeave={() => setHoveredId(null)}
                        onClick={() => {
                          setActiveConversation(convo.id);
                          setExpanded(false);
                        }}
                        className={`group relative flex items-center gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150 ${
                          activeConversationId === convo.id
                            ? 'bg-surface-2 text-foreground'
                            : 'text-muted-foreground hover:text-secondary-foreground hover:bg-surface-2/50'
                        }`}
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-[13px] truncate leading-tight">{convo.title}</p>
                          <p className="text-[11px] opacity-50 mt-0.5">{formatDate(convo.updatedAt)}</p>
                        </div>
                        
                        <AnimatePresence>
                          {hoveredId === convo.id && conversations.length > 1 && (
                            <motion.button
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              onClick={(e) => {
                                e.stopPropagation();
                                deleteConversation(convo.id);
                              }}
                              className="w-6 h-6 rounded-md hover:bg-destructive/20 flex items-center justify-center transition-colors"
                            >
                              <Trash2 className="w-3 h-3 text-destructive/70" />
                            </motion.button>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
