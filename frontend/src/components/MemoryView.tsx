import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Plus, Pencil, Trash2, Check, X } from 'lucide-react';
import { api } from '@/services/api';
import { Memory } from '@/services/types';
import { Input } from '@/components/ui/input';

export function MemoryView() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [newMemory, setNewMemory] = useState('');
  const [isAdding, setIsAdding] = useState(false);

  useEffect(() => {
    loadMemories();
  }, []);

  const loadMemories = async () => {
    const data = await api.getMemories();
    setMemories(data);
  };

  const handleAdd = async () => {
    if (!newMemory.trim()) return;
    const memory = await api.addMemory(newMemory.trim());
    setMemories([...memories, memory]);
    setNewMemory('');
    setIsAdding(false);
  };

  const handleUpdate = async (id: string) => {
    if (!editContent.trim()) return;
    await api.updateMemory(id, editContent.trim());
    setMemories(memories.map((m) => (m.id === id ? { ...m, content: editContent.trim() } : m)));
    setEditingId(null);
    setEditContent('');
  };

  const handleDelete = async (id: string) => {
    await api.deleteMemory(id);
    setMemories(memories.filter((m) => m.id !== id));
  };

  const startEdit = (memory: Memory) => {
    setEditingId(memory.id);
    setEditContent(memory.content);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
            <Brain className="w-3.5 h-3.5 text-primary" />
          </div>
          <div>
            <h3 className="text-[13px] font-medium text-foreground">Memory</h3>
            <p className="text-[11px] text-muted-foreground">{memories.length} items</p>
          </div>
        </div>
        <button
          onClick={() => setIsAdding(true)}
          className="w-7 h-7 rounded-lg hover:bg-surface-3 flex items-center justify-center transition-colors"
        >
          <Plus className="w-4 h-4 text-muted-foreground" />
        </button>
      </div>

      <div className="space-y-1.5">
        <AnimatePresence mode="popLayout">
          {isAdding && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="p-3 rounded-lg surface-3 border border-border/40 space-y-2">
                <Input
                  value={newMemory}
                  onChange={(e) => setNewMemory(e.target.value)}
                  placeholder="Add a memory..."
                  className="h-8 text-[13px] bg-surface-2"
                  autoFocus
                  onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleAdd}
                    className="flex-1 h-7 rounded-md bg-primary text-primary-foreground text-[12px] font-medium"
                  >
                    Add
                  </button>
                  <button
                    onClick={() => { setIsAdding(false); setNewMemory(''); }}
                    className="h-7 px-3 rounded-md bg-surface-2 text-muted-foreground text-[12px]"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {memories.map((memory) => (
            <motion.div
              key={memory.id}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="group p-3 rounded-lg surface-2 border border-border/40 hover:border-border/60 transition-colors"
            >
              {editingId === memory.id ? (
                <div className="space-y-2">
                  <Input
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="h-8 text-[13px] bg-surface-1"
                    autoFocus
                    onKeyDown={(e) => e.key === 'Enter' && handleUpdate(memory.id)}
                  />
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleUpdate(memory.id)}
                      className="w-7 h-7 rounded-md bg-primary/10 text-primary flex items-center justify-center"
                    >
                      <Check className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => { setEditingId(null); setEditContent(''); }}
                      className="w-7 h-7 rounded-md bg-surface-3 text-muted-foreground flex items-center justify-center"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex items-start justify-between gap-3">
                  <p className="text-[13px] text-secondary-foreground leading-relaxed">{memory.content}</p>
                  <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => startEdit(memory)}
                      className="w-6 h-6 rounded-md hover:bg-surface-3 flex items-center justify-center"
                    >
                      <Pencil className="w-3 h-3 text-muted-foreground" />
                    </button>
                    <button
                      onClick={() => handleDelete(memory.id)}
                      className="w-6 h-6 rounded-md hover:bg-destructive/10 flex items-center justify-center"
                    >
                      <Trash2 className="w-3 h-3 text-destructive/70" />
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {memories.length === 0 && !isAdding && (
          <div className="text-center py-8">
            <p className="text-[12px] text-muted-foreground">No memories yet</p>
          </div>
        )}
      </div>
    </div>
  );
}
