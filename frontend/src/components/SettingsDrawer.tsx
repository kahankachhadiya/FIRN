import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Server, Cpu, Database, Key, RotateCcw } from 'lucide-react';
import { useConfigStore } from '@/stores/configStore';
import { llmModels, embeddingModels, vectorDatabases } from '@/services/mockData';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface SettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsDrawer({ isOpen, onClose }: SettingsDrawerProps) {
  const { config, setConfig, resetConfig } = useConfigStore();
  const [localConfig, setLocalConfig] = useState(config);

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const handleSave = () => {
    setConfig(localConfig);
    onClose();
  };

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
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 400, damping: 40 }}
            className="fixed right-0 top-0 bottom-0 w-full max-w-[400px] surface-1 border-l border-border/50 z-50 flex flex-col"
          >
            {/* Header */}
            <div className="h-14 flex items-center justify-between px-4 border-b border-border/50">
              <span className="font-semibold text-[15px] text-foreground">Settings</span>
              <button
                onClick={onClose}
                className="w-7 h-7 rounded-lg hover:bg-surface-2 flex items-center justify-center transition-colors"
              >
                <X className="w-4 h-4 text-muted-foreground" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-6">
              {/* Models */}
              <section className="space-y-3">
                <div className="flex items-center gap-2 text-[12px] font-medium text-muted-foreground uppercase tracking-wider">
                  <Cpu className="w-3.5 h-3.5" />
                  Models
                </div>

                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <label className="text-[12px] text-muted-foreground">Language Model</label>
                    <Select
                      value={localConfig.llm_model}
                      onValueChange={(value) =>
                        setLocalConfig({ ...localConfig, llm_model: value })
                      }
                    >
                      <SelectTrigger className="h-9 surface-2 border-border/60 text-[13px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {llmModels.map((model) => (
                          <SelectItem key={model.value} value={model.value} className="text-[13px]">
                            {model.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-[12px] text-muted-foreground">Embedding Model</label>
                    <Select
                      value={localConfig.embedding_model}
                      onValueChange={(value) =>
                        setLocalConfig({ ...localConfig, embedding_model: value })
                      }
                    >
                      <SelectTrigger className="h-9 surface-2 border-border/60 text-[13px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {embeddingModels.map((model) => (
                          <SelectItem key={model.value} value={model.value} className="text-[13px]">
                            {model.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </section>

              {/* Database */}
              <section className="space-y-3">
                <div className="flex items-center gap-2 text-[12px] font-medium text-muted-foreground uppercase tracking-wider">
                  <Database className="w-3.5 h-3.5" />
                  Vector Store
                </div>

                <Select
                  value={localConfig.db_type}
                  onValueChange={(value) =>
                    setLocalConfig({ ...localConfig, db_type: value })
                  }
                >
                  <SelectTrigger className="h-9 surface-2 border-border/60 text-[13px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {vectorDatabases.map((db) => (
                      <SelectItem key={db.value} value={db.value} className="text-[13px]">
                        {db.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </section>

              {/* Connection */}
              <section className="space-y-3">
                <div className="flex items-center gap-2 text-[12px] font-medium text-muted-foreground uppercase tracking-wider">
                  <Server className="w-3.5 h-3.5" />
                  Connection
                </div>

                <div className="space-y-1.5">
                  <label className="text-[12px] text-muted-foreground">Backend URL</label>
                  <Input
                    value={localConfig.backend_url}
                    onChange={(e) =>
                      setLocalConfig({ ...localConfig, backend_url: e.target.value })
                    }
                    placeholder="http://localhost:8000"
                    className="h-9 surface-2 border-border/60 text-[13px] font-mono"
                  />
                </div>
              </section>

              {/* API Keys */}
              <section className="space-y-3">
                <div className="flex items-center gap-2 text-[12px] font-medium text-muted-foreground uppercase tracking-wider">
                  <Key className="w-3.5 h-3.5" />
                  API Keys
                </div>

                <div className="space-y-3">
                  {[
                    { key: 'openai', label: 'OpenAI', placeholder: 'sk-...' },
                    { key: 'anthropic', label: 'Anthropic', placeholder: 'sk-ant-...' },
                    { key: 'local', label: 'Custom', placeholder: 'Optional' },
                  ].map(({ key, label, placeholder }) => (
                    <div key={key} className="space-y-1.5">
                      <label className="text-[12px] text-muted-foreground">{label}</label>
                      <Input
                        type="password"
                        value={(localConfig.api_keys as any)[key]}
                        onChange={(e) =>
                          setLocalConfig({
                            ...localConfig,
                            api_keys: { ...localConfig.api_keys, [key]: e.target.value },
                          })
                        }
                        placeholder={placeholder}
                        className="h-9 surface-2 border-border/60 text-[13px] font-mono"
                      />
                    </div>
                  ))}
                </div>
              </section>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-border/50 space-y-2">
              <button
                onClick={handleSave}
                className="w-full h-9 rounded-lg bg-primary text-primary-foreground text-[13px] font-medium transition-all hover:bg-primary/90"
              >
                Save changes
              </button>
              <button
                onClick={resetConfig}
                className="w-full h-9 rounded-lg border border-border/60 hover:border-border hover:bg-surface-2 text-[13px] font-medium text-muted-foreground flex items-center justify-center gap-2 transition-all"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset defaults
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
