import { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { ChatInterface } from '@/components/ChatInterface';
import { SettingsDrawer } from '@/components/SettingsDrawer';
import { KnowledgeDrawer } from '@/components/KnowledgeDrawer';
import { useChatStore } from '@/stores/chatStore';
import { Circle } from 'lucide-react';

const Index = () => {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const conversation = useChatStore((state) => state.getActiveConversation());

  return (
    <div className="h-screen flex bg-background">
      <Sidebar
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenKnowledge={() => setKnowledgeOpen(true)}
      />

      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-14 flex items-center justify-between px-5 border-b border-border/50">
          <div className="flex items-center gap-3">
            <h1 className="text-[15px] font-medium text-foreground tracking-tight">
              {conversation?.title || 'New conversation'}
            </h1>
            <span className="text-[12px] text-muted-foreground">
              {conversation?.messages.length || 0} messages
            </span>
          </div>
          
          <div className="flex items-center gap-1.5 text-[12px] text-muted-foreground">
            <Circle className="w-2 h-2 fill-primary text-primary" />
            <span>Ready</span>
          </div>
        </header>

        <ChatInterface />
      </main>

      <SettingsDrawer isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <KnowledgeDrawer isOpen={knowledgeOpen} onClose={() => setKnowledgeOpen(false)} />
    </div>
  );
};

export default Index;
