import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useChatStore } from '@/stores/chatStore';
import { MessageBubble, markdownComponents } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { CitationCard } from './CitationCard';
import { Sparkles, Bot } from 'lucide-react';

/**
 * Guards against unclosed code fences or HTML tags during streaming.
 * - If the count of ``` openings is odd (fence still open), appends a closing ``` 
 * - If there are more <think> than </think>, appends </think>
 */
function safeStreamBuffer(text: string): string {
  let safe = text;
  
  const fenceCount = (safe.match(/^```/gm) ?? []).length;
  if (fenceCount % 2 !== 0) safe += '\n```';

  const thinkOpenCount = (safe.match(/<think>/g) ?? []).length;
  const thinkCloseCount = (safe.match(/<\/think>/g) ?? []).length;
  if (thinkOpenCount > thinkCloseCount) safe += '\n</think>';

  return safe;
}

export function ChatInterface() {
  const {
    activeCitation,
    setActiveCitation,
    isStreaming,
    streamingContent,
    isLoading,
  } = useChatStore();

  const conversation = useChatStore((state) => state.getActiveConversation());
  const messages = conversation?.messages || [];

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  const activeCitationData = activeCitation
    ? messages
        .find((m) => m.id === activeCitation.messageId)
        ?.citations.find((c) => c.id === activeCitation.citationId)
    : null;

  return (
    <div className="flex-1 flex flex-col h-full relative noise-overlay">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="max-w-[740px] mx-auto px-4 py-8">
          {messages.length === 0 && !isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4 }}
              className="py-24"
            >
              <div className="text-center">
                <div className="w-12 h-12 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-5">
                  <Sparkles className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-xl font-semibold text-foreground mb-2 tracking-tight">
                  How can I help you?
                </h2>
                <p className="text-[14px] text-muted-foreground max-w-sm mx-auto leading-relaxed">
                  Ask questions about your documents. I'll provide answers with precise citations.
                </p>
              </div>

              {/* Quick suggestions */}
              <div className="mt-10 flex flex-wrap justify-center gap-2">
                {[
                  'Summarize the key points',
                  'Compare these sections',
                  'Find specific data',
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    className="px-3.5 py-2 rounded-lg border border-border/60 text-[13px] text-muted-foreground hover:text-secondary-foreground hover:border-border hover:bg-surface-2/50 transition-all duration-150"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          <div className="space-y-6">
            <AnimatePresence mode="popLayout">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </AnimatePresence>
          </div>

          {/* ── Live streaming: ReactMarkdown on-the-fly ───────────────────── */}
          {isStreaming && streamingContent && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6"
            >
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-lg flex-shrink-0 flex items-center justify-center bg-primary/10 border border-primary/20">
                  <Bot className="w-3.5 h-3.5 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  {/*
                    prose-rag applies the same heading/table/hr styles as the
                    finalized message. markdownComponents is the shared renderer
                    map exported from MessageBubble.
                  */}
                  <div className="prose-rag">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={markdownComponents}
                    >
                      {safeStreamBuffer(streamingContent).replace(/<link>([\s\S]*?)<\/link>/gi, (_, rawUrl) => {
                        const url = rawUrl.trim();
                        if (/\.(png|jpg|jpeg|gif|webp)$/i.test(url)) return `\n\n![](${url})\n\n`;
                        return '';
                      }).replace(/!\[([^\]]*)\]\((\/[^)]+\.(?:png|jpg|jpeg|gif|webp))\)/gi, (_, alt, path) => {
                        return `![${alt}](http://localhost:8000/api/files${path})`;
                      })}
                    </ReactMarkdown>
                    <span className="streaming-cursor" />
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Initial loading dots (before first token) */}
          {isLoading && !isStreaming && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6"
            >
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-lg flex-shrink-0 flex items-center justify-center bg-primary/10 border border-primary/20">
                  <Bot className="w-3.5 h-3.5 text-primary" />
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-pulse" style={{ animationDelay: '0ms' }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-pulse" style={{ animationDelay: '150ms' }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-pulse" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="flex-shrink-0 pt-2">
        <ChatInput />
      </div>

      {/* Citation overlay */}
      <AnimatePresence>
        {activeCitationData && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setActiveCitation(null, null)}
              className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40"
            />
            <CitationCard
              citation={activeCitationData}
              onClose={() => setActiveCitation(null, null)}
            />
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
