import { useState, useRef, useEffect } from 'react';
import { ArrowUp, StopCircle } from 'lucide-react';
import { useChatStore } from '@/stores/chatStore';

export function ChatInput() {
  const [input, setInput] = useState('');
  const { sendMessage, isLoading, isStreaming } = useChatStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;
    const message = input.trim();
    setInput('');
    await sendMessage(message);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    }
  }, [input]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const canSubmit = input.trim() && !isLoading;

  return (
    <div className="w-full max-w-[740px] mx-auto px-4 pb-5">
      <div className="relative">
        {/* Main input container */}
        <div className="surface-2 border border-border/60 rounded-xl overflow-hidden transition-all duration-200 focus-within:border-border focus-within:ring-1 focus-within:ring-primary/20">
          {/* Input row */}
          <div className="flex items-end gap-2 p-2">

            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your documents..."
              rows={1}
              className="flex-1 resize-none bg-transparent border-0 outline-none text-foreground placeholder:text-muted-foreground text-[14px] leading-relaxed py-1.5 scrollbar-none"
              disabled={isLoading}
            />

            <button
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-150 ${
                canSubmit
                  ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                  : 'bg-surface-3 text-muted-foreground'
              }`}
            >
              {isStreaming ? (
                <StopCircle className="w-4 h-4" />
              ) : (
                <ArrowUp className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        {/* Hint text */}
        <div className="flex items-center justify-center gap-3 mt-2">
          <span className="text-[11px] text-muted-foreground/70">
            Enter to send · Shift+Enter for new line
          </span>
        </div>
      </div>
    </div>
  );
}
