import React from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';
import { Message } from '@/services/types';
import { useChatStore } from '@/stores/chatStore';
import { User, Bot, FileText, ExternalLink, FileImage } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
}

// ── Shared markdown component map ───────────────────────────────────────────
// Used by both MessageBubble (final) and the streaming renderer.
export const markdownComponents: Components = {
  // Headings
  h1: ({ children }) => (
    <h2 className="prose-rag-h2">{children}</h2>
  ),
  h2: ({ children }) => (
    <h2>{children}</h2>
  ),
  h3: ({ children }) => (
    <h3>{children}</h3>
  ),
  h4: ({ children }) => (
    <h4>{children}</h4>
  ),

  // Horizontal rule separator
  hr: () => <hr />,

  // Paragraph — supports inline citation [N] buttons
  p: ({ children }) => (
    <p>{children}</p>
  ),

  // Tables
  table: ({ children }) => (
    <div className="overflow-x-auto my-3">
      <table>{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead>{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => <tr>{children}</tr>,
  th: ({ children }) => <th>{children}</th>,
  td: ({ children }) => <td>{children}</td>,

  // Code
  code: ({ children, className }) => {
    const isBlock = className?.startsWith('language-');
    if (isBlock) {
      return <code className={className}>{children}</code>;
    }
    return <code>{children}</code>;
  },
  pre: ({ children }) => <pre>{children}</pre>,

  // Blockquote
  blockquote: ({ children }) => <blockquote>{children}</blockquote>,

  // Lists
  ul: ({ children }) => <ul>{children}</ul>,
  ol: ({ children }) => <ol>{children}</ol>,
  li: ({ children }) => <li>{children}</li>,

  // Inline image (standard markdown ![alt](url))
  img: ({ src, alt }) => (
    <img
      src={src}
      alt={alt ?? ''}
      className="rag-image"
      loading="lazy"
    />
  ),

  // DeepSeek / reasoning models <think> tags
  // @ts-expect-error — custom HTML element
  think: ({ children }: { children?: React.ReactNode }) => (
    <div className="text-muted-foreground text-[13px] my-5 border-l-[3px] border-primary/20 pl-4 py-1.5 bg-surface-2/30 rounded-r-xl font-mono leading-relaxed max-h-[400px] overflow-y-auto scrollbar-thin">
      <div className="font-sans font-semibold text-[11px] uppercase tracking-wider mb-2.5 flex items-center gap-2 opacity-60">
        <Bot className="w-3.5 h-3.5" />
        Thinking Process
      </div>
      <div className="opacity-75 whitespace-pre-wrap">
        {children}
      </div>
    </div>
  ),
};

// ── Document source card ─────────────────────────────────────────────────────
function DocSourceCard({ url }: { url: string }) {
  const filename = url.split('/').pop() ?? url;
  const isPdf = filename.toLowerCase().endsWith('.pdf');
  const Icon = isPdf ? FileText : FileImage;

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <a
      href={url}
      onClick={handleClick}
      rel="noopener noreferrer"
      className="flex items-center gap-3 px-3.5 py-2.5 rounded-xl bg-surface-2 border border-border/50 hover:bg-surface-3 hover:border-border transition-all duration-150 group/doc"
    >
      <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0">
        <Icon className="w-4 h-4 text-primary" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[13px] font-medium text-foreground/80 group-hover/doc:text-foreground truncate transition-colors">
          {filename}
        </p>
        <p className="text-[11px] text-muted-foreground mt-0.5">
          {isPdf ? 'PDF Document' : 'Source File'}
        </p>
      </div>
      <ExternalLink className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0 opacity-0 group-hover/doc:opacity-100 transition-opacity" />
    </a>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export function MessageBubble({ message }: MessageBubbleProps) {
  const { setActiveCitation, activeCitation } = useChatStore();
  const isUser = message.role === 'user';

  // Citation-aware paragraph renderer (used inside the prose block)
  const ParagraphWithCitations = ({ children }: { children?: React.ReactNode }) => {
    const processChildren = (nodes: React.ReactNode): React.ReactNode =>
      React.Children.map(nodes, (child, i) => {
        if (typeof child !== 'string') return child;
        const parts = child.split(/(\[\d+\])/g);
        if (parts.length === 1) return child;
        return parts.map((part, idx) => {
          const match = part.match(/^\[(\d+)\]$/);
          if (match) {
            const citationId = match[1];
            const citation = message.citations.find((c) => c.id === citationId);
            if (citation) {
              const isActive =
                activeCitation?.messageId === message.id &&
                activeCitation?.citationId === citationId;
              return (
                <button
                  key={`${i}-${idx}`}
                  onClick={() => setActiveCitation(message.id, citationId)}
                  className={`inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[11px] font-mono font-medium rounded mx-0.5 transition-all duration-150 ${
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-primary/20 text-primary hover:bg-primary/30'
                  }`}
                >
                  {citationId}
                </button>
              );
            }
          }
          return <span key={`${i}-${idx}`}>{part}</span>;
        });
      });

    return <p>{processChildren(children)}</p>;
  };

  const renderContent = () => {
    const backendUrl = useChatStore.getState ? 
      (window as any).__backendUrl || 'http://localhost:8000' : 
      'http://localhost:8000';

    const preprocessMarkdown = (text: string) => {
      // Remove think blocks
      let result = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
      // Convert <link>IMAGE</link> to markdown image, strip doc link tags
      result = result.replace(/<link>([\s\S]*?)<\/link>/gi, (_, rawUrl) => {
        const url = rawUrl.trim();
        const isImage = /\.(png|jpg|jpeg|gif|webp)$/i.test(url);
        if (isImage) return `\n\n![](${url})\n\n`;
        return '';
      });
      // Convert local absolute paths in markdown images to HTTP URLs
      // e.g. ![alt](/home/user/.../image.png) → ![alt](http://localhost:8000/api/files/home/user/.../image.png)
      result = result.replace(/!\[([^\]]*)\]\((\/[^)]+\.(?:png|jpg|jpeg|gif|webp))\)/gi, (_, alt, path) => {
        return `![${alt}](http://localhost:8000/api/files${path})`;
      });
      return result;
    };

    const processedContent = preprocessMarkdown(message.content);

    if (isUser) {
      return (
        <p className="whitespace-pre-wrap leading-relaxed text-[14px]">{processedContent}</p>
      );
    }

    return (
      <div className="prose-rag">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            ...markdownComponents,
            p: ParagraphWithCitations,
          }}
        >
          {processedContent}
        </ReactMarkdown>
      </div>
    );
  };

  // Source documents: only from LLM-emitted <link> tags classified as documents.
  // Images are rendered inline; documents appear as cards below.
  const documents = (message.references?.documents ?? []).filter(
    (url) => !/\.(png|jpg|jpeg|gif|webp)$/i.test(url)
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="group"
    >
      <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
        {/* Avatar */}
        <div
          className={`w-7 h-7 rounded-lg flex-shrink-0 flex items-center justify-center ${
            isUser
              ? 'bg-surface-3'
              : 'bg-primary/10 border border-primary/20'
          }`}
        >
          {isUser ? (
            <User className="w-3.5 h-3.5 text-muted-foreground" />
          ) : (
            <Bot className="w-3.5 h-3.5 text-primary" />
          )}
        </div>

        {/* Content */}
        <div className={`flex-1 min-w-0 ${isUser ? 'flex justify-end' : ''}`}>
          <div
            className={`inline-block max-w-[90%] ${
              isUser
                ? 'bg-surface-3 text-foreground rounded-2xl rounded-tr-md px-4 py-2.5 text-[14px]'
                : 'text-foreground/90 w-full'
            }`}
          >
            {renderContent()}
          </div>

          {/* Citation count badge */}
          {!isUser && message.citations.length > 0 && (
            <div className="mt-2 flex items-center gap-1.5">
              <span className="text-[11px] text-muted-foreground">
                {message.citations.length} source{message.citations.length !== 1 ? 's' : ''}
              </span>
            </div>
          )}

          {/* Source document cards — only LLM-cited documents */}
          {!isUser && documents.length > 0 && (
            <div className="mt-3 space-y-1.5 max-w-[90%]">
              <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Sources
              </p>
              {documents.map((url) => (
                <DocSourceCard key={url} url={url} />
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
