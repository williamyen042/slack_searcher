"use client";

import { useState, useRef, useEffect, FormEvent } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

// Render assistant message text: convert [source](url) citations to <a> links
function MessageContent({ text }: { text: string }) {
  // Split on markdown-style links: [source](url)
  const parts = text.split(/(\[.*?\]\(https?:\/\/[^\)]+\))/g);
  return (
    <span>
      {parts.map((part, i) => {
        const match = part.match(/^\[(.*?)\]\((https?:\/\/[^\)]+)\)$/);
        if (match) {
          return (
            <a
              key={i}
              href={match[2]}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#3b82d4", textDecoration: "underline", fontSize: 12 }}
            >
              {match[1]}
            </a>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    const newMessages: Message[] = [...messages, { role: "user", content: query }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    // Append empty assistant message to stream into
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages }),
      });

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") break;
          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta?.content ?? "";
            if (delta) {
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: "assistant",
                  content: updated[updated.length - 1].content + delta,
                };
                return updated;
              });
            }
          } catch {
            // skip malformed SSE frames
          }
        }
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", maxWidth: 760, margin: "0 auto", fontFamily: '-apple-system, "Segoe UI", system-ui, sans-serif', fontSize: 14 }}>
      {/* Header */}
      <div style={{ padding: "16px 20px", borderBottom: "1px solid #e5e7eb", background: "#fff" }}>
        <div style={{ fontWeight: 700, fontSize: 16 }}>Slack Searcher</div>
        <div style={{ color: "#57606a", fontSize: 12 }}>Answers grounded in your Slack workspace — no hallucination</div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "20px", background: "#f7f8fa" }}>
        {messages.length === 0 && (
          <div style={{ color: "#57606a", textAlign: "center", marginTop: 60 }}>
            Ask anything about your Slack workspace.
            <br />
            <span style={{ fontSize: 12 }}>Answers come only from real messages — with source links.</span>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              marginBottom: 16,
              display: "flex",
              flexDirection: "column",
              alignItems: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                maxWidth: "80%",
                padding: "10px 14px",
                borderRadius: 8,
                background: msg.role === "user" ? "#3b82d4" : "#fff",
                color: msg.role === "user" ? "#fff" : "#1f2328",
                border: msg.role === "assistant" ? "1px solid #e5e7eb" : "none",
                lineHeight: 1.6,
                whiteSpace: "pre-wrap",
              }}
            >
              {msg.role === "assistant" ? (
                <MessageContent text={msg.content} />
              ) : (
                msg.content
              )}
              {msg.role === "assistant" && loading && i === messages.length - 1 && (
                <span style={{ color: "#57606a" }}>▌</span>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        style={{ padding: "12px 20px", borderTop: "1px solid #e5e7eb", background: "#fff", display: "flex", gap: 8 }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your Slack workspace…"
          disabled={loading}
          style={{
            flex: 1,
            padding: "9px 12px",
            border: "1px solid #e5e7eb",
            borderRadius: 6,
            fontSize: 14,
            outline: "none",
            background: loading ? "#f7f8fa" : "#fff",
          }}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          style={{
            padding: "9px 18px",
            background: loading || !input.trim() ? "#e5e7eb" : "#3b82d4",
            color: loading || !input.trim() ? "#57606a" : "#fff",
            border: "none",
            borderRadius: 6,
            fontSize: 14,
            fontWeight: 600,
            cursor: loading || !input.trim() ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Searching…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
