import { useEffect, useMemo, useRef, useState } from "react";
import { parseSseStream } from "./sse";

type Source = { title?: string; chunk?: string; page?: number };

type ChatMessage =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; sources?: Source[]; latencyMs?: number; models?: Record<string, string> };

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((ev: any) => void) | null;
  onerror: ((ev: any) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

function createSpeechRecognition(): SpeechRecognitionLike | null {
  const w = window as any;
  const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
  if (!Ctor) return null;
  return new Ctor() as SpeechRecognitionLike;
}

function nowId() {
  return Math.random().toString(16).slice(2);
}

async function postJson(path: string, body: unknown) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${txt || resp.statusText}`);
  }
  return resp;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [limit, setLimit] = useState(5);
  const [scriptureFilter, setScriptureFilter] = useState<string>("");
  const [stream, setStream] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const lastSpokenRef = useRef<string>("");

  const apiPaths = useMemo(() => {
    // In dev, Vite proxies /api -> localhost:8000 (see vite.config.ts)
    // In Docker, Nginx proxies /api -> api:8000
    return {
      query: "/api/query",
      stream: "/api/query/stream",
      health: "/api/health",
    };
  }, []);

  useEffect(() => {
    setVoiceSupported(Boolean(createSpeechRecognition()));
  }, []);

  function speak(text: string) {
    if (!ttsEnabled) return;
    if (typeof window === "undefined") return;
    const synth = window.speechSynthesis;
    if (!synth) return;

    const cleaned = text.trim();
    if (!cleaned) return;

    // Prevent repeating if the UI re-renders after completion
    if (lastSpokenRef.current === cleaned) return;
    lastSpokenRef.current = cleaned;

    try {
      synth.cancel();
      const utter = new SpeechSynthesisUtterance(cleaned);
      utter.rate = 1.0;
      utter.pitch = 1.0;
      utter.lang = "en-US";
      synth.speak(utter);
    } catch {
      // ignore
    }
  }

  function toggleListening() {
    setVoiceError(null);
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    const rec = createSpeechRecognition();
    if (!rec) {
      setVoiceError("Voice input is not supported in this browser.");
      setVoiceSupported(false);
      return;
    }

    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;

    rec.onresult = (ev: any) => {
      const results = ev?.results;
      if (!results) return;
      let transcript = "";
      for (let i = ev.resultIndex ?? 0; i < results.length; i++) {
        transcript += results[i][0]?.transcript ?? "";
      }
      transcript = transcript.trim();
      if (!transcript) return;
      setQuestion(transcript);
    };
    rec.onerror = (ev: any) => {
      const msg = ev?.error ? String(ev.error) : "unknown_error";
      setVoiceError(`Voice error: ${msg}`);
      setIsListening(false);
    };
    rec.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = rec;
    try {
      rec.start();
      setIsListening(true);
    } catch (e) {
      setVoiceError(`Voice error: ${e instanceof Error ? e.message : String(e)}`);
      setIsListening(false);
    }
  }

  async function onSend() {
    const q = question.trim();
    if (!q || isLoading) return;

    setQuestion("");
    setMessages((prev) => [...prev, { role: "user", text: q }, { role: "assistant", text: "" }]);
    setIsLoading(true);
    lastSpokenRef.current = "";

    const controller = new AbortController();
    abortRef.current = controller;
    const started = performance.now();

    try {
      if (!stream) {
        const resp = await postJson(apiPaths.query, {
          question: q,
          limit,
          scripture_filter: scriptureFilter.trim() || null,
        });
        const data = (await resp.json()) as { answer: string; sources: Source[]; latency_ms: number; models: Record<string, string> };

        setMessages((prev) => {
          const next = [...prev];
          const lastIdx = next.length - 1;
          next[lastIdx] = {
            role: "assistant",
            text: data.answer,
            sources: data.sources,
            latencyMs: data.latency_ms,
            models: data.models,
          };
          return next;
        });
        speak(data.answer);
        return;
      }

      const resp = await fetch(apiPaths.stream, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          limit,
          scripture_filter: scriptureFilter.trim() || null,
        }),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) {
        const txt = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status}: ${txt || resp.statusText}`);
      }

      let sources: Source[] | undefined;
      let models: Record<string, string> | undefined;
      let latencyMs: number | undefined;
      const assistantMsgId = nowId();
      let streamedText = "";

      // Tag the placeholder assistant message so we always update the right one
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", text: "", sources: [], models: {}, latencyMs: undefined };
        // @ts-expect-error attach internal id
        (next[next.length - 1] as any).__id = assistantMsgId;
        return next;
      });

      for await (const evt of parseSseStream(resp.body)) {
        if (evt.event === "meta") {
          const meta = JSON.parse(evt.data) as { sources: Source[]; models: Record<string, string> };
          sources = meta.sources;
          models = meta.models;
          setMessages((prev) =>
            prev.map((m) => {
              // @ts-expect-error internal id
              if ((m as any).__id !== assistantMsgId) return m;
              if (m.role !== "assistant") return m;
              return { ...m, sources, models };
            }),
          );
        } else if (evt.event === "token") {
          const token = JSON.parse(evt.data) as string;
          streamedText += token;
          setMessages((prev) =>
            prev.map((m) => {
              // @ts-expect-error internal id
              if ((m as any).__id !== assistantMsgId) return m;
              if (m.role !== "assistant") return m;
              return { ...m, text: m.text + token };
            }),
          );
        } else if (evt.event === "latency") {
          const l = JSON.parse(evt.data) as { latency_ms: number };
          latencyMs = l.latency_ms;
          setMessages((prev) =>
            prev.map((m) => {
              // @ts-expect-error internal id
              if ((m as any).__id !== assistantMsgId) return m;
              if (m.role !== "assistant") return m;
              return { ...m, latencyMs };
            }),
          );
        } else if (evt.event === "done") {
          break;
        }
      }

      // Fallback if server didn't send latency
      if (latencyMs == null) {
        const elapsed = Math.round(performance.now() - started);
        setMessages((prev) =>
          prev.map((m) => {
            // @ts-expect-error internal id
            if ((m as any).__id !== assistantMsgId) return m;
            if (m.role !== "assistant") return m;
            return { ...m, latencyMs: elapsed };
          }),
        );
      }

      // Speak the final streamed text
      if (ttsEnabled) speak(streamedText);
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", text: `Error: ${err}` };
        return next;
      });
    } finally {
      setIsLoading(false);
      abortRef.current = null;
    }
  }

  function onStop() {
    abortRef.current?.abort();
  }

  return (
    <div className="page">
      <div className="shell">
        <header className="header">
          <div>
            <div className="title">Semantic QA</div>
            <div className="subtitle">FastAPI + Weaviate + Ollama</div>
          </div>
          <div className="controls">
            <label className="toggle">
              <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
              <span>Streaming</span>
            </label>
            <label className="field">
              <span>Top K</span>
              <input type="number" min={1} max={20} value={limit} onChange={(e) => setLimit(Number(e.target.value || 5))} />
            </label>
            <label className="field">
              <span>Title filter</span>
              <input
                placeholder='e.g. "thetanakh"'
                value={scriptureFilter}
                onChange={(e) => setScriptureFilter(e.target.value)}
              />
            </label>
          </div>
        </header>

        <main className="chat">
          {messages.length === 0 ? (
            <div className="empty">
              <div className="emptyCard">
                Ask a question after you’ve ingested PDFs into Weaviate.
                <div className="hint">Tip: use the streaming toggle for a chat-like feel.</div>
              </div>
            </div>
          ) : (
            messages.map((m, idx) => (
              <div key={idx} className={`msg ${m.role}`}>
                <div className="bubble">
                  <div className="role">{m.role === "user" ? "You" : "Assistant"}</div>
                  <div className="text">{m.text || (m.role === "assistant" ? (isLoading ? "…" : "") : "")}</div>
                  {m.role === "assistant" && (m.sources?.length || m.models || m.latencyMs != null) ? (
                    <div className="meta">
                      {m.latencyMs != null ? <span className="pill">Latency: {m.latencyMs}ms</span> : null}
                      {m.models?.generate_model ? <span className="pill">LLM: {m.models.generate_model}</span> : null}
                      {m.models?.embedding_model ? <span className="pill">Emb: {m.models.embedding_model}</span> : null}
                      {m.sources?.length ? <span className="pill">Sources: {m.sources.length}</span> : null}
                      {m.sources?.length ? (
                        <details className="sources">
                          <summary>Show sources</summary>
                          <ul>
                            {m.sources.map((s, i) => (
                              <li key={i}>
                                <div className="srcTitle">
                                  {s.title ?? "Untitled"} {s.page != null ? <span className="srcPage">p. {s.page}</span> : null}
                                </div>
                                <div className="srcChunk">{(s.chunk ?? "").slice(0, 420)}{(s.chunk?.length ?? 0) > 420 ? "…" : ""}</div>
                              </li>
                            ))}
                          </ul>
                        </details>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </div>
            ))
          )}
        </main>

        <footer className="composer">
          <textarea
            value={question}
            placeholder={voiceSupported ? "Ask a question… (or use Voice)" : "Ask a question…"}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onSend();
            }}
            disabled={isLoading}
          />
          <div className="composerRow">
            <button className="btn" onClick={onSend} disabled={isLoading || !question.trim()}>
              Send
            </button>
            <button className="btn secondary" onClick={toggleListening} disabled={!voiceSupported || isLoading}>
              {isListening ? "Stop voice" : "Voice"}
            </button>
            <label className="toggle" style={{ marginLeft: 0 }}>
              <input
                type="checkbox"
                checked={ttsEnabled}
                onChange={(e) => setTtsEnabled(e.target.checked)}
                disabled={isLoading}
              />
              <span>Read aloud</span>
            </label>
            <button className="btn secondary" onClick={onStop} disabled={!isLoading}>
              Stop
            </button>
            <a className="link" href={apiPaths.health} target="_blank" rel="noreferrer">
              API health
            </a>
            <div className="kbd">Ctrl/⌘ + Enter</div>
          </div>
          {voiceError ? <div style={{ marginTop: 10, color: "rgba(239,68,68,0.95)", fontSize: 12 }}>{voiceError}</div> : null}
        </footer>
      </div>
    </div>
  );
}

