export type SseEvent = { event: string; data: string };

// Minimal SSE parser for fetch() streaming responses.
export async function* parseSseStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SseEvent, void, void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line.
    while (true) {
      const idx = buffer.indexOf("\n\n");
      if (idx === -1) break;
      const rawEvent = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      let event = "message";
      const dataLines: string[] = [];

      for (const line of rawEvent.split("\n")) {
        const trimmed = line.replace(/\r$/, "");
        if (!trimmed) continue;
        if (trimmed.startsWith("event:")) {
          event = trimmed.slice("event:".length).trim() || "message";
        } else if (trimmed.startsWith("data:")) {
          dataLines.push(trimmed.slice("data:".length).trimStart());
        }
      }

      yield { event, data: dataLines.join("\n") };
    }
  }
}

