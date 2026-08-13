import { NextRequest } from "next/server";

export const runtime = "edge";

const SYSTEM_PROMPT = `You are an assistant connected to a real Slack workspace via MCP tools.

Rules you must follow without exception:
1. Answer ONLY using content returned by the search_slack or get_thread tools.
2. Do not use your training data or prior knowledge to answer questions about this Slack workspace.
3. Every factual claim in your answer must be followed by the permalink of the Slack message it came from, formatted as [source](permalink_url).
4. If the tools return no results or nothing relevant, respond with exactly: "I couldn't find anything in Slack about that."
5. Do not speculate, infer, or paraphrase beyond what the messages explicitly say.`;

export async function POST(req: NextRequest) {
  const { messages } = await req.json();

  const response = await fetch(`${process.env.LLM_API_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.LLM_API_KEY}`,
    },
    body: JSON.stringify({
      model: "default",
      stream: true,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        ...messages,
      ],
    }),
  });

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    },
  });
}
