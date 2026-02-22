import { UsagePayload } from "../state/types";

interface Props {
  usage: UsagePayload | null;
  model?: string;
}

// Pricing per million tokens.
// Sources:
//   Anthropic: https://www.anthropic.com/pricing
//   OpenAI:    https://openai.com/api/pricing/
//   Gemini:    https://ai.google.dev/gemini-api/docs/pricing
interface ModelPrice {
  input: number; output: number; cacheRead: number; cacheWrite: number;
}

// All values are USD per 1M tokens.
// Lookup: exact match first, then longest prefix match (see getPrice).
// Unknown models fall back to DEFAULT_PRICE (Sonnet 4.6).
const PRICING: Record<string, ModelPrice> = {
  // Anthropic — https://www.anthropic.com/pricing (Feb 2026)
  //   cacheRead = prompt caching read, cacheWrite = prompt caching write
  "claude-sonnet-4-6":          { input: 3.00,  output: 15.00, cacheRead: 0.30,  cacheWrite: 3.75 },  // $3/$15, cache 90% off read, 1.25× write
  "claude-opus-4-6":            { input: 15.00, output: 75.00, cacheRead: 1.50,  cacheWrite: 18.75 }, // $15/$75
  "claude-haiku-4-5":           { input: 0.80,  output: 4.00,  cacheRead: 0.08,  cacheWrite: 1.00 },  // $0.80/$4

  // Gemini — https://ai.google.dev/gemini-api/docs/pricing (Feb 2026)
  //   ≤200k context; long-context (>200k) is 2× input, 1.5× output
  "gemini-3.1-pro":             { input: 2.00,  output: 12.00, cacheRead: 0,     cacheWrite: 0 },     // $2/$12
  "gemini-3-flash":             { input: 0.50,  output: 3.00,  cacheRead: 0,     cacheWrite: 0 },     // $0.50/$3

  // OpenAI — https://openai.com/api/pricing/ (Feb 2026)
  //   cacheRead = automatic prompt caching (≥1024 prefix tokens, 75-90% off)
  //   https://platform.openai.com/docs/guides/prompt-caching
  "gpt-5.2":                    { input: 1.75,  output: 14.00, cacheRead: 0.175, cacheWrite: 0 },     // $1.75/$14, cache 90% off
};

function getPrice(model: string | undefined): ModelPrice | null {
  if (!model) return null;
  // Exact match first, then prefix match (e.g. "claude-sonnet-4-6-20251030")
  if (PRICING[model]) return PRICING[model];
  for (const key of Object.keys(PRICING)) {
    if (model.startsWith(key)) return PRICING[key];
  }
  return null;
}

function calcCost(u: UsagePayload, model: string | undefined): number | null {
  const p = getPrice(model);
  if (!p) return null;
  return (
    (u.total_input       * p.input      +
     u.total_output      * p.output     +
     u.total_cache_read  * p.cacheRead  +
     u.total_cache_write * p.cacheWrite) / 1_000_000
  );
}

function fmt(n: number | undefined) {
  return n != null ? n.toLocaleString() : "-";
}

function fmtCost(n: number) {
  if (n < 0.001) return `$${(n * 100).toFixed(4)}¢`;
  return `$${n.toFixed(4)}`;
}

export function UsageSection({ usage, model }: Props) {
  const cost = usage ? calcCost(usage, model) : null;

  return (
    <div className="section">
      <div className="section-title">Token Usage</div>
      <div className="usage-grid">
        <div className="usage-card">
          <div className="label">Input</div>
          <div className="value">{fmt(usage?.input_tokens)}</div>
        </div>
        <div className="usage-card">
          <div className="label">Output</div>
          <div className="value">{fmt(usage?.output_tokens)}</div>
        </div>
        <div className="usage-card">
          <div className="label">Cache Read</div>
          <div className="value">{fmt(usage?.cache_read)}</div>
        </div>
        <div className="usage-card">
          <div className="label">Cache Write</div>
          <div className="value">{fmt(usage?.cache_write)}</div>
        </div>
        <div className="usage-card">
          <div className="label">Total In</div>
          <div className="value total">{fmt(usage?.total_input)}</div>
        </div>
        <div className="usage-card">
          <div className="label">Total Out</div>
          <div className="value total">{fmt(usage?.total_output)}</div>
        </div>
      </div>
      {usage && (
        <div className="usage-cost">
          <span className="usage-cost-label">Session cost</span>
          <span className="usage-cost-value">{cost !== null ? fmtCost(cost) : "N/A"}</span>
        </div>
      )}
    </div>
  );
}
