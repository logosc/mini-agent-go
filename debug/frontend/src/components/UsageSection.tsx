import { UsagePayload } from "../state/types";

interface Props {
  usage: UsagePayload | null;
  model?: string;
}

// Pricing per million tokens. Source: https://www.anthropic.com/pricing
interface ModelPrice {
  input: number; output: number; cacheRead: number; cacheWrite: number;
}

const PRICING: Record<string, ModelPrice> = {
  "claude-sonnet-4-6":          { input: 3.00, output: 15.00, cacheRead: 0.30, cacheWrite: 3.75 },
  "claude-opus-4-6":            { input: 15.00, output: 75.00, cacheRead: 1.50, cacheWrite: 18.75 },
  "claude-haiku-4-5-20251001":  { input: 0.80, output: 4.00,  cacheRead: 0.08, cacheWrite: 1.00 },
  // Gemini: no cost tracking (pricing varies; treat as $0)
  "gemini":                     { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

const DEFAULT_PRICE = PRICING["claude-sonnet-4-6"];

function getPrice(model: string | undefined): ModelPrice {
  if (!model) return DEFAULT_PRICE;
  // Exact match first, then prefix match (e.g. "claude-sonnet-4-6-20251030")
  if (PRICING[model]) return PRICING[model];
  for (const key of Object.keys(PRICING)) {
    if (model.startsWith(key)) return PRICING[key];
  }
  return DEFAULT_PRICE;
}

function calcCost(u: UsagePayload, model: string | undefined): number {
  const p = getPrice(model);
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
  const isGemini = model?.startsWith("gemini");

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
      {cost !== null && !isGemini && (
        <div className="usage-cost">
          <span className="usage-cost-label">Session cost</span>
          <span className="usage-cost-value">{fmtCost(cost)}</span>
        </div>
      )}
    </div>
  );
}
