export interface ToolDef {
  name: string;
  description: string;
  parameters: unknown;
  lazy?: boolean;
}

export interface ConfigPayload {
  system_prompt: string;
  tools: ToolDef[];
  max_iterations: number;
  current_user?: string;
  default_user?: string;
  current_model?: string;
  default_model?: string;
}

export interface ChatPayload {
  role: "user" | "assistant";
  text: string;
  image_data?: string;        // base64
  image_media_type?: string;  // e.g. "image/png"
}

export interface StreamPayload {
  text: string;
}

export interface ToolStartPayload {
  name: string;
  args: unknown;
}

export interface ToolDonePayload {
  name: string;
  duration_ms: number;
  summary?: string;
  is_error?: boolean;
}

export interface UsagePayload {
  input_tokens: number;
  output_tokens: number;
  cache_read: number;
  cache_write: number;
  total_input: number;
  total_output: number;
  total_cache_read: number;
  total_cache_write: number;
}

export interface StatePayload {
  running: boolean;
}

export interface ErrorPayload {
  error: string;
}

export type ToolLogEntry =
  | { status: "pending"; name: string; args: unknown }
  | {
      status: "done";
      name: string;
      args: unknown;
      summary: string;
      is_error: boolean;
      duration_ms: number;
    };

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  imageData?: string;       // base64 data URL src
  imageMediaType?: string;
}

export interface AppState {
  connected: boolean;
  running: boolean;
  config: ConfigPayload | null;
  messages: ChatMessage[];
  streamBuffer: string;
  toolLog: ToolLogEntry[];
  usage: UsagePayload | null;
  errors: string[];
  currentUser: string;
  defaultUser: string;
  currentModel: string;
  defaultModel: string;
}

export const initialState: AppState = {
  connected: false,
  running: false,
  config: null,
  messages: [],
  streamBuffer: "",
  toolLog: [],
  usage: null,
  errors: [],
  currentUser: "",
  defaultUser: "",
  currentModel: "",
  defaultModel: "",
};

export type Action =
  | { type: "connected" }
  | { type: "config"; payload: ConfigPayload }
  | { type: "chat"; payload: ChatPayload }
  | { type: "stream"; payload: StreamPayload }
  | { type: "tool_start"; payload: ToolStartPayload }
  | { type: "tool_done"; payload: ToolDonePayload }
  | { type: "usage"; payload: UsagePayload }
  | { type: "state"; payload: StatePayload }
  | { type: "error"; payload: ErrorPayload }
  | { type: "clear" };

export const EVENT_TYPES = [
  "config",
  "chat",
  "stream",
  "tool_start",
  "tool_done",
  "usage",
  "state",
  "error",
] as const;
