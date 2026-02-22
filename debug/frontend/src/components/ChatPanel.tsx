import { useEffect, useRef } from "react";
import { ChatMessage as ChatMessageType } from "../state/types";
import { ChatMessage } from "./ChatMessage";
import { ChatInput, SendPayload } from "./ChatInput";

// Well-known model suggestions shown in the datalist.
// Users can also type any model ID manually.
const MODEL_SUGGESTIONS = [
  "claude-sonnet-4-6",
  "claude-opus-4-6",
  "gpt-5.2-chat-latest",
  "gemini-3.1-pro-preview",
  "gemini-3-flash-preview",
];

interface Props {
  connected: boolean;
  running: boolean;
  messages: ChatMessageType[];
  streamBuffer: string;
  onSend: (payload: SendPayload) => void;
  pendingUserId: string;
  onUserIdChange: (id: string) => void;
  pendingModel: string;
  onModelChange: (model: string) => void;
  defaultModel: string;
  pendingApiKey: string;
  onApiKeyChange: (key: string) => void;
  rememberApiKey: boolean;
  onRememberApiKeyChange: (remember: boolean) => void;
  pendingBaseUrl: string;
  onBaseUrlChange: (url: string) => void;
}

export function ChatPanel({
  connected,
  running,
  messages,
  streamBuffer,
  onSend,
  pendingUserId,
  onUserIdChange,
  pendingModel,
  onModelChange,
  defaultModel,
  pendingApiKey,
  onApiKeyChange,
  rememberApiKey,
  onRememberApiKeyChange,
  pendingBaseUrl,
  onBaseUrlChange,
}: Props) {
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = chatRef.current;
    if (el) {
      requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
      });
    }
  }, [messages, streamBuffer]);

  function statusText() {
    if (!connected) return "Connecting...";
    if (running) return "Agent running";
    return "Ready";
  }

  // Settings are only locked while the agent is actively running.
  const sessionActive = running;

  return (
    <div className="left">
      <div className="header">
        <div className={`dot${running ? " on" : ""}`} />
        <span className="header-text">{statusText()}</span>
        <div className="header-user">
          <span className="header-user-label">user:</span>
          <input
            className="header-user-input"
            value={pendingUserId}
            onChange={(e) => onUserIdChange(e.target.value)}
            disabled={sessionActive}
            placeholder="user id"
            title={sessionActive ? "Start a new session to change the user" : "User ID for the next session"}
          />
        </div>
        <div className="header-user">
          <span className="header-user-label">model:</span>
          <input
            className="header-user-input header-model-input"
            list="model-suggestions"
            value={pendingModel || defaultModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={sessionActive}
            placeholder="model id"
            title={sessionActive ? "Start a new session to change the model" : "Model for the next session"}
          />
          <datalist id="model-suggestions">
            {MODEL_SUGGESTIONS.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </div>
      </div>
      <div className="settings-bar">
        <div className="settings-field">
          <span className="header-user-label">key:</span>
          <input
            className="header-user-input settings-key-input"
            type="password"
            value={pendingApiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            disabled={sessionActive}
            placeholder="API key"
            autoComplete="off"
            title={sessionActive ? "Start a new session to change the API key" : "API key (not saved unless 'remember' is checked)"}
          />
          <label className="remember-label" title="Persist API key in localStorage">
            <input
              type="checkbox"
              checked={rememberApiKey}
              onChange={(e) => onRememberApiKeyChange(e.target.checked)}
            />
            remember
          </label>
        </div>
        <div className="settings-field">
          <span className="header-user-label">url:</span>
          <input
            className="header-user-input settings-url-input"
            value={pendingBaseUrl}
            onChange={(e) => onBaseUrlChange(e.target.value)}
            disabled={sessionActive}
            placeholder="base URL (optional)"
            title={sessionActive ? "Start a new session to change the base URL" : "Custom base URL (saved in browser)"}
          />
        </div>
      </div>
      <div className="messages" ref={chatRef}>
        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}
        {streamBuffer && (
          <div className="bubble assistant">{streamBuffer}</div>
        )}
      </div>
      <ChatInput onSend={onSend} />
    </div>
  );
}
