import { useEffect, useRef } from "react";
import { ChatMessage as ChatMessageType } from "../state/types";
import { ChatMessage } from "./ChatMessage";
import { ChatInput, SendPayload } from "./ChatInput";

// Well-known model options shown in the dropdown.
const MODEL_OPTIONS = [
  { value: "claude-sonnet-4-6", label: "Sonnet 4.6" },
  { value: "claude-opus-4-6",   label: "Opus 4.6" },
  { value: "claude-haiku-4-5-20251001", label: "Haiku 4.5" },
  { value: "gemini",            label: "Gemini" },
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

  // User can only change the ID between sessions (not while agent is running
  // or after the first message of the current session).
  const sessionActive = running || messages.length > 0;

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
          <select
            className="header-model-select"
            value={pendingModel || defaultModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={sessionActive}
            title={sessionActive ? "Start a new session to change the model" : "Model for the next session"}
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
            {/* If current value isn't in the list, show it as a custom option */}
            {(pendingModel || defaultModel) && !MODEL_OPTIONS.find((o) => o.value === (pendingModel || defaultModel)) && (
              <option value={pendingModel || defaultModel}>{pendingModel || defaultModel}</option>
            )}
          </select>
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
