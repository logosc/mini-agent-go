import { useReducer, useCallback, useState, useEffect } from "react";
import "./App.css";
import { initialState } from "./state/types";
import { reducer } from "./state/reducer";
import { useSSE } from "./hooks/useSSE";
import { ChatPanel } from "./components/ChatPanel";
import { DebugPanel } from "./components/DebugPanel";
import { SendPayload } from "./components/ChatInput";

export function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  useSSE(dispatch);

  // pendingUserId / pendingModel are used for the NEXT new session.
  // Seeded from the server on first config event; editable between sessions.
  const [pendingUserId, setPendingUserId] = useState("");
  const [pendingModel, setPendingModel] = useState("");

  useEffect(() => {
    if (pendingUserId === "" && state.currentUser !== "") {
      setPendingUserId(state.currentUser);
    }
  }, [state.currentUser]);

  useEffect(() => {
    if (pendingModel === "" && state.currentModel !== "") {
      setPendingModel(state.currentModel);
    }
  }, [state.currentModel]);

  const handleSend = useCallback(
    async ({ text, imageData, imageMediaType }: SendPayload) => {
      await fetch("/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          image_data: imageData,
          image_media_type: imageMediaType,
          user_id: pendingUserId || undefined,
          model: pendingModel || undefined,
        }),
      });
    },
    [pendingUserId, pendingModel]
  );

  return (
    <div className="layout">
      <ChatPanel
        connected={state.connected}
        running={state.running}
        messages={state.messages}
        streamBuffer={state.streamBuffer}
        onSend={handleSend}
        pendingUserId={pendingUserId}
        onUserIdChange={setPendingUserId}
        pendingModel={pendingModel}
        onModelChange={setPendingModel}
        defaultModel={state.defaultModel}
      />
      <DebugPanel state={state} />
    </div>
  );
}
