import { useReducer, useCallback, useState, useEffect } from "react";
import "./App.css";
import { initialState } from "./state/types";
import { reducer } from "./state/reducer";
import { useSSE } from "./hooks/useSSE";
import { ChatPanel } from "./components/ChatPanel";
import { DebugPanel } from "./components/DebugPanel";
import { SendPayload } from "./components/ChatInput";

const LS_KEY_PREFIX = "mini-agent-console:";

function loadSetting(key: string): string {
  try { return localStorage.getItem(LS_KEY_PREFIX + key) ?? ""; }
  catch { return ""; }
}

function saveSetting(key: string, value: string) {
  try { localStorage.setItem(LS_KEY_PREFIX + key, value); }
  catch { /* ignore quota errors */ }
}

export function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  useSSE(dispatch);

  // Settings are persisted in localStorage and editable between sessions.
  // API key is NOT persisted unless the user explicitly opts in.
  const [pendingUserId, setPendingUserId] = useState(() => loadSetting("user_id"));
  const [pendingModel, setPendingModel] = useState(() => loadSetting("model"));
  const [rememberApiKey, setRememberApiKey] = useState(() => loadSetting("remember_api_key") === "true");
  const [pendingApiKey, setPendingApiKey] = useState(() =>
    loadSetting("remember_api_key") === "true" ? loadSetting("api_key") : ""
  );
  const [pendingBaseUrl, setPendingBaseUrl] = useState(() => loadSetting("base_url"));

  // Seed from server defaults on first config, but only if no localStorage value.
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

  // Persist settings changes to localStorage.
  const setUserId = useCallback((v: string) => { setPendingUserId(v); saveSetting("user_id", v); }, []);
  const setModel = useCallback((v: string) => { setPendingModel(v); saveSetting("model", v); }, []);
  const setApiKey = useCallback((v: string) => {
    setPendingApiKey(v);
    if (rememberApiKey) saveSetting("api_key", v);
  }, [rememberApiKey]);
  const setRememberKey = useCallback((v: boolean) => {
    setRememberApiKey(v);
    saveSetting("remember_api_key", v ? "true" : "false");
    if (v) {
      saveSetting("api_key", pendingApiKey);
    } else {
      saveSetting("api_key", "");
    }
  }, [pendingApiKey]);
  const setBaseUrl = useCallback((v: string) => { setPendingBaseUrl(v); saveSetting("base_url", v); }, []);

  const handleSend = useCallback(
    async ({ text, imageData, imageMediaType }: SendPayload) => {
      // Only include credentials and session config when starting a new
      // session (agent not currently running). The backend ignores these
      // fields for follow-up messages anyway.
      const isNewSession = !state.running;
      await fetch("/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          image_data: imageData,
          image_media_type: imageMediaType,
          ...(isNewSession && {
            user_id: pendingUserId || undefined,
            model: pendingModel || undefined,
            api_key: pendingApiKey || undefined,
            base_url: pendingBaseUrl || undefined,
          }),
        }),
      });
    },
    [state.running, pendingUserId, pendingModel, pendingApiKey, pendingBaseUrl]
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
        onUserIdChange={setUserId}
        pendingModel={pendingModel}
        onModelChange={setModel}
        defaultModel={state.defaultModel}
        pendingApiKey={pendingApiKey}
        onApiKeyChange={setApiKey}
        rememberApiKey={rememberApiKey}
        onRememberApiKeyChange={setRememberKey}
        pendingBaseUrl={pendingBaseUrl}
        onBaseUrlChange={setBaseUrl}
      />
      <DebugPanel state={state} />
    </div>
  );
}
