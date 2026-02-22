import { AppState, Action, initialState, ToolLogEntry } from "./types";

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "connected":
      return { ...state, connected: true };

    case "clear":
      return { ...initialState, connected: state.connected };

    case "config":
      return {
        ...state,
        config: action.payload,
        currentUser: action.payload.current_user ?? state.currentUser,
        defaultUser: action.payload.default_user ?? state.defaultUser,
        currentModel: action.payload.current_model ?? state.currentModel,
        defaultModel: action.payload.default_model ?? state.defaultModel,
      };

    case "chat": {
      const { role, text, image_data, image_media_type, audio_data, audio_media_type, file_data, file_mime, file_name } = action.payload;
      // If assistant message arrives after streaming, finalize the stream bubble
      if (role === "assistant" && state.streamBuffer !== "") {
        return {
          ...state,
          streamBuffer: "",
          messages: [
            ...state.messages,
            { role: "assistant", text: state.streamBuffer },
          ],
        };
      }
      return {
        ...state,
        messages: [
          ...state.messages,
          { role, text, imageData: image_data, imageMediaType: image_media_type,
            audioData: audio_data, audioMediaType: audio_media_type,
            fileData: file_data, fileMime: file_mime, fileName: file_name },
        ],
      };
    }

    case "stream":
      return {
        ...state,
        streamBuffer: state.streamBuffer + action.payload.text,
      };

    case "tool_start": {
      const entry: ToolLogEntry = {
        status: "pending",
        name: action.payload.name,
        args: action.payload.args,
      };
      return { ...state, toolLog: [...state.toolLog, entry] };
    }

    case "tool_done": {
      const { name, duration_ms, summary, is_error } = action.payload;
      // Find last pending entry with matching name
      const log = [...state.toolLog];
      for (let i = log.length - 1; i >= 0; i--) {
        const entry = log[i];
        if (entry && entry.status === "pending" && entry.name === name) {
          log[i] = {
            status: "done",
            name: entry.name,
            args: entry.args,
            summary: summary ?? "",
            is_error: is_error ?? false,
            duration_ms,
          };
          break;
        }
      }
      return { ...state, toolLog: log };
    }

    case "usage":
      return { ...state, usage: action.payload };

    case "state":
      return { ...state, running: action.payload.running };

    case "error":
      return { ...state, errors: [...state.errors, action.payload.error] };

    default:
      return state;
  }
}
