import { Dispatch, useEffect } from "react";
import { Action, EVENT_TYPES } from "../state/types";

export function useSSE(dispatch: Dispatch<Action>) {
  useEffect(() => {
    const sse = new EventSource("/events");

    sse.onopen = () => {
      dispatch({ type: "clear" });
      dispatch({ type: "connected" });
    };

    for (const evt of EVENT_TYPES) {
      sse.addEventListener(evt, (e) => {
        const payload = JSON.parse((e as MessageEvent).data);
        dispatch({ type: evt, payload } as Action);
      });
    }

    sse.onerror = () => {
      // EventSource will auto-reconnect; we'll reset state on next open
    };

    return () => sse.close();
  }, [dispatch]);
}
