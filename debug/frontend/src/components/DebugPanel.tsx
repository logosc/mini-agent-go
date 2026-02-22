import { AppState } from "../state/types";
import { ConfigSection } from "./ConfigSection";
import { UsageSection } from "./UsageSection";
import { ToolLogSection } from "./ToolLogSection";
import { ErrorList } from "./ErrorList";

interface Props {
  state: AppState;
}

export function DebugPanel({ state }: Props) {
  return (
    <div className="right">
      {state.config && <ConfigSection config={state.config} />}
      <UsageSection usage={state.usage} model={state.currentModel || undefined} />
      <ToolLogSection toolLog={state.toolLog} />
      <ErrorList errors={state.errors} />
    </div>
  );
}
