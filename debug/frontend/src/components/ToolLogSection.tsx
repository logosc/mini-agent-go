import { ToolLogEntry } from "../state/types";
import { ToolEntry } from "./ToolEntry";

interface Props {
  toolLog: ToolLogEntry[];
}

export function ToolLogSection({ toolLog }: Props) {
  return (
    <div className="section">
      <div className="section-title">
        Tool Log{" "}
        <span className="count">{toolLog.length}</span>
      </div>
      {toolLog.map((entry, i) => (
        <ToolEntry key={i} entry={entry} />
      ))}
    </div>
  );
}
