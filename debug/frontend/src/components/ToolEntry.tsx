import { ToolLogEntry } from "../state/types";

interface Props {
  entry: ToolLogEntry;
}

function trunc(s: string, n: number) {
  return s.length > n ? s.slice(0, n) + "..." : s;
}

export function ToolEntry({ entry }: Props) {
  const argsStr =
    typeof entry.args === "string"
      ? entry.args
      : JSON.stringify(entry.args);

  return (
    <div className="tool-entry">
      <span className="t-name">{entry.name}</span>{" "}
      {entry.status === "pending" ? (
        <span className="t-pending">running...</span>
      ) : (
        <span className="t-time">{entry.duration_ms}ms</span>
      )}
      <div className="t-args">{trunc(argsStr, 200)}</div>
      {entry.status === "done" && (
        <div className={`t-result${entry.is_error ? " err" : ""}`}>
          → {trunc(entry.summary, 300)}
        </div>
      )}
    </div>
  );
}
