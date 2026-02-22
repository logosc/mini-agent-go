import { ConfigPayload } from "../state/types";

interface Props {
  config: ConfigPayload;
}

export function ConfigSection({ config }: Props) {
  return (
    <div className="section">
      <div className="section-title">Configuration</div>
      <div
        className="section-title"
        style={{ marginTop: 0, fontSize: 10, color: "var(--accent)" }}
      >
        System Prompt
      </div>
      <div className="config-box">{config.system_prompt || "(none)"}</div>
      {config.tools.length > 0 && (
        <>
          <div
            className="section-title"
            style={{ fontSize: 10, color: "var(--accent)" }}
          >
            Tools
          </div>
          {config.tools.map((t) => (
            <div key={t.name} className="tool-item">
              <span className="name">{t.name}</span>
              {t.lazy && <span className="lazy"> [lazy]</span>}
              {" — "}
              <span className="desc">{t.description}</span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
