import { ChatMessage as ChatMessageType } from "../state/types";

interface Props {
  message: ChatMessageType;
}

const URL_RE = /(https?:\/\/[^\s]+)/g;

function renderText(text: string) {
  const parts = text.split(URL_RE);
  return parts.map((part, i) =>
    URL_RE.test(part) ? (
      <a key={i} href={part} target="_blank" rel="noopener noreferrer">
        {part}
      </a>
    ) : (
      part
    )
  );
}

export function ChatMessage({ message }: Props) {
  const imgSrc = message.imageData
    ? `data:${message.imageMediaType ?? "image/png"};base64,${message.imageData}`
    : null;

  return (
    <div className={`bubble ${message.role}`}>
      {imgSrc && (
        <img src={imgSrc} alt="attachment" className="bubble-image" />
      )}
      {message.text && <span>{renderText(message.text)}</span>}
    </div>
  );
}
