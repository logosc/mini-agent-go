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
  const audioSrc = message.audioData
    ? `data:${message.audioMediaType ?? "audio/webm"};base64,${message.audioData}`
    : null;

  const fileSrc = message.fileData
    ? `data:${message.fileMime ?? "application/octet-stream"};base64,${message.fileData}`
    : null;
  const isFileAudio = message.fileMime?.startsWith("audio/");
  const isFileVideo = message.fileMime?.startsWith("video/");

  return (
    <div className={`bubble ${message.role}`}>
      {imgSrc && (
        <img src={imgSrc} alt="attachment" className="bubble-image" />
      )}
      {audioSrc && (
        <audio src={audioSrc} controls className="bubble-audio" />
      )}
      {fileSrc && isFileAudio && (
        <div className="bubble-file">
          <div className="bubble-file-name">{message.fileName}</div>
          <audio src={fileSrc} controls className="bubble-audio" />
        </div>
      )}
      {fileSrc && isFileVideo && (
        <div className="bubble-file">
          <div className="bubble-file-name">{message.fileName}</div>
          <video src={fileSrc} controls className="bubble-video" />
        </div>
      )}
      {message.text && <span>{renderText(message.text)}</span>}
    </div>
  );
}
