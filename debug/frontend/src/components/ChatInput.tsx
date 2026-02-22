import { useState, useRef, KeyboardEvent, ChangeEvent } from "react";

export interface SendPayload {
  text: string;
  imageData?: string;        // base64
  imageMediaType?: string;
  audioData?: string;        // base64
  audioMediaType?: string;
}

interface Props {
  onSend: (payload: SendPayload) => void;
}

export function ChatInput({ onSend }: Props) {
  const [value, setValue] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [imageData, setImageData] = useState<string | null>(null);
  const [imageMediaType, setImageMediaType] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [sendingAudio, setSendingAudio] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  function handleFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataURL = reader.result as string;
      // dataURL = "data:<mime>;base64,<data>"
      const comma = dataURL.indexOf(",");
      const meta = dataURL.slice(5, comma); // e.g. "image/png;base64"
      const mime = meta.split(";")[0] ?? "image/png";
      const b64 = dataURL.slice(comma + 1);
      setPreview(dataURL);
      setImageData(b64);
      setImageMediaType(mime);
    };
    reader.readAsDataURL(file);
    // reset so re-selecting same file fires change
    e.target.value = "";
  }

  function clearImage() {
    setPreview(null);
    setImageData(null);
    setImageMediaType(null);
  }

  async function toggleRecording() {
    if (recording) {
      setRecording(false); // immediate visual feedback
      mediaRecorderRef.current?.stop();
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      alert("Microphone access denied.");
      return;
    }
    const mr = new MediaRecorder(stream);
    audioChunksRef.current = [];
    mr.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };
    mr.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setSendingAudio(true);
      const mimeType = mr.mimeType.split(";")[0] || "audio/webm";
      const blob = new Blob(audioChunksRef.current, { type: mimeType });
      const ab = await blob.arrayBuffer();
      const bytes = new Uint8Array(ab);
      let binary = "";
      const chunk = 8192;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      const b64 = btoa(binary);
      onSend({ text: value.trim(), audioData: b64, audioMediaType: mimeType });
      setValue("");
      setSendingAudio(false);
    };
    mr.start();
    mediaRecorderRef.current = mr;
    setRecording(true);
  }

  function handleSend() {
    const text = value.trim();
    if (!text && !imageData) return;
    onSend({
      text,
      imageData: imageData ?? undefined,
      imageMediaType: imageMediaType ?? undefined,
    });
    setValue("");
    clearImage();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleSend();
  }

  return (
    <div className="input-area">
      {preview && (
        <div className="image-preview">
          <img src={preview} alt="preview" />
          <button className="image-preview-remove" onClick={clearImage} title="Remove">✕</button>
        </div>
      )}
      <div className="input-row">
        <button
          className="attach-btn"
          onClick={() => fileRef.current?.click()}
          title="Attach image"
        >
          📎
        </button>
        <button
          className={`attach-btn mic-btn${recording ? " mic-btn--recording" : ""}`}
          onClick={toggleRecording}
          disabled={sendingAudio}
          title={recording ? "Stop recording" : sendingAudio ? "Sending…" : "Record voice message"}
        >
          {recording ? "⏹" : sendingAudio ? "⏳" : "🎙"}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={handleFile}
        />
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}
