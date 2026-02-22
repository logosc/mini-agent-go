package debug

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	agent "github.com/logosc/mini-agent-go"
)

func TestConsoleOnAudioHookFires(t *testing.T) {
	// Minimal engine that exits immediately.
	llm := &stubLLM{}
	eng := &agent.Engine[struct{}]{
		LLM:             llm,
		IdleTurnsToExit: 1,
	}

	var gotAudio []byte
	var gotMIME string

	console := NewConsole(eng, struct{}{})
	console.OnAudio = func(_ struct{}, audio []byte, mimeType, caption string) {
		gotAudio = audio
		gotMIME = mimeType
	}

	// POST /send with audio_data.
	audio := []byte{0x52, 0x49, 0x46, 0x46}
	body, _ := json.Marshal(map[string]string{
		"text":             "hello",
		"audio_data":       base64.StdEncoding.EncodeToString(audio),
		"audio_media_type": "audio/wav",
	})
	req := httptest.NewRequest(http.MethodPost, "/send", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	console.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if string(gotAudio) != string(audio) {
		t.Errorf("OnAudio not called with correct audio bytes")
	}
	if gotMIME != "audio/wav" {
		t.Errorf("OnAudio mimeType = %q, want %q", gotMIME, "audio/wav")
	}
}

// stubLLM returns a single empty response.
type stubLLM struct{}

func (s *stubLLM) Chat(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef) (*agent.Response, error) {
	return &agent.Response{Content: "done"}, nil
}
