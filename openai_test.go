package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenAIProviderTextResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer test-key")
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want %q", got, "application/json")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-123",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Hello, world!",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
			},
		})
	}))
	defer ts.Close()

	p := NewOpenAICompatibleProvider(OpenAICompatibleOptions{
		APIKey:  "test-key",
		BaseURL: ts.URL,
		Model:   "gpt-4.1",
	})

	resp, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hi"},
	}, nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if resp.Content != "Hello, world!" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello, world!")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls len = %d, want 0", len(resp.ToolCalls))
	}
	if resp.Usage == nil {
		t.Fatal("Usage is nil")
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", resp.Usage.OutputTokens)
	}
}

func TestOpenAIProviderToolCallResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-456",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": nil,
						"tool_calls": []map[string]any{
							{
								"id":   "call_abc123",
								"type": "function",
								"function": map[string]any{
									"name":      "get_weather",
									"arguments": `{"city":"London"}`,
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     100,
				"completion_tokens": 50,
			},
		})
	}))
	defer ts.Close()

	p := NewOpenAICompatibleProvider(OpenAICompatibleOptions{
		APIKey:  "test-key",
		BaseURL: ts.URL,
		Model:   "gpt-4.1",
	})

	resp, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "What is the weather?"},
	}, []ToolDef{
		{Name: "get_weather", Description: "Get weather", Parameters: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_abc123" {
		t.Errorf("ToolCall.ID = %q, want %q", tc.ID, "call_abc123")
	}
	if tc.Name != "get_weather" {
		t.Errorf("ToolCall.Name = %q, want %q", tc.Name, "get_weather")
	}

	var args map[string]string
	if err := json.Unmarshal(tc.Args, &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["city"] != "London" {
		t.Errorf("args[city] = %q, want %q", args["city"], "London")
	}
}

func TestOpenAIProviderAuthorizationHeader(t *testing.T) {
	var gotAuth string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-789",
			"choices": []map[string]any{{
				"index":         0,
				"message":       map[string]any{"role": "assistant", "content": "ok"},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer ts.Close()

	p := NewOpenAICompatibleProvider(OpenAICompatibleOptions{
		APIKey:  "sk-secret-key-42",
		BaseURL: ts.URL,
		Model:   "gpt-4.1",
	})

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "test"},
	}, nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if gotAuth != "Bearer sk-secret-key-42" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer sk-secret-key-42")
	}
}

func TestOpenAIProviderImageMessage(t *testing.T) {
	var receivedBody openaiRequest
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-img",
			"choices": []map[string]any{{
				"index":         0,
				"message":       map[string]any{"role": "assistant", "content": "I see an image"},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{"prompt_tokens": 20, "completion_tokens": 5},
		})
	}))
	defer ts.Close()

	p := NewOpenAICompatibleProvider(OpenAICompatibleOptions{
		APIKey:  "test-key",
		BaseURL: ts.URL,
		Model:   "gpt-4.1",
	})

	// JPEG magic bytes: FF D8
	jpegData := []byte{0xFF, 0xD8, 0x01, 0x02}

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "What's in this image?", ImageData: jpegData},
	}, nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if len(receivedBody.Messages) < 1 {
		t.Fatal("expected at least 1 message")
	}

	// The user message content should be an array (multipart).
	userMsg := receivedBody.Messages[0]
	contentBytes, err := json.Marshal(userMsg.Content)
	if err != nil {
		t.Fatalf("marshal content: %v", err)
	}

	var parts []openaiContentPart
	if err := json.Unmarshal(contentBytes, &parts); err != nil {
		t.Fatalf("unmarshal content parts: %v", err)
	}

	if len(parts) != 2 {
		t.Fatalf("content parts len = %d, want 2", len(parts))
	}
	if parts[0].Type != "text" {
		t.Errorf("parts[0].Type = %q, want %q", parts[0].Type, "text")
	}
	if parts[1].Type != "image_url" {
		t.Errorf("parts[1].Type = %q, want %q", parts[1].Type, "image_url")
	}
	if parts[1].ImageURL == nil {
		t.Fatal("parts[1].ImageURL is nil")
	}
	if got := parts[1].ImageURL.URL; len(got) < 20 || got[:15] != "data:image/jpeg" {
		t.Errorf("image URL prefix = %q, want data:image/jpeg...", got[:min(30, len(got))])
	}
}
