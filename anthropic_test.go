package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAnthropicProviderChat(t *testing.T) {
	// Serve a tool_use response and verify headers + parsing.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify API key header.
		if got := r.Header.Get("x-api-key"); got != "test-key" {
			t.Errorf("x-api-key = %q, want %q", got, "test-key")
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version = %q, want %q", got, "2023-06-01")
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want %q", got, "application/json")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":   "msg_123",
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{
					"type":  "tool_use",
					"id":    "toolu_01",
					"name":  "get_weather",
					"input": map[string]any{"city": "London"},
				},
			},
			"stop_reason": "tool_use",
			"usage": map[string]any{
				"input_tokens":  100,
				"output_tokens": 50,
			},
		})
	}))
	defer ts.Close()

	p := NewAnthropicProvider(AnthropicOptions{
		APIKey:  "test-key",
		BaseURL: ts.URL,
		Model:   "claude-sonnet-4-20250514",
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
	if tc.ID != "toolu_01" {
		t.Errorf("ToolCall.ID = %q, want %q", tc.ID, "toolu_01")
	}
	if tc.Name != "get_weather" {
		t.Errorf("ToolCall.Name = %q, want %q", tc.Name, "get_weather")
	}

	if resp.Usage == nil {
		t.Fatal("Usage is nil")
	}
	if resp.Usage.InputTokens != 100 {
		t.Errorf("InputTokens = %d, want 100", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 50 {
		t.Errorf("OutputTokens = %d, want 50", resp.Usage.OutputTokens)
	}
}

func TestAnthropicProviderTextResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":   "msg_456",
			"type": "message",
			"role": "assistant",
			"content": []map[string]any{
				{"type": "text", "text": "Hello, world!"},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
			},
		})
	}))
	defer ts.Close()

	p := NewAnthropicProvider(AnthropicOptions{
		APIKey:  "test-key",
		BaseURL: ts.URL,
		Model:   "claude-sonnet-4-20250514",
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
}

func TestAnthropicMergesConsecutiveUserMessages(t *testing.T) {
	var receivedBody anthropicRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":          "msg_789",
			"type":        "message",
			"role":        "assistant",
			"content":     []map[string]any{{"type": "text", "text": "ok"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 5, "output_tokens": 2},
		})
	}))
	defer ts.Close()

	p := NewAnthropicProvider(AnthropicOptions{
		APIKey:  "test-key",
		BaseURL: ts.URL,
		Model:   "claude-sonnet-4-20250514",
	})

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "first"},
		{Role: "user", Content: "second"},
		{Role: "user", Content: "third"},
	}, nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	// All three user messages should be merged into a single API message.
	if len(receivedBody.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(receivedBody.Messages))
	}
	if receivedBody.Messages[0].Role != "user" {
		t.Errorf("Messages[0].Role = %q, want %q", receivedBody.Messages[0].Role, "user")
	}
	// The merged message should have 3 content blocks.
	if len(receivedBody.Messages[0].Content) != 3 {
		t.Errorf("Content blocks = %d, want 3", len(receivedBody.Messages[0].Content))
	}
}

func TestAnthropicProviderRejectsAudio(t *testing.T) {
	p := NewAnthropicProvider(AnthropicOptions{APIKey: "test", Model: "claude-3-5-sonnet-20241022"})
	msgs := []Message{
		{Role: "user", Content: "here", AudioData: []byte{0x00}, AudioMediaType: "audio/wav"},
	}
	_, err := p.Chat(context.Background(), msgs, nil)
	if err == nil {
		t.Fatal("expected error for audio input on Anthropic provider")
	}
	if !strings.Contains(err.Error(), "audio") {
		t.Errorf("error %q should mention 'audio'", err)
	}
}
