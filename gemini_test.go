package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGeminiProviderChat(t *testing.T) {
	// Mock Gemini response with a function call.
	resp := geminiResponse{
		Candidates: []geminiCandidate{{
			Content: geminiContent{
				Role: "model",
				Parts: []geminiPart{
					{
						FunctionCall: &geminiFunctionCall{
							Name: "get_weather",
							Args: map[string]interface{}{"city": "London"},
						},
						ThoughtSignature: "sig123",
					},
				},
			},
			FinishReason: "STOP",
		}},
		UsageMetadata: &geminiUsage{
			PromptTokenCount:     10,
			CandidatesTokenCount: 20,
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := NewGeminiProvider(GeminiOptions{
		APIKey:  "test-key",
		Model:   "gemini-2.0-flash",
		BaseURL: srv.URL,
	})

	messages := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "What is the weather in London?"},
	}
	tools := []ToolDef{
		{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		},
	}

	result, err := p.Chat(context.Background(), messages, tools)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
	tc := result.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("tool call name = %q, want %q", tc.Name, "get_weather")
	}
	if tc.ID != "get_weather" {
		t.Errorf("tool call ID = %q, want %q", tc.ID, "get_weather")
	}
	if tc.ThoughtSignature != "sig123" {
		t.Errorf("thought signature = %q, want %q", tc.ThoughtSignature, "sig123")
	}

	var args map[string]interface{}
	if err := json.Unmarshal(tc.Args, &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["city"] != "London" {
		t.Errorf("args[city] = %v, want London", args["city"])
	}

	if result.Usage == nil {
		t.Fatal("expected usage, got nil")
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("input tokens = %d, want 10", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 20 {
		t.Errorf("output tokens = %d, want 20", result.Usage.OutputTokens)
	}
}

func TestGeminiProviderTextResponse(t *testing.T) {
	// Mock Gemini response with a text-only reply.
	resp := geminiResponse{
		Candidates: []geminiCandidate{{
			Content: geminiContent{
				Role: "model",
				Parts: []geminiPart{
					{Text: "Hello, world!"},
				},
			},
			FinishReason: "STOP",
		}},
		UsageMetadata: &geminiUsage{
			PromptTokenCount:     5,
			CandidatesTokenCount: 3,
		},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := NewGeminiProvider(GeminiOptions{
		APIKey:  "test-key",
		Model:   "gemini-2.0-flash",
		BaseURL: srv.URL,
	})

	messages := []Message{
		{Role: "user", Content: "Say hello"},
	}

	result, err := p.Chat(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if result.Content != "Hello, world!" {
		t.Errorf("content = %q, want %q", result.Content, "Hello, world!")
	}
	if len(result.ToolCalls) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(result.ToolCalls))
	}
	if result.Usage == nil {
		t.Fatal("expected usage, got nil")
	}
	if result.Usage.InputTokens != 5 {
		t.Errorf("input tokens = %d, want 5", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 3 {
		t.Errorf("output tokens = %d, want 3", result.Usage.OutputTokens)
	}
}
