package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// mockLLM returns pre-configured responses in order.
type mockLLM struct {
	callCount int
	responses []Response
	onChat    func(msgs []Message)
}

func (m *mockLLM) Chat(ctx context.Context, messages []Message, tools []ToolDef) (*Response, error) {
	if m.onChat != nil {
		m.onChat(messages)
	}
	if m.callCount >= len(m.responses) {
		return &Response{Content: "Done."}, nil
	}
	resp := m.responses[m.callCount]
	m.callCount++
	return &resp, nil
}

// mockTool is a simple test tool that records calls.
type mockTool struct {
	name    string
	calls   int
	result  *ToolResult
	execErr error
}

func (t *mockTool) Name() string               { return t.name }
func (t *mockTool) Description() string         { return "test tool" }
func (t *mockTool) Parameters() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t *mockTool) Execute(ctx context.Context, state *testState, args json.RawMessage) (*ToolResult, error) {
	t.calls++
	if t.execErr != nil {
		return nil, t.execErr
	}
	if t.result != nil {
		return t.result, nil
	}
	return &ToolResult{Summary: "ok"}, nil
}

type testState struct {
	value string
}

func TestEngineIdleExit(t *testing.T) {
	llm := &mockLLM{responses: []Response{{Content: "Hello!"}}}
	engine := &Engine[*testState]{
		LLM:           llm,
		SystemPrompt:  "You are a test agent.",
		MaxIterations: 10,
	}
	state := &testState{value: "test"}
	err := engine.Run(context.Background(), state, "do something")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
}

func TestEngineMaxIterations(t *testing.T) {
	tool := &mockTool{name: "loop_tool", result: &ToolResult{Summary: "ok"}}
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "c1", Name: "loop_tool", Args: json.RawMessage(`{}`)}}},
			{ToolCalls: []ToolCall{{ID: "c2", Name: "loop_tool", Args: json.RawMessage(`{}`)}}},
			{ToolCalls: []ToolCall{{ID: "c3", Name: "loop_tool", Args: json.RawMessage(`{}`)}}},
		},
	}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		SystemPrompt:  "test",
		MaxIterations: 2,
	}
	err := engine.Run(context.Background(), &testState{}, "test")
	if err == nil {
		t.Fatal("expected max iterations error")
	}
	if !strings.Contains(err.Error(), "max iterations") {
		t.Errorf("error = %v", err)
	}
}

func TestEngineContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	engine := &Engine[*testState]{
		LLM:          &mockLLM{},
		SystemPrompt: "test",
	}
	err := engine.Run(ctx, &testState{}, "test")
	if err == nil {
		t.Fatal("expected cancellation error")
	}
}

func TestEngineUnknownTool(t *testing.T) {
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "nonexistent", Args: json.RawMessage(`{}`)}}},
			{Content: "ok"},
		},
	}
	engine := &Engine[*testState]{
		LLM:           llm,
		SystemPrompt:  "test",
		MaxIterations: 10,
	}
	err := engine.Run(context.Background(), &testState{}, "test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
}

func TestEngineToolExecution(t *testing.T) {
	tool := &mockTool{name: "my_tool", result: &ToolResult{Summary: "done"}}
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "my_tool", Args: json.RawMessage(`{}`)}}},
		},
	}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		SystemPrompt:  "test",
		MaxIterations: 10,
	}
	err := engine.Run(context.Background(), &testState{}, "test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tool.calls != 1 {
		t.Errorf("tool called %d times, want 1", tool.calls)
	}
}

func TestEngineChatMessage(t *testing.T) {
	var sentMessages []string
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error {
			sentMessages = append(sentMessages, text)
			return nil
		},
		ReplyCh: make(chan Reply, 1),
	}
	tool := &mockTool{name: "url_tool", result: &ToolResult{Summary: "ok", ChatMessage: "https://example.com"}}
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "url_tool", Args: json.RawMessage(`{}`)}}},
		},
	}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		Chat:          chat,
		SystemPrompt:  "test",
		MaxIterations: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "test")

	found := false
	for _, msg := range sentMessages {
		if msg == "https://example.com" {
			found = true
		}
	}
	if !found {
		t.Errorf("ChatMessage not sent to user. sent = %v", sentMessages)
	}
}

func TestEngineOnToolDone(t *testing.T) {
	var hookCalls []string
	tool := &mockTool{name: "hook_tool", result: &ToolResult{Summary: "ok"}}
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "hook_tool", Args: json.RawMessage(`{}`)}}},
		},
	}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		SystemPrompt:  "test",
		MaxIterations: 10,
		OnToolDone: func(state *testState, name string, result *ToolResult) {
			hookCalls = append(hookCalls, name)
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "test")

	if len(hookCalls) != 1 || hookCalls[0] != "hook_tool" {
		t.Errorf("OnToolDone calls = %v", hookCalls)
	}
}

func TestEngineBufferedMessages(t *testing.T) {
	var lastMessages []Message
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "slow_tool", Args: json.RawMessage(`{}`)}}},
		},
		onChat: func(msgs []Message) {
			cp := make([]Message, len(msgs))
			copy(cp, msgs)
			lastMessages = cp
		},
	}
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  make(chan Reply, 1),
	}
	chat.BufferMessage("make it about dinosaurs")

	tool := &mockTool{name: "slow_tool", result: &ToolResult{Summary: "ok"}}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		Chat:          chat,
		SystemPrompt:  "test",
		MaxIterations: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "test")

	foundBuffered := false
	foundStatus := false
	for _, msg := range lastMessages {
		if msg.Role == "user" && msg.Content == "make it about dinosaurs" {
			foundBuffered = true
		}
		if msg.Role == "user" && strings.Contains(msg.Content, "[System:") {
			foundStatus = true
		}
	}
	if !foundBuffered {
		t.Error("buffered message not found in LLM history")
	}
	if !foundStatus {
		t.Error("system status message not found in LLM history")
	}
}

func TestEngineOnUsage(t *testing.T) {
	var captured []Usage
	llm := &mockLLM{
		responses: []Response{
			{Content: "hi", Usage: &Usage{InputTokens: 100, OutputTokens: 50}},
		},
	}
	engine := &Engine[*testState]{
		LLM:          llm,
		SystemPrompt: "test",
		OnUsage: func(u Usage) {
			captured = append(captured, u)
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "test")

	if len(captured) == 0 {
		t.Fatal("OnUsage not called")
	}
	if captured[0].InputTokens != 100 {
		t.Errorf("input tokens = %d", captured[0].InputTokens)
	}
}

func TestEngineExpectationMessages(t *testing.T) {
	var sentMessages []string
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error {
			sentMessages = append(sentMessages, text)
			return nil
		},
		ReplyCh: make(chan Reply, 1),
	}
	tool := &mockTool{name: "slow_tool", result: &ToolResult{Summary: "ok"}}
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "slow_tool", Args: json.RawMessage(`{}`)}}},
		},
	}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		Chat:          chat,
		SystemPrompt:  "test",
		MaxIterations: 10,
		ExpectationMessages: map[string]string{
			"slow_tool": "Working on it...",
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "test")

	found := false
	for _, msg := range sentMessages {
		if msg == "Working on it..." {
			found = true
		}
	}
	if !found {
		t.Errorf("expectation message not sent. sent = %v", sentMessages)
	}
}

func TestEngineLazyToolPromotion(t *testing.T) {
	lazyTool := &mockTool{
		name:   "rare_tool",
		result: &ToolResult{Summary: "executed"},
	}
	// Give it a detailed schema so we can verify it's returned on promotion.
	lazyToolWithSchema := &mockToolWithParams{
		mockTool: lazyTool,
		params:   json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`),
	}

	var toolResultMessages []string
	llm := &mockLLM{
		responses: []Response{
			// Turn 1: LLM calls the lazy tool (will be promoted, not executed).
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "rare_tool", Args: json.RawMessage(`{}`)}}},
			// Turn 2: LLM re-calls with proper args (now promoted, will execute).
			{ToolCalls: []ToolCall{{ID: "tc2", Name: "rare_tool", Args: json.RawMessage(`{"query":"test"}`)}}},
			// Turn 3: Done.
			{Content: "All done."},
		},
		onChat: func(msgs []Message) {
			// Capture tool result messages.
			for _, m := range msgs {
				if m.Role == "tool" {
					toolResultMessages = append(toolResultMessages, m.Content)
				}
			}
		},
	}

	engine := &Engine[*testState]{
		LLM:           llm,
		LazyTools:     []Tool[*testState]{lazyToolWithSchema},
		SystemPrompt:  "test",
		MaxIterations: 10,
	}

	err := engine.Run(context.Background(), &testState{}, "test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Tool should only have been executed once (second call, after promotion).
	if lazyTool.calls != 1 {
		t.Errorf("lazy tool executed %d times, want 1", lazyTool.calls)
	}

	// First tool result should contain the parameter schema.
	foundSchema := false
	for _, msg := range toolResultMessages {
		if strings.Contains(msg, `"query"`) && strings.Contains(msg, "Re-call") {
			foundSchema = true
		}
	}
	if !foundSchema {
		t.Errorf("promotion message with schema not found in tool results: %v", toolResultMessages)
	}
}

// TestEngineImageOnlyPlaceholder verifies that an image sent without a caption
// still produces a non-empty placeholder in the conversation history.
// Previously, image-only messages collapsed to empty Content and were lost.
func TestEngineImageOnlyPlaceholder(t *testing.T) {
	var capturedUserMsg Message
	llm := &mockLLM{
		responses: []Response{
			{Content: "Let me help!"},
		},
		onChat: func(msgs []Message) {
			for _, m := range msgs {
				if m.Role == "user" && len(m.ImageData) > 0 {
					capturedUserMsg = m
				}
			}
		},
	}

	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  make(chan Reply, 1),
	}
	// Send an image with no caption.
	chat.ReplyCh <- Reply{ImageData: []byte{0x89, 0x50, 0x4e, 0x47}} // PNG magic bytes

	engine := &Engine[*testState]{
		LLM:             llm,
		Chat:            chat,
		SystemPrompt:    "test",
		MaxIterations:   5,
		IdleTurnsToExit: 2, // allow one WaitForReply before exiting
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "hello")

	if capturedUserMsg.Content == "" {
		t.Fatal("image-only message produced empty Content — would be lost in compaction")
	}
	if capturedUserMsg.Content != "[Photo received]" {
		t.Errorf("got Content %q, want %q", capturedUserMsg.Content, "[Photo received]")
	}
}

// TestEngineImageWithCaptionPlaceholder verifies that an image sent with a
// caption produces the full "[Photo received with caption: ...]" placeholder.
func TestEngineImageWithCaptionPlaceholder(t *testing.T) {
	var capturedUserMsg Message
	llm := &mockLLM{
		responses: []Response{
			{Content: "Got it!"},
		},
		onChat: func(msgs []Message) {
			for _, m := range msgs {
				if m.Role == "user" && len(m.ImageData) > 0 {
					capturedUserMsg = m
				}
			}
		},
	}

	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  make(chan Reply, 1),
	}
	chat.ReplyCh <- Reply{
		Text:      "a sunset",
		ImageData: []byte{0x89, 0x50, 0x4e, 0x47},
	}

	engine := &Engine[*testState]{
		LLM:             llm,
		Chat:            chat,
		SystemPrompt:    "test",
		MaxIterations:   5,
		IdleTurnsToExit: 2, // allow one WaitForReply before exiting
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "hello")

	want := `[Photo received with caption: "a sunset"]`
	if capturedUserMsg.Content != want {
		t.Errorf("got Content %q, want %q", capturedUserMsg.Content, want)
	}
}

// TestEngineBufferedImagePlaceholder verifies that an image-only message
// buffered while a tool was running gets a placeholder rather than empty content.
func TestEngineBufferedImagePlaceholder(t *testing.T) {
	var lastMessages []Message
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "slow_tool", Args: json.RawMessage(`{}`)}}},
		},
		onChat: func(msgs []Message) {
			cp := make([]Message, len(msgs))
			copy(cp, msgs)
			lastMessages = cp
		},
	}

	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  make(chan Reply, 1),
	}
	// Buffer an image-only message (no caption) while the tool runs.
	chat.BufferMessageWithImage("", []byte{0x89, 0x50, 0x4e, 0x47})

	tool := &mockTool{name: "slow_tool", result: &ToolResult{Summary: "ok"}}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		Chat:          chat,
		SystemPrompt:  "test",
		MaxIterations: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "test")

	// The buffered image-only message must appear with a non-empty placeholder.
	foundPlaceholder := false
	for _, msg := range lastMessages {
		if msg.Role == "user" && msg.Content == "[Photo received]" {
			foundPlaceholder = true
		}
	}
	if !foundPlaceholder {
		t.Error("buffered image-only message did not produce a [Photo received] placeholder")
	}
}

// TestEngineAudioOnlyPlaceholder verifies that an audio message sent without a
// caption still produces a non-empty placeholder in the conversation history.
func TestEngineAudioOnlyPlaceholder(t *testing.T) {
	var history []Message
	llm := &mockLLM{
		responses: []Response{
			{Content: "I'll wait for more input."},
			{Content: "done"},
		},
		onChat: func(msgs []Message) {
			for _, m := range msgs {
				if m.Role == "user" && m.Content == "[Voice message received]" {
					cp := make([]Message, len(history)+1)
					copy(cp, history)
					cp[len(history)] = m
					history = cp
				}
			}
		},
	}
	ch := make(chan Reply, 1)
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  ch,
	}

	e := &Engine[*testState]{
		LLM:             llm,
		Chat:            chat,
		IdleTurnsToExit: 2,
	}

	audio := []byte{0x52, 0x49, 0x46, 0x46}
	ch <- Reply{AudioData: audio, AudioMediaType: "audio/wav"}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	state := &testState{}
	_ = e.Run(ctx, state, "start")

	if len(history) == 0 {
		t.Fatal("expected a user message with '[Voice message received]' placeholder")
	}
	if string(history[0].AudioData) != string(audio) {
		t.Error("AudioData not propagated into Message")
	}
	if history[0].AudioMediaType != "audio/wav" {
		t.Errorf("AudioMediaType = %q, want %q", history[0].AudioMediaType, "audio/wav")
	}
}

// TestEngineBufferedAudioPlaceholder verifies that an audio-only message
// buffered while a tool was running gets a placeholder rather than empty content.
func TestEngineBufferedAudioPlaceholder(t *testing.T) {
	var lastMessages []Message
	llm := &mockLLM{
		responses: []Response{
			{ToolCalls: []ToolCall{{ID: "tc1", Name: "slow_tool", Args: json.RawMessage(`{}`)}}},
		},
		onChat: func(msgs []Message) {
			cp := make([]Message, len(msgs))
			copy(cp, msgs)
			lastMessages = cp
		},
	}

	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  make(chan Reply, 1),
	}
	// Buffer an audio-only message (no caption) while the tool runs.
	chat.BufferMessageWithAudio("", []byte{0x52, 0x49, 0x46, 0x46}, "audio/wav")

	tool := &mockTool{name: "slow_tool", result: &ToolResult{Summary: "ok"}}
	engine := &Engine[*testState]{
		LLM:           llm,
		Tools:         []Tool[*testState]{tool},
		Chat:          chat,
		SystemPrompt:  "test",
		MaxIterations: 10,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = engine.Run(ctx, &testState{}, "test")

	// The buffered audio-only message must appear with a non-empty placeholder
	// and have AudioData propagated.
	var found Message
	for _, msg := range lastMessages {
		if msg.Role == "user" && msg.Content == "[Voice message received]" {
			found = msg
			break
		}
	}
	if found.Content == "" {
		t.Error("buffered audio-only message did not produce a [Voice message received] placeholder")
	}
	if string(found.AudioData) != string([]byte{0x52, 0x49, 0x46, 0x46}) {
		t.Error("AudioData not propagated into buffered Message")
	}
}

// mockToolWithParams wraps mockTool to return custom Parameters.
type mockToolWithParams struct {
	*mockTool
	params json.RawMessage
}

func (t *mockToolWithParams) Parameters() json.RawMessage { return t.params }
func (t *mockToolWithParams) Execute(ctx context.Context, state *testState, args json.RawMessage) (*ToolResult, error) {
	return t.mockTool.Execute(ctx, state, args)
}

// TestEngineGeminiAudioToolCall is a full-stack integration test that exercises:
//  1. An audio Reply sent into the engine results in a Gemini streaming request
//     containing an inlineData part with mimeType starting with "audio/".
//  2. The agent correctly executes a tool call returned by the mock Gemini server
//     in response to the audio input.
//
// Flow:
//
//	Request 1 (initial text "process audio"):
//	  Gemini returns plain text "ready" → idleTurns=1 → engine waits for user reply.
//	  The pre-loaded audio reply is consumed immediately.
//
//	Request 2 (contains audio inlineData):
//	  Mock server verifies inlineData with audio/* mimeType is present.
//	  Gemini returns a tool call for "audio_tool" → engine executes audio_tool → idleTurns=0.
//
//	Request 3 (after tool execution):
//	  Gemini returns plain text "done" → idleTurns=1 → engine calls WaitForReply.
//	  No more replies are queued; context (500 ms) cancels → WaitForReply returns error →
//	  engine.Run returns nil (clean exit).
func TestEngineGeminiAudioToolCall(t *testing.T) {
	var (
		capturedAudioPart bool
		callCount         int
	)

	// SSE snippets used by the mock server.
	readySSE := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ready\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n"
	toolCallSSE := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"audio_tool\",\"args\":{}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n"
	doneSSE := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"done\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n"

	// scanForAudio checks whether the Gemini request body contains an inlineData
	// part whose mimeType starts with "audio/".
	scanForAudio := func(body []byte) bool {
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			return false
		}
		contents, _ := req["contents"].([]any)
		for _, c := range contents {
			cm, _ := c.(map[string]any)
			parts, _ := cm["parts"].([]any)
			for _, p := range parts {
				pm, _ := p.(map[string]any)
				if id, ok := pm["inlineData"].(map[string]any); ok {
					if mime, _ := id["mimeType"].(string); strings.HasPrefix(mime, "audio/") {
						return true
					}
				}
			}
		}
		return false
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "text/event-stream")

		body, _ := io.ReadAll(r.Body)

		switch callCount {
		case 1:
			// Initial request — return text so the engine waits for a user reply.
			fmt.Fprint(w, readySSE)
		case 2:
			// Request containing the user's audio — verify inlineData, return tool call.
			if scanForAudio(body) {
				capturedAudioPart = true
			}
			fmt.Fprint(w, toolCallSSE)
		default:
			// After tool execution — return text; engine will block on WaitForReply
			// until the context is cancelled (clean exit).
			fmt.Fprint(w, doneSSE)
		}
	}))
	defer srv.Close()

	// GeminiProvider pointing at the test server.
	provider := NewGeminiProvider(GeminiOptions{
		APIKey:  "test-key",
		Model:   "gemini-test",
		BaseURL: srv.URL,
	})

	// Tool that records calls.
	tool := &mockTool{name: "audio_tool", result: &ToolResult{Summary: "audio processed"}}

	// ChannelChat — pre-load an audio reply so the engine receives it immediately
	// after the first "ready" response (request 1).
	audio := []byte{0x52, 0x49, 0x46, 0x46} // "RIFF" header bytes
	replyCh := make(chan Reply, 1)
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  replyCh,
	}
	replyCh <- Reply{AudioData: audio, AudioMediaType: "audio/wav"}

	engine := &Engine[*testState]{
		LLM:             provider,
		Tools:           []Tool[*testState]{tool},
		Chat:            chat,
		SystemPrompt:    "You are a test agent.",
		MaxIterations:   10,
		IdleTurnsToExit: 2, // needs two idle turns; after request 3 the context cancels cleanly
	}

	// 500 ms is enough for three fast httptest round-trips; the engine blocks on
	// WaitForReply after request 3 and exits cleanly when the context is cancelled.
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	err := engine.Run(ctx, &testState{}, "process audio")
	if err != nil {
		t.Fatalf("engine.Run: %v", err)
	}

	if !capturedAudioPart {
		t.Error("Gemini request did not contain an inlineData part with audio/* mimeType")
	}
	if tool.calls != 1 {
		t.Errorf("audio_tool called %d times, want 1", tool.calls)
	}
}
