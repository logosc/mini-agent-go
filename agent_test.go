package agent

import (
	"context"
	"encoding/json"
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

// mockToolWithParams wraps mockTool to return custom Parameters.
type mockToolWithParams struct {
	*mockTool
	params json.RawMessage
}

func (t *mockToolWithParams) Parameters() json.RawMessage { return t.params }
func (t *mockToolWithParams) Execute(ctx context.Context, state *testState, args json.RawMessage) (*ToolResult, error) {
	return t.mockTool.Execute(ctx, state, args)
}
