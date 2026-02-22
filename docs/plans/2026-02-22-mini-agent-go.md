# mini-agent-go SDK Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone open-source Go SDK for LLM-driven agent loops at `github.com/logosc/mini-agent-go`.

**Architecture:** Flat Go package (`agent`) with generics (`Engine[S]`). Core types (Message, ToolCall, ToolDef, Response, Usage), LLM provider interface + two implementations (Anthropic, Gemini), chat interface with buffering, and the tool-calling loop. All stdlib — zero external dependencies.

**Tech Stack:** Go 1.23+, stdlib only (`net/http`, `encoding/json`, `bufio`, `sync`)

---

### Task 1: Project scaffold

**Files:**
- Create: `/home/storymaker/mini-agent-go/go.mod`
- Create: `/home/storymaker/mini-agent-go/LICENSE`

**Step 1: Create go.mod**

```
module github.com/logosc/mini-agent-go

go 1.23
```

**Step 2: Create MIT LICENSE**

Standard MIT license, copyright 2026 Logosc.

**Step 3: Commit**

```bash
cd /home/storymaker/mini-agent-go
git init
git add go.mod LICENSE
git commit -m "chore: init mini-agent-go module with MIT license"
```

---

### Task 2: Core types — provider.go

Define `Message`, `ToolCall`, `ToolDef`, `Response` types and the `Usage` struct for token tracking.

**Files:**
- Create: `/home/storymaker/mini-agent-go/provider.go`
- Test: `/home/storymaker/mini-agent-go/provider_test.go`

**Step 1: Write the failing test**

```go
package agent

import (
	"encoding/json"
	"testing"
)

func TestMessageSerialization(t *testing.T) {
	msg := Message{
		Role:    "assistant",
		Content: "Hello",
		ToolCalls: []ToolCall{
			{ID: "tc1", Name: "my_tool", Args: json.RawMessage(`{"key":"val"}`)},
		},
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Role != "assistant" || got.Content != "Hello" {
		t.Errorf("got %+v", got)
	}
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].Name != "my_tool" {
		t.Errorf("tool calls = %+v", got.ToolCalls)
	}
}

func TestUsageFields(t *testing.T) {
	u := Usage{InputTokens: 100, OutputTokens: 50, CacheReadTokens: 10, CacheWriteTokens: 5}
	if u.InputTokens != 100 || u.OutputTokens != 50 {
		t.Errorf("usage = %+v", u)
	}
}

func TestResponseWithUsage(t *testing.T) {
	resp := Response{
		Content: "hi",
		Usage:   &Usage{InputTokens: 100, OutputTokens: 200},
	}
	if resp.Usage.InputTokens != 100 {
		t.Errorf("input tokens = %d", resp.Usage.InputTokens)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /home/storymaker/mini-agent-go && go test -run TestMessage -v`
Expected: FAIL — types not defined

**Step 3: Write provider.go**

```go
package agent

import (
	"context"
	"encoding/json"
)

// Message represents a single message in the agent conversation.
type Message struct {
	Role       string          `json:"role"`                   // "system", "user", "assistant", "tool"
	Content    string          `json:"content"`
	ImageData  []byte          `json:"-"`                      // Optional inline image for user messages.
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"` // set when Role="tool"
}

// ToolCall represents a tool invocation requested by the LLM.
type ToolCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"arguments"`
}

// ToolDef describes a tool available to the LLM.
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"` // JSON Schema
}

// Usage tracks token consumption for an LLM call.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

// Response is what the LLM returns from a Chat call.
type Response struct {
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Usage     *Usage     `json:"usage,omitempty"`
}

// LLMProvider abstracts the LLM used for agent orchestration decisions.
type LLMProvider interface {
	Chat(ctx context.Context, messages []Message, tools []ToolDef) (*Response, error)
}

// StreamingLLMProvider is optionally implemented by providers that support streaming.
type StreamingLLMProvider interface {
	LLMProvider
	StreamChat(ctx context.Context, messages []Message, tools []ToolDef, onText func(chunk string)) (*Response, error)
}
```

**Step 4: Run tests**

Run: `cd /home/storymaker/mini-agent-go && go test -v ./...`
Expected: PASS

**Step 5: Commit**

```bash
git add provider.go provider_test.go
git commit -m "feat: add core types — Message, ToolCall, Response, Usage, LLMProvider"
```

---

### Task 3: Tool types — tool.go

**Files:**
- Create: `/home/storymaker/mini-agent-go/tool.go`
- Test: `/home/storymaker/mini-agent-go/tool_test.go`

**Step 1: Write the failing test**

```go
package agent

import "testing"

func TestToolResultWithError(t *testing.T) {
	r := ToolResult{}.WithError("something broke")
	if !r.IsError {
		t.Error("expected IsError=true")
	}
	if r.Summary != "something broke" {
		t.Errorf("summary = %q", r.Summary)
	}
}

func TestAttachmentFields(t *testing.T) {
	a := Attachment{Type: "file", Data: []byte("pdf"), Name: "story.pdf", Caption: "Your story"}
	if a.Type != "file" || a.Name != "story.pdf" {
		t.Errorf("attachment = %+v", a)
	}
}
```

**Step 2: Run test — expect FAIL**

**Step 3: Write tool.go**

```go
package agent

import (
	"context"
	"encoding/json"
)

// Tool is a capability the agent can invoke. S is the application-defined state type.
type Tool[S any] interface {
	Name() string
	Description() string
	Parameters() json.RawMessage // JSON Schema for arguments
	Execute(ctx context.Context, state S, args json.RawMessage) (*ToolResult, error)
}

// Attachment represents a file or image to send to the user.
type Attachment struct {
	Type    string // "file", "image"
	Data    []byte
	Name    string // filename
	Caption string // optional
}

// ToolResult is returned by tool execution.
type ToolResult struct {
	// Summary is sent to the LLM as the tool response.
	Summary string
	// ChatMessage is sent directly to the user, bypassing the LLM.
	ChatMessage string
	// Attachments are files/images sent to the user.
	Attachments []Attachment
	// IsError indicates the tool failed.
	IsError bool
}

// WithError creates an error ToolResult from a message string.
func (ToolResult) WithError(msg string) *ToolResult {
	return &ToolResult{
		Summary: msg,
		IsError: true,
	}
}
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add tool.go tool_test.go
git commit -m "feat: add Tool[S] interface, ToolResult, Attachment"
```

---

### Task 4: Chat interface — chat.go

Extract `ChatInterface`, `NullChat`, `ChannelChat`, and buffering types. Adapt interface: replace `SendImage`/`SendFile` with `SendAttachment`. Drop `DrainPhotos` (domain-specific — photos are just buffered messages with ImageData).

**Files:**
- Create: `/home/storymaker/mini-agent-go/chat.go`
- Test: `/home/storymaker/mini-agent-go/chat_test.go`

**Step 1: Write the failing tests**

Tests for ChatInterface implementations:

```go
package agent

import (
	"context"
	"testing"
)

func TestNullChatSendDoesNotBlock(t *testing.T) {
	chat := NullChat{}
	if err := chat.Send(context.Background(), "hello"); err != nil {
		t.Fatalf("Send: %v", err)
	}
}

func TestNullChatSendAttachmentDoesNotBlock(t *testing.T) {
	chat := NullChat{}
	if err := chat.SendAttachment(context.Background(), Attachment{Type: "file", Data: []byte("x")}); err != nil {
		t.Fatalf("SendAttachment: %v", err)
	}
}

func TestNullChatWaitForReplyReturnsError(t *testing.T) {
	chat := NullChat{}
	_, err := chat.WaitForReply(context.Background())
	if err == nil {
		t.Fatal("WaitForReply should return error in non-interactive mode")
	}
}

func TestNullChatDrainMessages(t *testing.T) {
	chat := NullChat{}
	if got := chat.DrainMessages(); got != nil {
		t.Errorf("NullChat.DrainMessages() = %v, want nil", got)
	}
}

func TestChannelChatSendAndReply(t *testing.T) {
	ch := make(chan Reply, 1)
	var sent []string
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error {
			sent = append(sent, text)
			return nil
		},
		ReplyCh: ch,
	}

	if err := chat.Send(context.Background(), "What style?"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if len(sent) != 1 || sent[0] != "What style?" {
		t.Errorf("sent = %v", sent)
	}

	ch <- Reply{Text: "watercolor"}
	reply, err := chat.WaitForReply(context.Background())
	if err != nil {
		t.Fatalf("WaitForReply: %v", err)
	}
	if reply.Text != "watercolor" {
		t.Errorf("reply = %q, want %q", reply.Text, "watercolor")
	}
}

func TestChannelChatContextCancellation(t *testing.T) {
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  make(chan Reply),
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := chat.WaitForReply(ctx)
	if err == nil {
		t.Fatal("should return error on cancelled context")
	}
}

func TestChannelChatBufferMessage(t *testing.T) {
	chat := &ChannelChat{ReplyCh: make(chan Reply)}

	chat.BufferMessage("hello")
	chat.BufferMessage("world")

	msgs := chat.DrainMessages()
	if len(msgs) != 2 {
		t.Fatalf("DrainMessages() returned %d, want 2", len(msgs))
	}
	if msgs[0].Text != "hello" {
		t.Errorf("msgs[0].Text = %q", msgs[0].Text)
	}
	if msgs[1].Text != "world" {
		t.Errorf("msgs[1].Text = %q", msgs[1].Text)
	}

	// Second drain should be empty.
	if got := chat.DrainMessages(); len(got) != 0 {
		t.Errorf("second DrainMessages() returned %d, want 0", len(got))
	}
}

func TestChannelChatDrainMessagesEmpty(t *testing.T) {
	chat := &ChannelChat{ReplyCh: make(chan Reply)}
	if got := chat.DrainMessages(); got != nil {
		t.Errorf("DrainMessages() on empty = %v, want nil", got)
	}
}
```

**Step 2: Run tests — expect FAIL**

**Step 3: Write chat.go**

```go
package agent

import (
	"context"
	"fmt"
	"log"
	"sync"
)

// Reply holds the user's response.
type Reply struct {
	Text      string
	ImageData []byte // Optional image data.
}

// BufferedMessage holds a message that arrived while the agent was busy.
type BufferedMessage struct {
	Text      string
	ImageData []byte
}

// ChatInterface bridges the agent with a user for interactive mode.
type ChatInterface interface {
	Send(ctx context.Context, text string) error
	SendAttachment(ctx context.Context, att Attachment) error
	WaitForReply(ctx context.Context) (Reply, error)
	DrainMessages() []BufferedMessage
}

// NullChat is a no-op ChatInterface for headless/background mode.
type NullChat struct{}

func (NullChat) Send(ctx context.Context, text string) error              { return nil }
func (NullChat) SendAttachment(ctx context.Context, att Attachment) error  { return nil }
func (NullChat) WaitForReply(ctx context.Context) (Reply, error) {
	return Reply{}, fmt.Errorf("non-interactive mode: no user available")
}
func (NullChat) DrainMessages() []BufferedMessage { return nil }

// ChannelChat implements ChatInterface using a send function and reply channel.
type ChannelChat struct {
	SendFunc       func(ctx context.Context, text string) error
	AttachmentFunc func(ctx context.Context, att Attachment) error
	ReplyCh        chan Reply

	mu       sync.Mutex
	messages []BufferedMessage
}

func (c *ChannelChat) Send(ctx context.Context, text string) error {
	if c.SendFunc != nil {
		return c.SendFunc(ctx, text)
	}
	return nil
}

func (c *ChannelChat) SendAttachment(ctx context.Context, att Attachment) error {
	if c.AttachmentFunc != nil {
		return c.AttachmentFunc(ctx, att)
	}
	return nil
}

func (c *ChannelChat) WaitForReply(ctx context.Context) (Reply, error) {
	select {
	case msg := <-c.ReplyCh:
		return msg, nil
	case <-ctx.Done():
		return Reply{}, ctx.Err()
	}
}

// BufferMessage stores a text message that arrived while the agent was busy.
// Thread-safe.
func (c *ChannelChat) BufferMessage(text string) {
	c.mu.Lock()
	c.messages = append(c.messages, BufferedMessage{Text: text})
	c.mu.Unlock()
	log.Printf("[agent-chat] buffered message (text=%q), total buffered: %d", text, len(c.messages))
}

// BufferMessageWithImage stores a message with an image that arrived while the agent was busy.
// Thread-safe.
func (c *ChannelChat) BufferMessageWithImage(text string, imageData []byte) {
	c.mu.Lock()
	c.messages = append(c.messages, BufferedMessage{Text: text, ImageData: imageData})
	c.mu.Unlock()
	log.Printf("[agent-chat] buffered message with image (%d bytes), total buffered: %d", len(imageData), len(c.messages))
}

// DrainMessages returns and clears all buffered messages. Thread-safe.
func (c *ChannelChat) DrainMessages() []BufferedMessage {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.messages) == 0 {
		return nil
	}
	msgs := c.messages
	c.messages = nil
	log.Printf("[agent-chat] drained %d buffered message(s)", len(msgs))
	return msgs
}
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add chat.go chat_test.go
git commit -m "feat: add ChatInterface, NullChat, ChannelChat with message buffering"
```

---

### Task 5: Agent engine — agent.go

The main agent loop. Generic `Engine[S]` with tool-calling loop, compaction, idle detection, and hooks.

**Files:**
- Create: `/home/storymaker/mini-agent-go/agent.go`
- Test: `/home/storymaker/mini-agent-go/agent_test.go`

**Step 1: Write the failing tests**

These test the generic loop behavior: idle exit, max iterations, context cancellation, unknown tool handling, buffered message injection, tool result side effects.

```go
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
	onChat    func(msgs []Message) // optional capture callback
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

func (t *mockTool) Name() string                  { return t.name }
func (t *mockTool) Description() string            { return "test tool" }
func (t *mockTool) Parameters() json.RawMessage    { return json.RawMessage(`{"type":"object"}`) }
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
	// LLM returns text-only responses. NullChat WaitForReply fails → exit after first idle.
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
			// No Content — should trigger expectation message.
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
```

**Step 2: Run tests — expect FAIL**

**Step 3: Write agent.go**

Extract the generic loop from `engine.go:Run()`. Key changes from the source:
- `Engine[S any]` struct with public fields instead of `AgentEngine` with private fields
- Remove `buildState`, `buildSystemPrompt`, `buildTaskPrompt`, `buildContextMessage`, `finalizeJob`, `loadCharactersForFollowUp` — all domain-specific
- Remove `WithChat` — not needed; `Engine` is constructed directly
- `Run(ctx, state S, taskPrompt string)` instead of `Run(ctx, *service.Job)`
- Tool lookup by name from `Tools` slice
- `OnToolDone` hook replaces hardcoded checkpoint logic
- `OnUsage` hook replaces hardcoded cost accumulation
- `ExpectationMessages` map replaces hardcoded `toolExpectationMessage`
- `compactMessages` stays — it's generic
- `chatLLM` stays — it's generic

```go
package agent

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"
)

// Engine is an LLM-driven agent loop. S is the application-defined state type.
type Engine[S any] struct {
	// LLM is the language model provider (required).
	LLM LLMProvider
	// Tools available to the agent.
	Tools []Tool[S]
	// Chat bridges the agent with a user. Nil defaults to NullChat.
	Chat ChatInterface
	// SystemPrompt is sent as the first message to the LLM.
	SystemPrompt string
	// MaxIterations limits the number of loop turns. Default: 20.
	MaxIterations int
	// ExpectationMessages maps tool names to user-facing messages sent when the
	// LLM invokes a tool without including any text.
	ExpectationMessages map[string]string
	// OnToolDone is called after each successful tool execution.
	OnToolDone func(state S, name string, result *ToolResult)
	// OnUsage is called with token usage after each LLM response.
	OnUsage func(usage Usage)
}

// Run starts the agent loop. It sends the system prompt and task prompt to the
// LLM, then enters a tool-calling loop until the conversation ends or
// MaxIterations is exceeded.
func (e *Engine[S]) Run(ctx context.Context, state S, taskPrompt string) error {
	chat := e.Chat
	if chat == nil {
		chat = NullChat{}
	}
	maxIter := e.MaxIterations
	if maxIter <= 0 {
		maxIter = 20
	}

	// Build tool lookup and definitions.
	toolMap := make(map[string]Tool[S], len(e.Tools))
	toolDefs := make([]ToolDef, 0, len(e.Tools))
	for _, t := range e.Tools {
		toolMap[t.Name()] = t
		toolDefs = append(toolDefs, ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Parameters(),
		})
	}

	messages := []Message{
		{Role: "system", Content: e.SystemPrompt},
		{Role: "user", Content: taskPrompt},
	}

	idleTurns := 0
	compactions := 0
	maxCompactions := 2

	for i := 0; i < maxIter; i++ {
		// Auto-compact at 75% of iteration limit.
		if i >= maxIter*3/4 && compactions < maxCompactions {
			compacted, err := e.compactMessages(ctx, messages)
			if err != nil {
				log.Printf("[agent] compaction failed: %v", err)
			} else if len(compacted) < len(messages) {
				log.Printf("[agent] compacted %d → %d messages, resetting counter", len(messages), len(compacted))
				messages = compacted
				i = 0
				compactions++
			}
		}

		if err := ctx.Err(); err != nil {
			return fmt.Errorf("agent: cancelled: %w", err)
		}

		resp, err := e.chatLLM(ctx, messages, toolDefs)
		if err != nil {
			return fmt.Errorf("agent: llm chat: %w", err)
		}

		if resp.Usage != nil && e.OnUsage != nil {
			e.OnUsage(*resp.Usage)
		}

		// No tool calls — idle turn.
		if len(resp.ToolCalls) == 0 {
			if resp.Content != "" {
				_ = chat.Send(ctx, resp.Content)
			}

			idleTurns++
			if idleTurns >= 2 {
				return nil
			}

			reply, err := chat.WaitForReply(ctx)
			if err != nil {
				return nil // no user available — finish
			}
			idleTurns = 0

			userContent := reply.Text
			if len(reply.ImageData) > 0 && reply.Text != "" {
				userContent = fmt.Sprintf("[Photo received with caption: %q]", reply.Text)
			}
			messages = append(messages,
				Message{Role: "assistant", Content: resp.Content},
				Message{Role: "user", Content: userContent, ImageData: reply.ImageData},
			)
			continue
		}

		idleTurns = 0

		// Send text or fallback expectation message.
		if resp.Content != "" {
			_ = chat.Send(ctx, resp.Content)
		} else if msg := e.expectationMessage(resp.ToolCalls); msg != "" {
			_ = chat.Send(ctx, msg)
		}

		messages = append(messages, Message{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})

		// Execute each tool call.
		for _, tc := range resp.ToolCalls {
			tool, ok := toolMap[tc.Name]
			if !ok {
				messages = append(messages, Message{
					Role:       "tool",
					ToolCallID: tc.ID,
					Content:    fmt.Sprintf("unknown tool: %s", tc.Name),
				})
				continue
			}

			log.Printf("[agent] executing tool: %s", tc.Name)
			start := time.Now()
			result, err := tool.Execute(ctx, state, tc.Args)
			elapsed := time.Since(start)

			toolResultContent := ""
			if err != nil {
				log.Printf("[agent] tool %s error: %v", tc.Name, err)
				toolResultContent = fmt.Sprintf("tool execution error: %v", err)
			} else {
				log.Printf("[agent] tool %s completed in %v", tc.Name, elapsed)
				toolResultContent = result.Summary
			}

			messages = append(messages, Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    toolResultContent,
			})

			// Drain buffered messages.
			buffered := chat.DrainMessages()
			if len(buffered) > 0 {
				var parts []string
				for _, msg := range buffered {
					parts = append(parts, msg.Text)
				}
				status := fmt.Sprintf("[System: while %s was running (%s), the user sent additional input: %q — treat as extra context/preferences, NOT as a reply or confirmation.]",
					tc.Name, elapsed.Round(time.Second), strings.Join(parts, "; "))
				messages = append(messages, Message{Role: "user", Content: status})
				for _, msg := range buffered {
					messages = append(messages, Message{
						Role:      "user",
						Content:   msg.Text,
						ImageData: msg.ImageData,
					})
				}
			}

			// Handle side effects.
			if err == nil && result != nil {
				if result.ChatMessage != "" {
					_ = chat.Send(ctx, result.ChatMessage)
				}
				for _, att := range result.Attachments {
					_ = chat.SendAttachment(ctx, att)
				}
				if e.OnToolDone != nil {
					e.OnToolDone(state, tc.Name, result)
				}
			}
		}
	}

	return fmt.Errorf("agent: max iterations (%d) exceeded", maxIter)
}

// compactMessages summarizes older messages to free up iteration budget.
func (e *Engine[S]) compactMessages(ctx context.Context, messages []Message) ([]Message, error) {
	const keepRecent = 6
	if len(messages) <= keepRecent+2 {
		return messages, nil
	}

	systemMsg := messages[0]
	middle := messages[1 : len(messages)-keepRecent]
	recent := messages[len(messages)-keepRecent:]

	var sb strings.Builder
	sb.WriteString("Summarize the following conversation history concisely. ")
	sb.WriteString("Include all actionable context, tool results, and user preferences.\n\n")
	for _, m := range middle {
		role := m.Role
		if role == "tool" {
			role = "tool_result"
		}
		content := m.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		fmt.Fprintf(&sb, "[%s]: %s\n", role, content)
	}

	summaryResp, err := e.LLM.Chat(ctx, []Message{
		{Role: "user", Content: sb.String()},
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("compaction LLM call: %w", err)
	}
	if summaryResp.Content == "" {
		return nil, fmt.Errorf("compaction returned empty summary")
	}

	compacted := make([]Message, 0, 2+len(recent))
	compacted = append(compacted, systemMsg)
	compacted = append(compacted, Message{
		Role:    "user",
		Content: fmt.Sprintf("[Conversation summary — earlier messages compacted]\n%s", summaryResp.Content),
	})
	compacted = append(compacted, recent...)
	return compacted, nil
}

// chatLLM calls the LLM, using streaming if available.
func (e *Engine[S]) chatLLM(ctx context.Context, messages []Message, toolDefs []ToolDef) (*Response, error) {
	streamer, ok := e.LLM.(StreamingLLMProvider)
	if !ok {
		return e.LLM.Chat(ctx, messages, toolDefs)
	}
	return streamer.StreamChat(ctx, messages, toolDefs, func(chunk string) {})
}

// expectationMessage returns a fallback message for tool calls with no text.
func (e *Engine[S]) expectationMessage(calls []ToolCall) string {
	if len(calls) == 0 || e.ExpectationMessages == nil {
		return ""
	}
	if msg, ok := e.ExpectationMessages[calls[0].Name]; ok {
		return msg
	}
	return ""
}
```

**Step 4: Run tests**

Run: `cd /home/storymaker/mini-agent-go && go test -v ./...`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agent.go agent_test.go
git commit -m "feat: add Engine[S] agent loop with compaction, buffering, and hooks"
```

---

### Task 6: Anthropic provider — anthropic.go

Claude API client with streaming SSE support. Configurable model, base URL, max tokens.

**Files:**
- Create: `/home/storymaker/mini-agent-go/anthropic.go`
- Test: `/home/storymaker/mini-agent-go/anthropic_test.go`

**Step 1: Write the failing tests**

Port from `anthropic_test.go`. Use `net/http/httptest` to mock the API.

```go
package agent

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAnthropicProviderChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "test-key" {
			t.Error("missing API key header")
		}
		resp := map[string]interface{}{
			"id":   "msg_test",
			"type": "message",
			"role": "assistant",
			"content": []map[string]interface{}{
				{"type": "tool_use", "id": "tc1", "name": "my_tool", "input": map[string]string{"key": "val"}},
			},
			"usage": map[string]int{"input_tokens": 100, "output_tokens": 50},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewAnthropicProvider(AnthropicOptions{
		APIKey:  "test-key",
		Model:   "claude-test",
		BaseURL: server.URL,
	})

	resp, err := p.Chat(nil, []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "my_tool" {
		t.Errorf("tool calls = %+v", resp.ToolCalls)
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 100 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestAnthropicProviderTextResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"id":   "msg_test",
			"type": "message",
			"role": "assistant",
			"content": []map[string]interface{}{
				{"type": "text", "text": "Hello there!"},
			},
			"usage": map[string]int{"input_tokens": 50, "output_tokens": 10},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewAnthropicProvider(AnthropicOptions{
		APIKey:  "test-key",
		Model:   "claude-test",
		BaseURL: server.URL,
	})

	resp, err := p.Chat(nil, []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "Hello there!" {
		t.Errorf("content = %q", resp.Content)
	}
}

func TestAnthropicMergesConsecutiveUserMessages(t *testing.T) {
	var receivedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedBody)
		resp := map[string]interface{}{
			"id": "msg_test", "type": "message", "role": "assistant",
			"content": []map[string]interface{}{{"type": "text", "text": "ok"}},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewAnthropicProvider(AnthropicOptions{
		APIKey: "k", Model: "m", BaseURL: server.URL,
	})

	msgs := []Message{
		{Role: "user", Content: "first"},
		{Role: "user", Content: "second"},
	}
	_, err := p.Chat(nil, msgs, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	apiMsgs := receivedBody["messages"].([]interface{})
	if len(apiMsgs) != 1 {
		t.Fatalf("expected 1 merged message, got %d", len(apiMsgs))
	}
}
```

**Step 2: Run tests — expect FAIL**

**Step 3: Write anthropic.go**

Port from source. Key changes:
- `NewAnthropicProvider(opts AnthropicOptions)` with configurable model, base URL, max tokens
- Replace `models.CostBreakdown` / `models.NewCostBreakdown()` / `cost.AddModel()` with `Usage{}`
- Replace `models.ModelSonnet` constant with `opts.Model`
- Remove `ModelName()` method (not in interface)
- Default base URL: `https://api.anthropic.com/v1/messages`
- Default max tokens: 8192
- Keep: streaming, retries, request building, consecutive user message merging

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add anthropic.go anthropic_test.go
git commit -m "feat: add AnthropicProvider with streaming SSE support"
```

---

### Task 7: Gemini provider — gemini.go

Gemini function-calling API client with streaming. Same pattern as Anthropic.

**Files:**
- Create: `/home/storymaker/mini-agent-go/gemini.go`
- Test: `/home/storymaker/mini-agent-go/gemini_test.go`

**Step 1: Write the failing tests**

```go
package agent

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGeminiProviderChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"candidates": []map[string]interface{}{
				{
					"content": map[string]interface{}{
						"role": "model",
						"parts": []map[string]interface{}{
							{"functionCall": map[string]interface{}{"name": "my_tool", "args": map[string]string{"x": "1"}}},
						},
					},
				},
			},
			"usageMetadata": map[string]int{"promptTokenCount": 80, "candidatesTokenCount": 20},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewGeminiProvider(GeminiOptions{
		APIKey:  "test-key",
		Model:   "gemini-test",
		BaseURL: server.URL,
	})

	resp, err := p.Chat(nil, []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "my_tool" {
		t.Errorf("tool calls = %+v", resp.ToolCalls)
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 80 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestGeminiProviderTextResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"candidates": []map[string]interface{}{
				{
					"content": map[string]interface{}{
						"role":  "model",
						"parts": []map[string]interface{}{{"text": "Hello!"}},
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewGeminiProvider(GeminiOptions{
		APIKey: "k", Model: "m", BaseURL: server.URL,
	})

	resp, err := p.Chat(nil, []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "Hello!" {
		t.Errorf("content = %q", resp.Content)
	}
}
```

**Step 2: Run tests — expect FAIL**

**Step 3: Write gemini.go**

Port from source. Key changes:
- `NewGeminiProvider(opts GeminiOptions)` with configurable model, base URL
- Replace `models.CostBreakdown` with `Usage{}`
- Replace `models.ModelFlash` constant with `opts.Model`
- Move `retryableStatus` and `truncateBody` to a shared `helpers.go` (or keep in gemini.go if anthropic.go also defines them — need to deduplicate)
- Keep: streaming, function calling, thought signatures, request building

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add gemini.go gemini_test.go
git commit -m "feat: add GeminiProvider with function calling and streaming"
```

---

### Task 8: Shared helpers — helpers.go

Both providers use `retryableStatus` and `truncateBody`. Deduplicate into a shared file.

**Files:**
- Create: `/home/storymaker/mini-agent-go/helpers.go`

**Step 1: Create helpers.go**

```go
package agent

// retryableStatus returns true for HTTP status codes that warrant a retry.
func retryableStatus(code int) bool {
	return code == 429 || code == 500 || code == 502 || code == 503 || code == 504
}

// truncateBody truncates a byte slice for logging.
func truncateBody(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "..."
}
```

**Step 2: Remove duplicate definitions from anthropic.go and gemini.go**

**Step 3: Run all tests**

Run: `cd /home/storymaker/mini-agent-go && go test -v ./...`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add helpers.go anthropic.go gemini.go
git commit -m "refactor: deduplicate retryableStatus and truncateBody into helpers.go"
```

---

### Task 9: README

**Files:**
- Create: `/home/storymaker/mini-agent-go/README.md`

**Step 1: Write README**

Include:
- One-line description
- Installation: `go get github.com/logosc/mini-agent-go`
- Quick example showing Engine[S] with a custom state and tool
- API overview (Engine, Tool, ChatInterface, providers)
- License: MIT

Keep it concise — under 150 lines.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage example"
```

---

### Task 10: Final verification

**Step 1: Run full test suite**

```bash
cd /home/storymaker/mini-agent-go && go test -v -count=1 ./...
```

Expected: ALL PASS, zero external dependencies.

**Step 2: Verify no external deps**

```bash
cd /home/storymaker/mini-agent-go && go list -m all
```

Expected: only `github.com/logosc/mini-agent-go` — no third-party modules.

**Step 3: Verify build**

```bash
cd /home/storymaker/mini-agent-go && go vet ./...
```

Expected: clean
