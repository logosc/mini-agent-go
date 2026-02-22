# mini-agent-go SDK Design

A minimal, generic agent loop SDK in Go.

**Module:** `github.com/logosc/mini-agent-go`
**License:** MIT
**Package name:** `agent`

## Package Layout

```
mini-agent-go/
├── agent.go        # Engine[S], Run loop, config
├── provider.go     # LLMProvider, StreamingLLMProvider, Message, Response, Usage
├── tool.go         # Tool[S], ToolDef, ToolResult, Attachment
├── chat.go         # ChatInterface, ChannelChat, NullChat, buffering
├── anthropic.go    # AnthropicProvider (Claude API client with streaming)
├── gemini.go       # GeminiProvider (Gemini function-calling client)
├── go.mod
├── go.sum
├── LICENSE         # MIT
├── README.md
└── docs/
    └── design.md   # this file
```

Flat layout — no sub-packages. Both LLM providers use only stdlib (`net/http`, `encoding/json`).

## Core Types

### Messages & LLM

```go
type Message struct {
    Role       string          // "system", "user", "assistant", "tool"
    Content    string
    ImageData  []byte          // optional inline image for user messages
    ToolCalls  []ToolCall
    ToolCallID string          // set when Role="tool"
}

type ToolCall struct {
    ID   string
    Name string
    Args json.RawMessage
}

type ToolDef struct {
    Name        string
    Description string
    Parameters  json.RawMessage // JSON Schema
}

type Usage struct {
    InputTokens        int
    OutputTokens       int
    CacheReadTokens    int
    CacheWriteTokens   int
}

type Response struct {
    Content   string
    ToolCalls []ToolCall
    Usage     *Usage
}
```

### LLM Provider

```go
type LLMProvider interface {
    Chat(ctx context.Context, messages []Message, tools []ToolDef) (*Response, error)
}

type StreamingLLMProvider interface {
    LLMProvider
    StreamChat(ctx context.Context, messages []Message, tools []ToolDef, onText func(chunk string)) (*Response, error)
}
```

### Tool

Uses Go generics — `S` is the application-defined state type.

```go
type Tool[S any] interface {
    Name() string
    Description() string
    Parameters() json.RawMessage
    Execute(ctx context.Context, state S, args json.RawMessage) (*ToolResult, error)
}

type Attachment struct {
    Type    string // "file", "image"
    Data    []byte
    Name    string // filename
    Caption string // optional
}

type ToolResult struct {
    Summary     string       // sent to LLM as tool response text
    ChatMessage string       // sent directly to user, bypassing LLM
    Attachments []Attachment // files/images sent to user
    IsError     bool
}
```

### Chat Interface

Bidirectional communication with the user. The engine uses this for sending messages, waiting for replies, and draining buffered messages that arrived during tool execution.

```go
type Reply struct {
    Text      string
    ImageData []byte
}

type BufferedMessage struct {
    Text      string
    ImageData []byte
}

type ChatInterface interface {
    Send(ctx context.Context, text string) error
    SendAttachment(ctx context.Context, att Attachment) error
    WaitForReply(ctx context.Context) (Reply, error)
    DrainMessages() []BufferedMessage
}
```

**Provided implementations:**

- `NullChat` — no-op for headless/background agents. `WaitForReply` returns an error.
- `ChannelChat` — function-based implementation with goroutine-safe message buffering. Constructed with `SendFunc`, `AttachmentFunc`, and a `ReplyCh chan Reply`.

## Engine

```go
type Engine[S any] struct {
    LLM           LLMProvider
    Tools         []Tool[S]
    Chat          ChatInterface    // nil defaults to NullChat
    SystemPrompt  string
    MaxIterations int              // default 20
    OnToolDone    func(state S, name string, result *ToolResult) // optional hook
    OnUsage       func(usage Usage) // optional hook for token tracking
}

func (e *Engine[S]) Run(ctx context.Context, state S, taskPrompt string) error
```

## Engine.Run Loop Behavior

The loop runs up to `MaxIterations` turns:

### 1. LLM Call
Calls `LLM.Chat` (or `StreamChat` if available) with the full message history and tool definitions.

### 2. No Tool Calls (Idle Turn)
- Sends `resp.Content` to the user via `Chat.Send`
- Increments idle counter
- Two consecutive idle turns → `Run()` returns nil (conversation done)
- Otherwise calls `Chat.WaitForReply()` and appends the user's reply to history

### 3. Tool Calls
- Sends `resp.Content` (or a fallback expectation message) to the user
- Executes each tool call sequentially
- For each tool result:
  - Appends tool result message to history
  - Drains buffered messages via `Chat.DrainMessages()`, injects with `[System: ...]` status prefix
  - Sends `ChatMessage` via `Chat.Send` if set
  - Sends each `Attachment` via `Chat.SendAttachment`
  - Calls `OnToolDone` hook if set
- Calls `OnUsage` hook with response usage if set

### 4. Conversation Compaction
At 75% of `MaxIterations`, the engine asks the LLM to summarize the middle of the conversation (keeping system prompt + last 6 messages). Replaces the middle with the summary and resets the iteration counter. Up to 2 compactions per session.

### 5. Fallback Expectation Messages
When the LLM makes tool calls but includes no text, the engine can send a user-facing progress message. This is configurable via a map:

```go
type Engine[S any] struct {
    // ...
    ExpectationMessages map[string]string // tool name → "Working on it..."
}
```

## Anthropic Provider

Full Claude API client.

```go
func NewAnthropicProvider(apiKey, model string) *AnthropicProvider
func NewAnthropicProviderWithOptions(opts AnthropicOptions) *AnthropicProvider

type AnthropicOptions struct {
    APIKey     string
    Model      string
    BaseURL    string // default: https://api.anthropic.com
    MaxTokens  int    // default: 8192
    SystemHint string // optional cache-control system prefix
}
```

Implements `StreamingLLMProvider`. Handles:
- Streaming SSE responses
- Tool use blocks
- Consecutive user message merging (Anthropic API requirement)
- Cache control headers

## Gemini Provider

Full Gemini function-calling client.

```go
func NewGeminiProvider(apiKey, model string) *GeminiProvider

type GeminiOptions struct {
    APIKey string
    Model  string
}
```

Implements `LLMProvider`. Handles:
- Function calling / tool use
- Thought signatures (echo back for multi-turn)

## What the SDK Does NOT Include

These are application-specific and stay in the consuming app:

- System prompt content
- Tool implementations
- State type definition
- Job/pipeline management
- Storage backends
- Platform integrations (Telegram, Messenger, etc.)
- Parallel tool execution (future consideration)

## Usage Example

```go
import agent "github.com/logosc/mini-agent-go"

type MyState struct {
    UserID string
    Data   map[string]string
}

// Tools implement agent.Tool[*MyState]
type MyTool struct{}
func (t *MyTool) Name() string { return "my_tool" }
func (t *MyTool) Description() string { return "Does something useful" }
func (t *MyTool) Parameters() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t *MyTool) Execute(ctx context.Context, state *MyState, args json.RawMessage) (*agent.ToolResult, error) {
    return &agent.ToolResult{Summary: "done"}, nil
}

// Engine setup
engine := &agent.Engine[*MyState]{
    LLM:          agent.NewAnthropicProvider(agent.AnthropicOptions{APIKey: apiKey, Model: "claude-sonnet-4-6"}),
    Tools:        []agent.Tool[*MyState]{&MyTool{}},
    Chat:         chat,
    SystemPrompt: "You are a helpful assistant.",
    OnToolDone:   func(state *MyState, name string, result *agent.ToolResult) {
        // handle post-tool logic
    },
}
engine.Run(ctx, &MyState{UserID: "u1"}, "Help me with something")
```
