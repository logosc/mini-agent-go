# mini-agent-go

A minimal, generic agent loop SDK in Go for building LLM-driven agents.

## Features

- Generic `Engine[S]` agent loop with configurable state type
- LLM provider abstraction with streaming support
- Built-in Anthropic (Claude) and Gemini providers — zero external dependencies
- Bidirectional chat interface with message buffering
- Conversation compaction for long-running sessions
- Tool execution with typed results, attachments, and lifecycle hooks

## Installation

```bash
go get github.com/logosc/mini-agent-go
```

## Quick Start

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"

	agent "github.com/logosc/mini-agent-go"
)

// Define your application state.
type MyState struct {
	UserID string
}

// Implement a tool.
type GreetTool struct{}

func (t *GreetTool) Name() string               { return "greet" }
func (t *GreetTool) Description() string         { return "Greet someone by name" }
func (t *GreetTool) Parameters() json.RawMessage { return json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`) }
func (t *GreetTool) Execute(ctx context.Context, state *MyState, args json.RawMessage) (*agent.ToolResult, error) {
	var p struct{ Name string }
	json.Unmarshal(args, &p)
	return &agent.ToolResult{Summary: fmt.Sprintf("Greeted %s", p.Name)}, nil
}

func main() {
	engine := &agent.Engine[*MyState]{
		LLM:          agent.NewAnthropicProvider(agent.AnthropicOptions{APIKey: "your-key", Model: "claude-sonnet-4-6"}),
		Tools:        []agent.Tool[*MyState]{&GreetTool{}},
		SystemPrompt: "You are a helpful assistant.",
		MaxIterations: 20,
	}
	engine.Run(context.Background(), &MyState{UserID: "u1"}, "Say hello to Alice")
}
```

## API Overview

### Engine

`Engine[S]` is the core agent loop. It calls the LLM, executes tool calls, and manages conversation state.

```go
type Engine[S any] struct {
    LLM                LLMProvider
    Tools              []Tool[S]
    Chat               ChatInterface       // nil defaults to NullChat
    SystemPrompt       string
    MaxIterations      int                 // default: 20
    ExpectationMessages map[string]string  // tool name → fallback message
    OnToolDone         func(state S, name string, result *ToolResult)
    OnUsage            func(usage Usage)
}
```

### Providers

**Anthropic (Claude):**

```go
p := agent.NewAnthropicProvider(agent.AnthropicOptions{
    APIKey:    "sk-...",
    Model:     "claude-sonnet-4-6",
    BaseURL:   "https://api.anthropic.com/v1/messages", // default
    MaxTokens: 8192,                                     // default
})
```

**Gemini:**

```go
p := agent.NewGeminiProvider(agent.GeminiOptions{
    APIKey: "...",
    Model:  "gemini-2.0-flash",
})
```

### Tools

Tools implement `Tool[S]` where `S` is your state type:

```go
type Tool[S any] interface {
    Name() string
    Description() string
    Parameters() json.RawMessage
    Execute(ctx context.Context, state S, args json.RawMessage) (*ToolResult, error)
}
```

### Chat Interface

For interactive agents, implement `ChatInterface` or use the built-in `ChannelChat`:

```go
chat := &agent.ChannelChat{
    SendFunc: func(ctx context.Context, text string) error {
        fmt.Println("Agent:", text)
        return nil
    },
    ReplyCh: make(chan agent.Reply),
}
```

Use `NullChat` (the default) for headless/background agents.

## License

MIT — see [LICENSE](LICENSE).
