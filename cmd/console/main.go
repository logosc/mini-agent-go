// Example: Debug console for the to-do agent.
//
// Launches the to-do agent with a web-based debug UI that shows all internal
// events (messages, tool calls, token usage, engine state) in real time.
//
// Usage:
//
//	echo "ANTHROPIC_API_KEY=sk-..." > .env
//	go run .
//	# Open http://localhost:9741 in your browser, click "Start Agent"
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	agent "github.com/logosc/mini-agent-go"
	"github.com/logosc/mini-agent-go/debug"
)

// State holds the to-do list.
type State struct {
	Items []TodoItem
}

type TodoItem struct {
	Text string
	Done bool
}

// --- Tools ---

type AddTool struct{}

func (t *AddTool) Name() string        { return "add_todo" }
func (t *AddTool) Description() string { return "Add a new item to the to-do list" }
func (t *AddTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"text": {"type": "string", "description": "The to-do item text"}
		},
		"required": ["text"]
	}`)
}
func (t *AddTool) Execute(ctx context.Context, state *State, args json.RawMessage) (*agent.ToolResult, error) {
	var p struct{ Text string }
	if err := json.Unmarshal(args, &p); err != nil {
		return agent.ToolResult{}.WithError("invalid args: " + err.Error()), nil
	}
	state.Items = append(state.Items, TodoItem{Text: p.Text})
	return &agent.ToolResult{
		Summary:     fmt.Sprintf("Added: %q (total: %d items)", p.Text, len(state.Items)),
		ChatMessage: fmt.Sprintf("Added: %s", p.Text),
	}, nil
}

type CompleteTool struct{}

func (t *CompleteTool) Name() string        { return "complete_todo" }
func (t *CompleteTool) Description() string { return "Mark a to-do item as complete by its 1-based index" }
func (t *CompleteTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"index": {"type": "integer", "description": "1-based index of the item to complete"}
		},
		"required": ["index"]
	}`)
}
func (t *CompleteTool) Execute(ctx context.Context, state *State, args json.RawMessage) (*agent.ToolResult, error) {
	var p struct{ Index int }
	if err := json.Unmarshal(args, &p); err != nil {
		return agent.ToolResult{}.WithError("invalid args: " + err.Error()), nil
	}
	idx := p.Index - 1
	if idx < 0 || idx >= len(state.Items) {
		return agent.ToolResult{}.WithError(fmt.Sprintf("invalid index %d, have %d items", p.Index, len(state.Items))), nil
	}
	state.Items[idx].Done = true
	return &agent.ToolResult{
		Summary:     fmt.Sprintf("Completed: %q", state.Items[idx].Text),
		ChatMessage: fmt.Sprintf("Done: %s", state.Items[idx].Text),
	}, nil
}

type ListTool struct{}

func (t *ListTool) Name() string        { return "list_todos" }
func (t *ListTool) Description() string { return "List all to-do items with their status" }
func (t *ListTool) Parameters() json.RawMessage {
	return json.RawMessage(`{"type": "object"}`)
}
func (t *ListTool) Execute(ctx context.Context, state *State, args json.RawMessage) (*agent.ToolResult, error) {
	if len(state.Items) == 0 {
		return &agent.ToolResult{Summary: "No items yet."}, nil
	}
	var sb strings.Builder
	for i, item := range state.Items {
		mark := "[ ]"
		if item.Done {
			mark = "[x]"
		}
		fmt.Fprintf(&sb, "%d. %s %s\n", i+1, mark, item.Text)
	}
	return &agent.ToolResult{
		Summary:     sb.String(),
		ChatMessage: sb.String(),
	}, nil
}

// loadEnv reads a .env file (if present) and returns key-value pairs.
// Does not modify the process environment.
func loadEnv(path string) map[string]string {
	m := make(map[string]string)
	f, err := os.Open(path)
	if err != nil {
		return m
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		v = strings.Trim(v, `"'`)
		m[k] = v
	}
	return m
}

// envOr returns the environment variable value, falling back to the .env map.
func envOr(env map[string]string, key string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return env[key]
}

func main() {
	env := loadEnv("../../.env")

	apiKey := envOr(env, "ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "Set ANTHROPIC_API_KEY in .env or environment")
		os.Exit(1)
	}

	state := &State{}

	engine := &agent.Engine[*State]{
		LLM: agent.NewAnthropicProvider(agent.AnthropicOptions{
			APIKey: apiKey,
			Model:  "claude-sonnet-4-6",
		}),
		Tools: []agent.Tool[*State]{
			&AddTool{},
			&CompleteTool{},
			&ListTool{},
		},
		SystemPrompt:  "You are a helpful to-do list assistant. Use the available tools to manage the user's to-do list. Be concise.",
		MaxIterations: 50,
	}

	addr := ":9741"
	if v := envOr(env, "PORT"); v != "" {
		addr = ":" + v
	}
	fmt.Printf("Debug console starting at http://localhost%s\n", addr)
	if err := debug.Serve(context.Background(), engine, state, addr); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
