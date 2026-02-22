package agent

import (
	"context"
	"encoding/json"
)

// Message represents a single message in the agent conversation.
type Message struct {
	Role           string          `json:"role"`                   // "system", "user", "assistant", "tool"
	Content        string          `json:"content"`
	ImageData      []byte          `json:"-"`                      // Optional inline image for user messages.
	AudioData      []byte          `json:"-"`                      // Optional inline audio for user messages.
	AudioMediaType string          `json:"-"`                      // MIME type for AudioData.
	ToolCalls      []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID     string          `json:"tool_call_id,omitempty"` // set when Role="tool"
}

// ToolCall represents a tool invocation requested by the LLM.
type ToolCall struct {
	ID               string          `json:"id"`
	Name             string          `json:"name"`
	Args             json.RawMessage `json:"arguments"`
	ThoughtSignature string          `json:"thought_signature,omitempty"`
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
