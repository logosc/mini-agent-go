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
