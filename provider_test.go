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
