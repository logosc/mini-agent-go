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

// Run starts the agent loop.
func (e *Engine[S]) Run(ctx context.Context, state S, taskPrompt string) error {
	chat := e.Chat
	if chat == nil {
		chat = NullChat{}
	}
	maxIter := e.MaxIterations
	if maxIter <= 0 {
		maxIter = 20
	}

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
				return nil
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

func (e *Engine[S]) chatLLM(ctx context.Context, messages []Message, toolDefs []ToolDef) (*Response, error) {
	streamer, ok := e.LLM.(StreamingLLMProvider)
	if !ok {
		return e.LLM.Chat(ctx, messages, toolDefs)
	}
	return streamer.StreamChat(ctx, messages, toolDefs, func(chunk string) {})
}

func (e *Engine[S]) expectationMessage(calls []ToolCall) string {
	if len(calls) == 0 || e.ExpectationMessages == nil {
		return ""
	}
	if msg, ok := e.ExpectationMessages[calls[0].Name]; ok {
		return msg
	}
	return ""
}
