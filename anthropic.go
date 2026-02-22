package agent

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"context"
)

const (
	defaultAnthropicBaseURL    = "https://api.anthropic.com/v1/messages"
	defaultAnthropicMaxTokens  = 8192
	anthropicAPIVersion        = "2023-06-01"
)

// Compile-time checks.
var _ LLMProvider = (*AnthropicProvider)(nil)
var _ StreamingLLMProvider = (*AnthropicProvider)(nil)

// AnthropicOptions configures an AnthropicProvider.
type AnthropicOptions struct {
	APIKey     string
	Model      string
	BaseURL    string // default: https://api.anthropic.com/v1/messages
	MaxTokens  int    // default: 8192
	SystemHint string // optional cache-control system prefix
}

// AnthropicProvider implements LLMProvider and StreamingLLMProvider using Claude's messages API.
type AnthropicProvider struct {
	apiKey     string
	model      string
	baseURL    string
	maxTokens  int
	systemHint string
	httpClient *http.Client
}

// NewAnthropicProvider creates an AnthropicProvider with the given options.
func NewAnthropicProvider(opts AnthropicOptions) *AnthropicProvider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultAnthropicBaseURL
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultAnthropicMaxTokens
	}
	return &AnthropicProvider{
		apiKey:     opts.APIKey,
		model:      opts.Model,
		baseURL:    baseURL,
		maxTokens:  maxTokens,
		systemHint: opts.SystemHint,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

// ---------- Anthropic request types ----------

type anthropicRequest struct {
	Model     string                `json:"model"`
	MaxTokens int                   `json:"max_tokens"`
	System    anthropicMsgContent   `json:"system,omitempty"`
	Messages  []anthropicMessage    `json:"messages"`
	Tools     []anthropicTool       `json:"tools,omitempty"`
	Stream    bool                  `json:"stream,omitempty"`
}

type anthropicMessage struct {
	Role    string              `json:"role"`
	Content anthropicMsgContent `json:"content"`
}

// anthropicMsgContent can be a plain string or an array of content blocks.
// We always use the array form for flexibility.
type anthropicMsgContent []anthropicContentBlock

type anthropicContentBlock struct {
	Type string `json:"type"`

	// type: "text"
	Text string `json:"text,omitempty"`

	// type: "image"
	Source *anthropicImageSource `json:"source,omitempty"`

	// type: "tool_use"
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// type: "tool_result"
	ToolUseID string              `json:"tool_use_id,omitempty"`
	Content   anthropicMsgContent `json:"content,omitempty"`

	// prompt caching
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicCacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

type anthropicImageSource struct {
	Type      string `json:"type"`       // "base64"
	MediaType string `json:"media_type"` // "image/jpeg", "image/png"
	Data      string `json:"data"`
}

type anthropicTool struct {
	Name         string                `json:"name"`
	Description  string                `json:"description"`
	InputSchema  json.RawMessage       `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// ---------- Anthropic response types ----------

type anthropicResponse struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"` // "message"
	Role       string                 `json:"role"`
	Content    []anthropicContentBlock `json:"content"`
	StopReason string                 `json:"stop_reason"`
	Usage      *anthropicUsage        `json:"usage,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// ---------- Streaming event types ----------

type anthropicStreamEvent struct {
	Type         string                `json:"type"`
	Message      *anthropicResponse    `json:"message,omitempty"`
	Index        int                   `json:"index,omitempty"`
	ContentBlock *anthropicContentBlock `json:"content_block,omitempty"`
	Delta        *anthropicStreamDelta `json:"delta,omitempty"`
	Usage        *anthropicUsage       `json:"usage,omitempty"`
}

type anthropicStreamDelta struct {
	Type        string          `json:"type,omitempty"`
	Text        string          `json:"text,omitempty"`
	PartialJSON string          `json:"partial_json,omitempty"`
	StopReason  string          `json:"stop_reason,omitempty"`
	ID          string          `json:"id,omitempty"`
	Name        string          `json:"name,omitempty"`
	Input       json.RawMessage `json:"input,omitempty"`
}

// ---------- Chat implementation ----------

func (p *AnthropicProvider) Chat(ctx context.Context, messages []Message, tools []ToolDef) (*Response, error) {
	req, err := p.buildRequest(messages, tools, false)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build request: %w", err)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	log.Printf("[anthropic-chat] POST (%d bytes): %s", len(body), truncateBody(body, 500))

	const maxRetries = 3
	var lastErr error
	for attempt := range maxRetries {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			log.Printf("[anthropic-chat] retrying in %v (attempt %d/%d)", backoff, attempt+1, maxRetries)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("anthropic: new request: %w", err)
		}
		p.setHeaders(httpReq)

		start := time.Now()
		httpResp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("anthropic: do request (after %v): %w", time.Since(start), err)
			continue
		}

		respBody, err := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("anthropic: read response: %w", err)
			continue
		}
		log.Printf("[anthropic-chat] response: %d (%d bytes, %v)", httpResp.StatusCode, len(respBody), time.Since(start))

		if retryableStatus(httpResp.StatusCode) {
			lastErr = fmt.Errorf("anthropic: HTTP %d: %s", httpResp.StatusCode, string(respBody))
			continue
		}
		if httpResp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("anthropic: HTTP %d: %s", httpResp.StatusCode, string(respBody))
		}

		return p.parseResponse(respBody)
	}
	return nil, lastErr
}

func (p *AnthropicProvider) StreamChat(ctx context.Context, messages []Message, tools []ToolDef, onText func(chunk string)) (*Response, error) {
	req, err := p.buildRequest(messages, tools, true)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build request: %w", err)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	log.Printf("[anthropic-stream] POST (%d bytes): %s", len(body), truncateBody(body, 500))

	const maxRetries = 3
	var httpResp *http.Response
	var lastErr error
	streamStart := time.Now()
	for attempt := range maxRetries {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			log.Printf("[anthropic-stream] retrying in %v (attempt %d/%d)", backoff, attempt+1, maxRetries)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("anthropic: new request: %w", err)
		}
		p.setHeaders(httpReq)

		start := time.Now()
		httpResp, err = p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("anthropic: do request (after %v): %w", time.Since(start), err)
			continue
		}

		if retryableStatus(httpResp.StatusCode) {
			respBody, _ := io.ReadAll(httpResp.Body)
			httpResp.Body.Close()
			lastErr = fmt.Errorf("anthropic: HTTP %d: %s", httpResp.StatusCode, string(respBody))
			continue
		}
		if httpResp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(httpResp.Body)
			httpResp.Body.Close()
			return nil, fmt.Errorf("anthropic: HTTP %d: %s", httpResp.StatusCode, string(respBody))
		}

		lastErr = nil
		break
	}
	if lastErr != nil {
		return nil, lastErr
	}
	defer httpResp.Body.Close()

	// Claude streaming uses SSE with "event: <type>\ndata: <json>\n\n" format.
	resp := &Response{}
	scanner := bufio.NewScanner(httpResp.Body)
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)

	// Track in-progress tool_use blocks by index.
	type toolAccum struct {
		id   string
		name string
		args strings.Builder
	}
	toolBlocks := map[int]*toolAccum{}

	var eventType string
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		switch eventType {
		case "message_start":
			var evt anthropicStreamEvent
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				log.Printf("[anthropic-stream] skip unparseable message_start: %v", err)
				continue
			}
			if evt.Message != nil && evt.Message.Usage != nil {
				u := evt.Message.Usage
				resp.Usage = &Usage{
					InputTokens:      u.InputTokens,
					OutputTokens:     0,
					CacheReadTokens:  u.CacheReadInputTokens,
					CacheWriteTokens: u.CacheCreationInputTokens,
				}
				logCacheStats(u)
			}

		case "content_block_start":
			var evt anthropicStreamEvent
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				continue
			}
			if evt.ContentBlock != nil && evt.ContentBlock.Type == "tool_use" {
				toolBlocks[evt.Index] = &toolAccum{
					id:   evt.ContentBlock.ID,
					name: evt.ContentBlock.Name,
				}
			}

		case "content_block_delta":
			var evt anthropicStreamEvent
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				continue
			}
			if evt.Delta == nil {
				continue
			}
			if evt.Delta.Type == "text_delta" && evt.Delta.Text != "" {
				onText(evt.Delta.Text)
				resp.Content += evt.Delta.Text
			}
			if evt.Delta.Type == "input_json_delta" {
				if ta, ok := toolBlocks[evt.Index]; ok {
					ta.args.WriteString(evt.Delta.PartialJSON)
				}
			}

		case "content_block_stop":
			var evt anthropicStreamEvent
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				continue
			}
			if ta, ok := toolBlocks[evt.Index]; ok {
				args := json.RawMessage(ta.args.String())
				if !json.Valid(args) {
					args = json.RawMessage("{}")
				}
				resp.ToolCalls = append(resp.ToolCalls, ToolCall{
					ID:   ta.id,
					Name: ta.name,
					Args: args,
				})
				delete(toolBlocks, evt.Index)
			}

		case "message_delta":
			var evt anthropicStreamEvent
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				continue
			}
			if evt.Usage != nil && resp.Usage != nil {
				resp.Usage.OutputTokens = evt.Usage.OutputTokens
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("anthropic: read stream: %w", err)
	}

	log.Printf("[anthropic-stream] completed (%v, text=%d chars, tools=%d)", time.Since(streamStart), len(resp.Content), len(resp.ToolCalls))
	return resp, nil
}

// ---------- Helpers ----------

func (p *AnthropicProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicAPIVersion)
	req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
}

func (p *AnthropicProvider) buildRequest(messages []Message, tools []ToolDef, stream bool) (*anthropicRequest, error) {
	req := &anthropicRequest{
		Model:     p.model,
		MaxTokens: p.maxTokens,
		Stream:    stream,
	}

	var apiMsgs []anthropicMessage

	ephemeral := &anthropicCacheControl{Type: "ephemeral"}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			text := msg.Content
			if p.systemHint != "" {
				text = p.systemHint + "\n\n" + text
			}
			req.System = anthropicMsgContent{{
				Type:         "text",
				Text:         text,
				CacheControl: ephemeral,
			}}

		case "user":
			if len(msg.AudioData) > 0 {
				return nil, fmt.Errorf("anthropic: audio input is not supported; use a Gemini provider for voice messages")
			}
			var blocks []anthropicContentBlock
			if msg.Content != "" {
				blocks = append(blocks, anthropicContentBlock{
					Type: "text",
					Text: msg.Content,
				})
			}
			if len(msg.ImageData) > 0 {
				mediaType := "image/jpeg"
				if len(msg.ImageData) > 1 && msg.ImageData[0] == 0x89 && msg.ImageData[1] == 0x50 {
					mediaType = "image/png"
				}
				blocks = append(blocks, anthropicContentBlock{
					Type: "image",
					Source: &anthropicImageSource{
						Type:      "base64",
						MediaType: mediaType,
						Data:      base64.StdEncoding.EncodeToString(msg.ImageData),
					},
				})
			}
			// Merge into previous user message if consecutive.
			if len(apiMsgs) > 0 && apiMsgs[len(apiMsgs)-1].Role == "user" {
				last := &apiMsgs[len(apiMsgs)-1]
				last.Content = append(last.Content, blocks...)
				continue
			}
			apiMsgs = append(apiMsgs, anthropicMessage{
				Role:    "user",
				Content: blocks,
			})

		case "assistant":
			blocks := p.assistantBlocks(msg)
			apiMsgs = append(apiMsgs, anthropicMessage{
				Role:    "assistant",
				Content: blocks,
			})

		case "tool":
			// Claude requires tool_result blocks inside a "user" role message.
			block := anthropicContentBlock{
				Type:      "tool_result",
				ToolUseID: msg.ToolCallID,
				Content: anthropicMsgContent{
					{Type: "text", Text: msg.Content},
				},
			}

			if len(apiMsgs) > 0 && apiMsgs[len(apiMsgs)-1].Role == "user" {
				last := &apiMsgs[len(apiMsgs)-1]
				if len(last.Content) > 0 && last.Content[0].Type == "tool_result" {
					last.Content = append(last.Content, block)
					continue
				}
			}
			apiMsgs = append(apiMsgs, anthropicMessage{
				Role:    "user",
				Content: anthropicMsgContent{block},
			})
		}
	}

	req.Messages = apiMsgs

	if len(tools) > 0 {
		for _, t := range tools {
			req.Tools = append(req.Tools, anthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.Parameters,
			})
		}
		// Mark the last tool for prompt caching so all tools + system are cached.
		req.Tools[len(req.Tools)-1].CacheControl = ephemeral
	}

	return req, nil
}

func (p *AnthropicProvider) assistantBlocks(msg Message) []anthropicContentBlock {
	var blocks []anthropicContentBlock
	if msg.Content != "" {
		blocks = append(blocks, anthropicContentBlock{
			Type: "text",
			Text: msg.Content,
		})
	}
	for _, tc := range msg.ToolCalls {
		blocks = append(blocks, anthropicContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Name,
			Input: tc.Args,
		})
	}
	return blocks
}

func (p *AnthropicProvider) parseResponse(body []byte) (*Response, error) {
	var ar anthropicResponse
	if err := json.Unmarshal(body, &ar); err != nil {
		return nil, fmt.Errorf("anthropic: unmarshal response: %w", err)
	}

	resp := &Response{}
	for _, block := range ar.Content {
		switch block.Type {
		case "text":
			if resp.Content != "" {
				resp.Content += "\n"
			}
			resp.Content += block.Text
		case "tool_use":
			resp.ToolCalls = append(resp.ToolCalls, ToolCall{
				ID:   block.ID,
				Name: block.Name,
				Args: block.Input,
			})
		}
	}

	if ar.Usage != nil {
		resp.Usage = &Usage{
			InputTokens:      ar.Usage.InputTokens,
			OutputTokens:     ar.Usage.OutputTokens,
			CacheReadTokens:  ar.Usage.CacheReadInputTokens,
			CacheWriteTokens: ar.Usage.CacheCreationInputTokens,
		}
		logCacheStats(ar.Usage)
	}

	return resp, nil
}

// logCacheStats logs prompt cache hit/miss stats when available.
func logCacheStats(u *anthropicUsage) {
	if u == nil {
		return
	}
	if u.CacheReadInputTokens > 0 || u.CacheCreationInputTokens > 0 {
		log.Printf("[anthropic] tokens: input=%d, output=%d, cache_read=%d, cache_create=%d",
			u.InputTokens, u.OutputTokens, u.CacheReadInputTokens, u.CacheCreationInputTokens)
	}
}

// NOTE: retryableStatus and truncateBody helpers are defined in helpers.go
// and shared across providers.
