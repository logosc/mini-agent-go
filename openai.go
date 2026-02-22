package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	defaultOpenAIBaseURL = "https://api.openai.com/v1"
	defaultOpenAIModel   = "gpt-4.1"
)

// Compile-time checks.
var _ LLMProvider = (*OpenAICompatibleProvider)(nil)
var _ StreamingLLMProvider = (*OpenAICompatibleProvider)(nil)

// OpenAICompatibleOptions configures an OpenAICompatibleProvider.
type OpenAICompatibleOptions struct {
	APIKey  string
	Model   string // default: "gpt-4.1"
	BaseURL string // default: "https://api.openai.com/v1"
}

// OpenAICompatibleProvider implements LLMProvider and StreamingLLMProvider
// using the OpenAI chat completions API format. Compatible with OpenAI,
// DeepSeek, Groq, Together, Fireworks, Ollama, LM Studio, vLLM, and any
// backend that speaks this format.
type OpenAICompatibleProvider struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

// NewOpenAICompatibleProvider creates an OpenAICompatibleProvider with the given options.
func NewOpenAICompatibleProvider(opts OpenAICompatibleOptions) *OpenAICompatibleProvider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = defaultOpenAIBaseURL
	}
	model := opts.Model
	if model == "" {
		model = defaultOpenAIModel
	}
	return &OpenAICompatibleProvider{
		apiKey:  opts.APIKey,
		model:   model,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

// ---------- OpenAI request types ----------

type openaiRequest struct {
	Model         string          `json:"model"`
	Messages      []openaiMessage `json:"messages"`
	Tools         []openaiTool    `json:"tools,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *openaiStreamOp `json:"stream_options,omitempty"`
}

type openaiStreamOp struct {
	IncludeUsage bool `json:"include_usage"`
}

type openaiMessage struct {
	Role       string          `json:"role"`
	Content    any             `json:"content,omitempty"`     // string or []openaiContentPart
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type openaiContentPart struct {
	Type     string           `json:"type"`
	Text     string           `json:"text,omitempty"`
	ImageURL *openaiImageURL  `json:"image_url,omitempty"`
}

type openaiImageURL struct {
	URL string `json:"url"`
}

type openaiToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"` // "function"
	Function openaiToolFunction `json:"function"`
}

type openaiToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiTool struct {
	Type     string         `json:"type"` // "function"
	Function openaiToolDef  `json:"function"`
}

type openaiToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ---------- OpenAI response types ----------

type openaiResponse struct {
	ID      string          `json:"id"`
	Choices []openaiChoice  `json:"choices"`
	Usage   *openaiUsage    `json:"usage,omitempty"`
}

type openaiChoice struct {
	Index        int            `json:"index"`
	Message      openaiRespMsg  `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

type openaiRespMsg struct {
	Role      string           `json:"role"`
	Content   *string          `json:"content"`
	ToolCalls []openaiToolCall  `json:"tool_calls,omitempty"`
}

type openaiUsage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	// Prompt caching is automatic for requests with 1024+ matching prefix tokens.
	// Cached tokens are billed at a reduced rate (e.g. 75-90% off).
	// https://platform.openai.com/docs/guides/prompt-caching
	// https://openai.com/api/pricing/
	PromptTokensDetails     *openaiPromptTokenDetail `json:"prompt_tokens_details,omitempty"`
}

type openaiPromptTokenDetail struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// ---------- Streaming chunk types ----------

type openaiStreamChunk struct {
	ID      string               `json:"id"`
	Choices []openaiStreamChoice `json:"choices"`
	Usage   *openaiUsage         `json:"usage,omitempty"`
}

type openaiStreamChoice struct {
	Index        int              `json:"index"`
	Delta        openaiStreamDelta `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
}

type openaiStreamDelta struct {
	Role      string           `json:"role,omitempty"`
	Content   string           `json:"content,omitempty"`
	ToolCalls []openaiStreamTC `json:"tool_calls,omitempty"`
}

type openaiStreamTC struct {
	Index    int                `json:"index"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function openaiStreamTCFunc `json:"function"`
}

type openaiStreamTCFunc struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ---------- Chat implementation ----------

func (p *OpenAICompatibleProvider) Chat(ctx context.Context, messages []Message, tools []ToolDef) (*Response, error) {
	req := p.buildRequest(messages, tools, false)

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	log.Printf("[openai-chat] POST (%d bytes): %s", len(body), truncateBody(body, 500))

	const maxRetries = 3
	var lastErr error
	for attempt := range maxRetries {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			log.Printf("[openai-chat] retrying in %v (attempt %d/%d)", backoff, attempt+1, maxRetries)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("openai: new request: %w", err)
		}
		p.setHeaders(httpReq)

		start := time.Now()
		httpResp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("openai: do request (after %v): %w", time.Since(start), err)
			continue
		}

		respBody, err := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("openai: read response: %w", err)
			continue
		}
		log.Printf("[openai-chat] response: %d (%d bytes, %v)", httpResp.StatusCode, len(respBody), time.Since(start))

		if retryableStatus(httpResp.StatusCode) {
			lastErr = fmt.Errorf("openai: HTTP %d: %s", httpResp.StatusCode, string(respBody))
			continue
		}
		if httpResp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("openai: HTTP %d: %s", httpResp.StatusCode, string(respBody))
		}

		return p.parseResponse(respBody)
	}
	return nil, lastErr
}

func (p *OpenAICompatibleProvider) StreamChat(ctx context.Context, messages []Message, tools []ToolDef, onText func(chunk string)) (*Response, error) {
	req := p.buildRequest(messages, tools, true)

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	log.Printf("[openai-stream] POST (%d bytes): %s", len(body), truncateBody(body, 500))

	const maxRetries = 3
	var httpResp *http.Response
	var lastErr error
	streamStart := time.Now()
	for attempt := range maxRetries {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			log.Printf("[openai-stream] retrying in %v (attempt %d/%d)", backoff, attempt+1, maxRetries)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("openai: new request: %w", err)
		}
		p.setHeaders(httpReq)

		start := time.Now()
		httpResp, err = p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("openai: do request (after %v): %w", time.Since(start), err)
			continue
		}

		if retryableStatus(httpResp.StatusCode) {
			respBody, _ := io.ReadAll(httpResp.Body)
			httpResp.Body.Close()
			lastErr = fmt.Errorf("openai: HTTP %d: %s", httpResp.StatusCode, string(respBody))
			continue
		}
		if httpResp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(httpResp.Body)
			httpResp.Body.Close()
			return nil, fmt.Errorf("openai: HTTP %d: %s", httpResp.StatusCode, string(respBody))
		}

		lastErr = nil
		break
	}
	if lastErr != nil {
		return nil, lastErr
	}
	defer httpResp.Body.Close()

	resp := &Response{}
	scanner := bufio.NewScanner(httpResp.Body)
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)

	// Track in-progress tool calls by index.
	type toolAccum struct {
		id   string
		name string
		args strings.Builder
	}
	toolBlocks := map[int]*toolAccum{}

	// SSE events can span multiple "data:" lines. Accumulate them and
	// process the complete payload when we hit an empty line (event boundary).
	var dataBuf strings.Builder
	flushSSE := func() {
		if dataBuf.Len() == 0 {
			return
		}
		data := dataBuf.String()
		dataBuf.Reset()

		if data == "[DONE]" {
			return
		}

		var chunk openaiStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("[openai-stream] skip unparseable chunk: %v", err)
			return
		}

		// Usage typically arrives in the final chunk.
		if chunk.Usage != nil {
			resp.Usage = &Usage{
				InputTokens:  chunk.Usage.PromptTokens,
				OutputTokens: chunk.Usage.CompletionTokens,
			}
			if chunk.Usage.PromptTokensDetails != nil {
				resp.Usage.CacheReadTokens = chunk.Usage.PromptTokensDetails.CachedTokens
			}
		}

		if len(chunk.Choices) == 0 {
			return
		}
		delta := chunk.Choices[0].Delta

		if delta.Content != "" {
			onText(delta.Content)
			resp.Content += delta.Content
		}

		for _, tc := range delta.ToolCalls {
			ta, ok := toolBlocks[tc.Index]
			if !ok {
				ta = &toolAccum{}
				toolBlocks[tc.Index] = ta
			}
			if tc.ID != "" {
				ta.id = tc.ID
			}
			if tc.Function.Name != "" {
				ta.name = tc.Function.Name
			}
			ta.args.WriteString(tc.Function.Arguments)
		}
	}

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			if dataBuf.Len() > 0 {
				dataBuf.WriteByte('\n')
			}
			dataBuf.WriteString(strings.TrimPrefix(line, "data: "))
		} else if line == "" {
			// Empty line = end of SSE event.
			flushSSE()
		}
		// Other lines (e.g. "event:", "id:", "retry:") are ignored.
	}
	// Flush any trailing event without a final blank line.
	flushSSE()
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("openai: read stream: %w", err)
	}

	// Finalize accumulated tool calls in index order.
	// Indices may be sparse, so sort map keys rather than assuming 0..n-1.
	indices := make([]int, 0, len(toolBlocks))
	for idx := range toolBlocks {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	for _, idx := range indices {
		ta := toolBlocks[idx]
		args := json.RawMessage(ta.args.String())
		if !json.Valid(args) {
			args = json.RawMessage("{}")
		}
		resp.ToolCalls = append(resp.ToolCalls, ToolCall{
			ID:   ta.id,
			Name: ta.name,
			Args: args,
		})
	}

	log.Printf("[openai-stream] completed (%v, text=%d chars, tools=%d)", time.Since(streamStart), len(resp.Content), len(resp.ToolCalls))
	return resp, nil
}

// ---------- Helpers ----------

func (p *OpenAICompatibleProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
}

func (p *OpenAICompatibleProvider) buildRequest(messages []Message, tools []ToolDef, stream bool) *openaiRequest {
	req := &openaiRequest{
		Model:  p.model,
		Stream: stream,
	}
	if stream {
		req.StreamOptions = &openaiStreamOp{IncludeUsage: true}
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			req.Messages = append(req.Messages, openaiMessage{
				Role:    "system",
				Content: msg.Content,
			})

		case "user":
			req.Messages = append(req.Messages, p.userMessage(msg))

		case "assistant":
			m := openaiMessage{
				Role:    "assistant",
				Content: msg.Content,
			}
			for _, tc := range msg.ToolCalls {
				m.ToolCalls = append(m.ToolCalls, openaiToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: openaiToolFunction{
						Name:      tc.Name,
						Arguments: string(tc.Args),
					},
				})
			}
			req.Messages = append(req.Messages, m)

		case "tool":
			req.Messages = append(req.Messages, openaiMessage{
				Role:       "tool",
				Content:    msg.Content,
				ToolCallID: msg.ToolCallID,
			})
		}
	}

	if len(tools) > 0 {
		for _, t := range tools {
			req.Tools = append(req.Tools, openaiTool{
				Type: "function",
				Function: openaiToolDef{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
				},
			})
		}
	}

	return req
}

func (p *OpenAICompatibleProvider) userMessage(msg Message) openaiMessage {
	if len(msg.ImageData) == 0 {
		return openaiMessage{
			Role:    "user",
			Content: msg.Content,
		}
	}

	// Build multipart content with text + image.
	mediaType := "image/jpeg"
	if len(msg.ImageData) > 1 && msg.ImageData[0] == 0x89 && msg.ImageData[1] == 0x50 {
		mediaType = "image/png"
	}
	dataURI := fmt.Sprintf("data:%s;base64,%s", mediaType, base64.StdEncoding.EncodeToString(msg.ImageData))

	parts := []openaiContentPart{
		{Type: "text", Text: msg.Content},
		{Type: "image_url", ImageURL: &openaiImageURL{URL: dataURI}},
	}
	return openaiMessage{
		Role:    "user",
		Content: parts,
	}
}

func (p *OpenAICompatibleProvider) parseResponse(body []byte) (*Response, error) {
	var or openaiResponse
	if err := json.Unmarshal(body, &or); err != nil {
		return nil, fmt.Errorf("openai: unmarshal response: %w", err)
	}

	resp := &Response{}
	if len(or.Choices) > 0 {
		m := or.Choices[0].Message
		if m.Content != nil {
			resp.Content = *m.Content
		}
		for _, tc := range m.ToolCalls {
			args := json.RawMessage(tc.Function.Arguments)
			if !json.Valid(args) {
				args = json.RawMessage("{}")
			}
			resp.ToolCalls = append(resp.ToolCalls, ToolCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: args,
			})
		}
	}

	if or.Usage != nil {
		resp.Usage = &Usage{
			InputTokens:  or.Usage.PromptTokens,
			OutputTokens: or.Usage.CompletionTokens,
		}
		if or.Usage.PromptTokensDetails != nil {
			resp.Usage.CacheReadTokens = or.Usage.PromptTokensDetails.CachedTokens
		}
	}

	return resp, nil
}
