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
	"strings"
	"time"
)

// Compile-time checks.
var _ LLMProvider = (*GeminiProvider)(nil)
var _ StreamingLLMProvider = (*GeminiProvider)(nil)

// GeminiOptions configures a GeminiProvider.
type GeminiOptions struct {
	APIKey  string
	Model   string
	BaseURL string // default: https://generativelanguage.googleapis.com/v1beta/models
}

// GeminiProvider implements LLMProvider and StreamingLLMProvider using
// Gemini's function-calling REST API with streaming SSE support.
type GeminiProvider struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

// NewGeminiProvider creates a GeminiProvider with the given options.
func NewGeminiProvider(opts GeminiOptions) *GeminiProvider {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta/models"
	}
	model := opts.Model
	if model == "" {
		model = "gemini-3-flash-preview"
	}
	return &GeminiProvider{
		apiKey:  opts.APIKey,
		model:   model,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

// ---------- Gemini request types ----------

type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiGenerationConfig struct {
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string              `json:"text,omitempty"`
	InlineData       *geminiInlineData   `json:"inlineData,omitempty"`
	FunctionCall     *geminiFunctionCall `json:"functionCall,omitempty"`
	FunctionResponse *geminiFuncResponse `json:"functionResponse,omitempty"`
	ThoughtSignature string              `json:"thoughtSignature,omitempty"`
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64-encoded
}

type geminiFunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args,omitempty"`
}

type geminiFuncResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFuncDecl `json:"functionDeclarations,omitempty"`
}

type geminiFuncDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ---------- Gemini response types ----------

type geminiResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata *geminiUsage      `json:"usageMetadata,omitempty"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
}

// ---------- Chat implementation ----------

// Chat sends a conversation to the Gemini API and returns the parsed response.
func (p *GeminiProvider) Chat(ctx context.Context, messages []Message, tools []ToolDef) (*Response, error) {
	req, err := p.buildRequest(messages, tools)
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	url := p.endpoint()
	log.Printf("[gemini-chat] POST (%d bytes): %s", len(body), truncateBody(body, 500))

	const maxRetries = 3
	var lastErr error
	for attempt := range maxRetries {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			log.Printf("[gemini-chat] retrying in %v (attempt %d/%d)", backoff, attempt+1, maxRetries)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("gemini: new request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		start := time.Now()
		httpResp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("gemini: do request (after %v): %w", time.Since(start), err)
			continue
		}

		respBody, err := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("gemini: read response: %w", err)
			continue
		}
		log.Printf("[gemini-chat] response: %d (%d bytes, %v)", httpResp.StatusCode, len(respBody), time.Since(start))

		if retryableStatus(httpResp.StatusCode) {
			lastErr = fmt.Errorf("gemini: HTTP %d: %s", httpResp.StatusCode, string(respBody))
			continue
		}
		if httpResp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("gemini: HTTP %d: %s", httpResp.StatusCode, string(respBody))
		}

		return p.parseResponse(respBody)
	}
	return nil, lastErr
}

// StreamChat sends a conversation to the Gemini streaming API.
// It calls onText with incremental text chunks as they arrive, and returns
// the fully accumulated Response (with tool calls) when the stream ends.
func (p *GeminiProvider) StreamChat(ctx context.Context, messages []Message, tools []ToolDef, onText func(chunk string)) (*Response, error) {
	req, err := p.buildRequest(messages, tools)
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	url := p.streamEndpoint()
	log.Printf("[gemini-stream] POST (%d bytes): %s", len(body), truncateBody(body, 500))

	const maxRetries = 3
	var httpResp *http.Response
	var lastErr error
	streamStart := time.Now()
	for attempt := range maxRetries {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			log.Printf("[gemini-stream] retrying in %v (attempt %d/%d)", backoff, attempt+1, maxRetries)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("gemini: new request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		start := time.Now()
		httpResp, err = p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("gemini: do request (after %v): %w", time.Since(start), err)
			continue
		}

		if retryableStatus(httpResp.StatusCode) {
			respBody, _ := io.ReadAll(httpResp.Body)
			httpResp.Body.Close()
			lastErr = fmt.Errorf("gemini: HTTP %d: %s", httpResp.StatusCode, string(respBody))
			continue
		}
		if httpResp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(httpResp.Body)
			httpResp.Body.Close()
			return nil, fmt.Errorf("gemini: HTTP %d: %s", httpResp.StatusCode, string(respBody))
		}

		lastErr = nil
		break
	}
	if lastErr != nil {
		return nil, lastErr
	}
	defer httpResp.Body.Close()

	// Gemini streaming with alt=sse returns Server-Sent Events.
	// Each event has "data: {json}\n\n" format.
	resp := &Response{}
	scanner := bufio.NewScanner(httpResp.Body)
	// Allow large lines (Gemini can send big chunks).
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var chunk geminiResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("[gemini-stream] skip unparseable chunk: %v", err)
			continue
		}

		// Usage metadata typically arrives in the final chunk.
		if chunk.UsageMetadata != nil {
			resp.Usage = &Usage{
				InputTokens:  chunk.UsageMetadata.PromptTokenCount,
				OutputTokens: chunk.UsageMetadata.CandidatesTokenCount,
			}
		}

		if len(chunk.Candidates) == 0 {
			continue
		}
		for _, part := range chunk.Candidates[0].Content.Parts {
			if part.Text != "" {
				onText(part.Text)
				resp.Content += part.Text
			}
			if part.FunctionCall != nil {
				argsJSON, _ := json.Marshal(part.FunctionCall.Args)
				resp.ToolCalls = append(resp.ToolCalls, ToolCall{
					ID:               part.FunctionCall.Name,
					Name:             part.FunctionCall.Name,
					Args:             argsJSON,
					ThoughtSignature: part.ThoughtSignature,
				})
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("gemini: read stream: %w", err)
	}

	log.Printf("[gemini-stream] completed (%v, text=%d chars, tools=%d)", time.Since(streamStart), len(resp.Content), len(resp.ToolCalls))
	return resp, nil
}

// endpoint returns the full URL for the generateContent call.
func (p *GeminiProvider) endpoint() string {
	base := p.baseURL
	path := fmt.Sprintf("/%s:generateContent", p.model)

	if strings.HasPrefix(base, "http://") || strings.HasPrefix(base, "https://127") {
		return base + path
	}
	return fmt.Sprintf("%s%s?key=%s", base, path, p.apiKey)
}

// streamEndpoint returns the URL for streaming generateContent.
func (p *GeminiProvider) streamEndpoint() string {
	base := p.baseURL
	path := fmt.Sprintf("/%s:streamGenerateContent?alt=sse", p.model)

	if strings.HasPrefix(base, "http://") || strings.HasPrefix(base, "https://127") {
		return base + path
	}
	return fmt.Sprintf("%s%s&key=%s", base, path, p.apiKey)
}

// buildRequest converts agent messages and tools into a Gemini API request.
func (p *GeminiProvider) buildRequest(messages []Message, tools []ToolDef) (*geminiRequest, error) {
	req := &geminiRequest{
		GenerationConfig: &geminiGenerationConfig{
			MaxOutputTokens: 4096,
		},
	}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			req.SystemInstruction = &geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
			}
		case "user":
			parts := []geminiPart{{Text: msg.Content}}
			if len(msg.ImageData) > 0 {
				mimeType := "image/jpeg"
				if len(msg.ImageData) > 1 && msg.ImageData[0] == 0x89 && msg.ImageData[1] == 0x50 {
					mimeType = "image/png"
				}
				parts = append(parts, geminiPart{
					InlineData: &geminiInlineData{
						MimeType: mimeType,
						Data:     base64.StdEncoding.EncodeToString(msg.ImageData),
					},
				})
			}
			req.Contents = append(req.Contents, geminiContent{
				Role:  "user",
				Parts: parts,
			})
		case "assistant":
			parts := p.assistantParts(msg)
			req.Contents = append(req.Contents, geminiContent{
				Role:  "model",
				Parts: parts,
			})
		case "tool":
			resp := p.toolResponsePart(msg)
			req.Contents = append(req.Contents, geminiContent{
				Role:  "function",
				Parts: []geminiPart{resp},
			})
		}
	}

	if len(tools) > 0 {
		decls := make([]geminiFuncDecl, len(tools))
		for i, t := range tools {
			decls[i] = geminiFuncDecl{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			}
		}
		req.Tools = []geminiTool{{FunctionDeclarations: decls}}
	}

	return req, nil
}

func (p *GeminiProvider) assistantParts(msg Message) []geminiPart {
	var parts []geminiPart
	if msg.Content != "" {
		parts = append(parts, geminiPart{Text: msg.Content})
	}
	for _, tc := range msg.ToolCalls {
		var args map[string]interface{}
		_ = json.Unmarshal(tc.Args, &args)
		part := geminiPart{
			FunctionCall: &geminiFunctionCall{
				Name: tc.Name,
				Args: args,
			},
			ThoughtSignature: tc.ThoughtSignature,
		}
		parts = append(parts, part)
	}
	return parts
}

func (p *GeminiProvider) toolResponsePart(msg Message) geminiPart {
	name := msg.ToolCallID
	content := json.RawMessage(msg.Content)
	// If content is not valid JSON, wrap it as a string value.
	if !json.Valid(content) {
		content, _ = json.Marshal(map[string]string{"result": msg.Content})
	}
	return geminiPart{
		FunctionResponse: &geminiFuncResponse{
			Name:     name,
			Response: content,
		},
	}
}

// parseResponse extracts tool calls and text from the Gemini response.
func (p *GeminiProvider) parseResponse(body []byte) (*Response, error) {
	var gr geminiResponse
	if err := json.Unmarshal(body, &gr); err != nil {
		return nil, fmt.Errorf("gemini: unmarshal response: %w", err)
	}

	if len(gr.Candidates) == 0 {
		return &Response{}, nil
	}

	resp := &Response{}
	for _, part := range gr.Candidates[0].Content.Parts {
		if part.FunctionCall != nil {
			argsJSON, _ := json.Marshal(part.FunctionCall.Args)
			resp.ToolCalls = append(resp.ToolCalls, ToolCall{
				ID:               part.FunctionCall.Name,
				Name:             part.FunctionCall.Name,
				Args:             argsJSON,
				ThoughtSignature: part.ThoughtSignature,
			})
		}
		if part.Text != "" {
			if resp.Content != "" {
				resp.Content += "\n"
			}
			resp.Content += part.Text
		}
	}

	if gr.UsageMetadata != nil {
		resp.Usage = &Usage{
			InputTokens:  gr.UsageMetadata.PromptTokenCount,
			OutputTokens: gr.UsageMetadata.CandidatesTokenCount,
		}
	}

	return resp, nil
}

