// Example: A podcast producer agent that turns URLs or pasted text into
// two-host podcast MP3s using LLM script generation and Gemini TTS.
//
// Usage:
//
//	export GOOGLE_API_KEY=...
//	export ANTHROPIC_API_KEY=sk-...
//	go run . -model claude-sonnet-4-6
//
// Or with Gemini for both LLM and TTS:
//
//	export GOOGLE_API_KEY=...
//	go run . -model gemini-3-flash-preview
//
// Then open the debug console in your browser.
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"regexp"
	"strings"
	"time"

	readability "github.com/go-shiori/go-readability"
	agent "github.com/logosc/mini-agent-go"
	"github.com/logosc/mini-agent-go/debug"
)

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

// PodcastState holds all intermediate data between tool calls.
type PodcastState struct {
	SourceURL  string
	Title      string
	SourceText string
	Script     []Segment
	AudioFile  []byte
}

// Segment is one turn in the podcast script.
type Segment struct {
	Host string // "A" or "B"
	Text string
}

// ---------------------------------------------------------------------------
// Tool 1: fetch_content
// ---------------------------------------------------------------------------

type FetchContentTool struct{}

func (t *FetchContentTool) Name() string        { return "fetch_content" }
func (t *FetchContentTool) Description() string { return "Fetch a URL and extract its readable text content. If the URL is empty, pass the user's pasted text in the 'text' parameter instead." }
func (t *FetchContentTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"url":  {"type": "string", "description": "URL to fetch content from"},
			"text": {"type": "string", "description": "Pasted text if no URL is available"}
		}
	}`)
}

func (t *FetchContentTool) Execute(ctx context.Context, state *PodcastState, args json.RawMessage) (*agent.ToolResult, error) {
	var p struct {
		URL  string `json:"url"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(args, &p); err != nil {
		return agent.ToolResult{}.WithError("invalid args: " + err.Error()), nil
	}

	if p.URL != "" {
		article, err := readability.FromURL(p.URL, 30*time.Second)
		if err != nil {
			return agent.ToolResult{}.WithError(fmt.Sprintf("failed to fetch URL: %v — ask the user to paste the text instead", err)), nil
		}
		state.SourceURL = p.URL
		state.Title = article.Title
		text := strings.TrimSpace(article.TextContent)
		if len(text) < 50 {
			return agent.ToolResult{}.WithError(fmt.Sprintf("URL fetched but content too short (%d chars) — the site may block scraping. Ask the user to paste the article text instead.", len(text))), nil
		}
		state.SourceText = truncateText(text, 50000)
	} else if p.Text != "" {
		state.Title = "User-provided content"
		state.SourceText = truncateText(p.Text, 50000)
	} else {
		return agent.ToolResult{}.WithError("provide either a 'url' or 'text' parameter"), nil
	}

	return &agent.ToolResult{
		Summary: fmt.Sprintf("Fetched content: %q (%d chars)", state.Title, len(state.SourceText)),
	}, nil
}

// ---------------------------------------------------------------------------
// Tool 2: generate_script
// ---------------------------------------------------------------------------

type GenerateScriptTool struct {
	LLM agent.LLMProvider
}

func (t *GenerateScriptTool) Name() string        { return "generate_script" }
func (t *GenerateScriptTool) Description() string { return "Generate a two-host podcast script from the fetched content. Must call fetch_content first." }
func (t *GenerateScriptTool) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"style":       {"type": "string",  "description": "Script style, e.g. 'casual', 'educational', 'debate'. Default: casual"},
			"max_minutes": {"type": "integer", "description": "Target podcast length in minutes. Default: 5"}
		}
	}`)
}

const scriptSystemPrompt = `You are a podcast script writer. Generate a two-host conversational podcast script based on the source material provided.

Rules:
- Use exactly two hosts: "A" (female host, knowledgeable) and "B" (male host, curious and witty)
- Write natural, conversational dialogue — not a lecture
- Each line must be prefixed with "A:" or "B:" on its own line
- Include an intro, main discussion, and a brief outro
- Keep it engaging with back-and-forth exchanges
- Target approximately %d minutes of speech (roughly %d words)
- Style: %s

Output ONLY the script lines, no other commentary.`

func (t *GenerateScriptTool) Execute(ctx context.Context, state *PodcastState, args json.RawMessage) (*agent.ToolResult, error) {
	if state.SourceText == "" {
		return agent.ToolResult{}.WithError("no source text — call fetch_content first"), nil
	}

	var p struct {
		Style      string `json:"style"`
		MaxMinutes int    `json:"max_minutes"`
	}
	if err := json.Unmarshal(args, &p); err != nil {
		return agent.ToolResult{}.WithError("invalid args: " + err.Error()), nil
	}
	if p.Style == "" {
		p.Style = "casual and engaging"
	}
	if p.MaxMinutes <= 0 {
		p.MaxMinutes = 5
	}

	wordsTarget := p.MaxMinutes * 150 // ~150 wpm for natural speech

	messages := []agent.Message{
		{Role: "system", Content: fmt.Sprintf(scriptSystemPrompt, p.MaxMinutes, wordsTarget, p.Style)},
		{Role: "user", Content: fmt.Sprintf("Source material title: %s\n\n%s", state.Title, state.SourceText)},
	}

	log.Printf("[podcast] generating script (%d min, style=%s)", p.MaxMinutes, p.Style)
	resp, err := t.LLM.Chat(ctx, messages, nil)
	if err != nil {
		return agent.ToolResult{}.WithError(fmt.Sprintf("LLM script generation failed: %v", err)), nil
	}

	segments := parseScript(resp.Content)
	if len(segments) == 0 {
		return agent.ToolResult{}.WithError("failed to parse script — no A:/B: lines found in LLM output"), nil
	}
	state.Script = segments

	// Build preview
	preview := fmt.Sprintf("Generated %d-segment script for %q:\n", len(segments), state.Title)
	for i, seg := range segments {
		line := seg.Text
		if len(line) > 80 {
			line = line[:80] + "..."
		}
		preview += fmt.Sprintf("  %s: %s\n", seg.Host, line)
		if i >= 5 {
			preview += fmt.Sprintf("  ... (%d more segments)\n", len(segments)-6)
			break
		}
	}

	return &agent.ToolResult{
		Summary:     preview,
		ChatMessage: fmt.Sprintf("Script ready: %d segments for %q", len(segments), state.Title),
	}, nil
}

// parseScript extracts Host A / Host B segments from script text.
func parseScript(text string) []Segment {
	var segments []Segment
	re := regexp.MustCompile(`(?m)^([AB]):\s*(.+)`)
	for _, match := range re.FindAllStringSubmatch(text, -1) {
		segments = append(segments, Segment{
			Host: match[1],
			Text: strings.TrimSpace(match[2]),
		})
	}
	return segments
}

// ---------------------------------------------------------------------------
// Tool 3: generate_audio
// ---------------------------------------------------------------------------

type GenerateAudioTool struct {
	GoogleAPIKey string
	FFmpegPath   string
}

func (t *GenerateAudioTool) Name() string        { return "generate_audio" }
func (t *GenerateAudioTool) Description() string { return "Generate the podcast MP3 from the script. Must call generate_script first." }
func (t *GenerateAudioTool) Parameters() json.RawMessage {
	return json.RawMessage(`{"type": "object"}`)
}

func (t *GenerateAudioTool) Execute(ctx context.Context, state *PodcastState, args json.RawMessage) (*agent.ToolResult, error) {
	if len(state.Script) == 0 {
		return agent.ToolResult{}.WithError("no script — call generate_script first"), nil
	}
	if t.GoogleAPIKey == "" {
		return agent.ToolResult{}.WithError("GOOGLE_API_KEY is required for TTS"), nil
	}

	log.Printf("[podcast] generating audio for %d segments", len(state.Script))

	var allMP3s [][]byte
	for i, seg := range state.Script {
		voice := "Kore" // Host A — female
		if seg.Host == "B" {
			voice = "Charon" // Host B — male
		}

		mp3, err := generateTTS(ctx, t.GoogleAPIKey, seg.Text, voice, t.FFmpegPath)
		if err != nil {
			return agent.ToolResult{}.WithError(fmt.Sprintf("TTS failed on segment %d: %v", i+1, err)), nil
		}
		log.Printf("[podcast] segment %d/%d (%s): %d bytes", i+1, len(state.Script), seg.Host, len(mp3))
		allMP3s = append(allMP3s, mp3)
	}

	final, err := concatMP3s(t.FFmpegPath, allMP3s)
	if err != nil {
		return agent.ToolResult{}.WithError(fmt.Sprintf("MP3 concat failed: %v", err)), nil
	}
	state.AudioFile = final

	log.Printf("[podcast] audio complete: %d bytes", len(final))

	return &agent.ToolResult{
		Summary: fmt.Sprintf("Generated podcast MP3: %d segments, %.1f MB", len(state.Script), float64(len(final))/(1024*1024)),
		Attachments: []agent.Attachment{
			{Type: "file", Name: "podcast.mp3", Data: final},
		},
	}, nil
}

// ---------------------------------------------------------------------------
// Gemini TTS
// ---------------------------------------------------------------------------

const (
	geminiTTSModel = "gemini-2.5-flash-preview-tts"
	geminiAPIBase  = "https://generativelanguage.googleapis.com/v1beta/models"
	maxTTSRetries  = 3
)

// generateTTS calls Gemini TTS to convert text to MP3 via PCM → ffmpeg.
func generateTTS(ctx context.Context, apiKey, text, voiceName, ffmpegPath string) ([]byte, error) {
	body := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]interface{}{
					{"text": text},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"responseModalities": []string{"AUDIO"},
			"speechConfig": map[string]interface{}{
				"voiceConfig": map[string]interface{}{
					"prebuiltVoiceConfig": map[string]interface{}{
						"voiceName": voiceName,
					},
				},
			},
		},
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal TTS request: %w", err)
	}

	url := fmt.Sprintf("%s/%s:generateContent?key=%s", geminiAPIBase, geminiTTSModel, apiKey)

	var pcmData []byte
	var lastErr error

	for attempt := 0; attempt < maxTTSRetries; attempt++ {
		if attempt > 0 {
			wait := time.Duration(3*(attempt)) * time.Second
			log.Printf("[tts] retry %d/%d (waiting %v): %v", attempt+1, maxTTSRetries, wait, lastErr)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, fmt.Errorf("create TTS request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("http call: %w", err)
			continue
		}

		respBytes, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("read TTS response: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("API error %d: %s", resp.StatusCode, truncateText(string(respBytes), 200))
			if resp.StatusCode == 429 || resp.StatusCode >= 500 {
				continue
			}
			return nil, fmt.Errorf("gemini TTS: %v", lastErr)
		}

		var result struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						InlineData *struct {
							MIMEType string `json:"mimeType"`
							Data     string `json:"data"`
						} `json:"inlineData,omitempty"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}

		if err := json.Unmarshal(respBytes, &result); err != nil {
			return nil, fmt.Errorf("unmarshal TTS response: %w", err)
		}

		for _, cand := range result.Candidates {
			for _, part := range cand.Content.Parts {
				if part.InlineData != nil {
					decoded, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
					if err != nil {
						return nil, fmt.Errorf("decode TTS audio: %w", err)
					}
					pcmData = decoded
					break
				}
			}
			if pcmData != nil {
				break
			}
		}
		if pcmData != nil {
			break
		}
		lastErr = fmt.Errorf("empty audio response")
	}

	if pcmData == nil {
		return nil, fmt.Errorf("no TTS audio after %d attempts: %v", maxTTSRetries, lastErr)
	}

	return pcmToMP3(ctx, ffmpegPath, pcmData)
}

// pcmToMP3 converts raw PCM L16 24kHz mono to MP3 via ffmpeg stdin/stdout.
func pcmToMP3(ctx context.Context, ffmpegPath string, pcmData []byte) ([]byte, error) {
	cmd := exec.CommandContext(ctx, ffmpegPath,
		"-y",
		"-f", "s16le",
		"-ar", "24000",
		"-ac", "1",
		"-i", "pipe:0",
		"-acodec", "libmp3lame",
		"-ar", "44100",
		"-ab", "128k",
		"-f", "mp3",
		"pipe:1",
	)
	cmd.Stdin = bytes.NewReader(pcmData)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg PCM→MP3 failed: %w\nstderr: %s", err, stderr.String())
	}
	return stdout.Bytes(), nil
}

// concatMP3s concatenates multiple MP3 byte slices into one using ffmpeg.
func concatMP3s(ffmpegPath string, segments [][]byte) ([]byte, error) {
	if len(segments) == 1 {
		return segments[0], nil
	}

	// Write segments to temp files for ffmpeg concat.
	tmpDir, err := os.MkdirTemp("", "podcast-concat-")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	var listLines []string
	for i, seg := range segments {
		path := fmt.Sprintf("%s/seg_%03d.mp3", tmpDir, i)
		if err := os.WriteFile(path, seg, 0644); err != nil {
			return nil, fmt.Errorf("write segment %d: %w", i, err)
		}
		listLines = append(listLines, fmt.Sprintf("file '%s'", path))
	}

	listPath := tmpDir + "/concat_list.txt"
	if err := os.WriteFile(listPath, []byte(strings.Join(listLines, "\n")+"\n"), 0644); err != nil {
		return nil, fmt.Errorf("write concat list: %w", err)
	}

	outPath := tmpDir + "/output.mp3"
	cmd := exec.Command(ffmpegPath,
		"-y",
		"-f", "concat",
		"-safe", "0",
		"-i", listPath,
		"-c:a", "libmp3lame",
		"-b:a", "128k",
		outPath,
	)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg concat failed: %w\nstderr: %s", err, stderr.String())
	}

	return os.ReadFile(outPath)
}

// ---------------------------------------------------------------------------
// Provider selection (same pattern as todo-agent)
// ---------------------------------------------------------------------------

func selectProvider(model, apiKey, baseURL string) agent.LLMProvider {
	switch {
	case strings.HasPrefix(model, "gpt-") || strings.HasPrefix(model, "o3") || strings.HasPrefix(model, "o4"):
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		return agent.NewOpenAICompatibleProvider(agent.OpenAICompatibleOptions{
			APIKey:  apiKey,
			Model:   model,
			BaseURL: baseURL,
		})
	case strings.HasPrefix(model, "gemini"):
		if apiKey == "" {
			apiKey = os.Getenv("GEMINI_API_KEY")
		}
		return agent.NewGeminiProvider(agent.GeminiOptions{
			APIKey:  apiKey,
			Model:   model,
			BaseURL: baseURL,
		})
	default: // Anthropic
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		return agent.NewAnthropicProvider(agent.AnthropicOptions{
			APIKey:  apiKey,
			Model:   model,
			BaseURL: baseURL,
		})
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const systemPrompt = `You are a podcast producer agent. You turn articles, blog posts, and web content into engaging two-host podcast episodes.

Your workflow:
1. Ask the user for a URL or text to turn into a podcast
2. Use fetch_content to get the source material — pass the URL in the "url" parameter, OR if the user pasted text directly, pass it in the "text" parameter
3. Use generate_script to create a two-host conversational script
4. Use generate_audio to produce the final MP3

IMPORTANT: Sometimes the source text is pre-loaded into state automatically (e.g. when the user pastes a long article). When you see a message like "source text is already loaded", skip fetch_content and call generate_script directly. The text is in state even though you can't see it.

Keep your messages concise. Always use the tools — don't write podcast scripts yourself.`

func main() {
	model := flag.String("model", "claude-sonnet-4-6", "LLM model")
	port := flag.String("port", ":8080", "debug console port")
	defaultTTSKey := os.Getenv("GOOGLE_API_KEY")
	if defaultTTSKey == "" {
		defaultTTSKey = os.Getenv("GEMINI_API_KEY")
	}
	googleKey := flag.String("google-key", defaultTTSKey, "Google API key for Gemini TTS")
	flag.Parse()

	factory := func(cfg debug.SessionConfig) (*agent.Engine[*PodcastState], *PodcastState) {
		state := &PodcastState{}
		llm := selectProvider(cfg.Model, cfg.APIKey, cfg.BaseURL)
		engine := &agent.Engine[*PodcastState]{
			LLM: llm,
			Tools: []agent.Tool[*PodcastState]{
				&FetchContentTool{},
				&GenerateScriptTool{LLM: llm},
				&GenerateAudioTool{GoogleAPIKey: *googleKey, FFmpegPath: "ffmpeg"},
			},
			SystemPrompt:    systemPrompt,
			MaxIterations:   30,
			IdleTurnsToExit: 100,
		}
		return engine, state
	}

	console := debug.NewConsoleUserFactory(factory, "user", *model)
	console.Title = "Podcast Agent"
	console.ReplyAggregateWindow = 2 * time.Second
	console.OnTextInput = func(state *PodcastState, text string) string {
		if len(text) > 500 && state.SourceText == "" {
			state.SourceText = truncateText(text, 50000)
			state.Title = "User-provided content"
			// Count words for a rough summary
			words := len(strings.Fields(text))
			log.Printf("[podcast] long text input intercepted (%d chars, ~%d words) → stored in state", len(text), words)
			return fmt.Sprintf("[System: the user pasted a %d-word article which has been pre-loaded into state. Source text is ready. Call generate_script now to create the podcast script.]", words)
		}
		return text
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	log.Printf("Podcast Agent — debug console at http://localhost%s", *port)
	if err := console.Serve(ctx, *port); err != nil {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}
