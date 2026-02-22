// Package debug provides a web-based DevTools-like UI for debugging agent loops.
// It wraps an Engine[S] by intercepting its LLM provider, tools, chat interface,
// and hooks, broadcasting all events via SSE to an embedded HTML/JS frontend.
package debug

import (
	"context"
	"embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	agent "github.com/logosc/mini-agent-go"
)

//go:embed frontend/dist
var consoleHTML embed.FS

// SessionConfig holds the per-session settings editable in the browser UI.
// All fields are passed to the user factory at session start.
type SessionConfig struct {
	UserID  string `json:"user_id"`
	Model   string `json:"model"`
	APIKey  string `json:"api_key,omitempty"`
	BaseURL string `json:"base_url,omitempty"`
}

// Console wraps an Engine[S] with debug instrumentation and serves a web UI.
//
// Create with NewConsole for a fixed engine, NewConsoleFactory to get a fresh
// engine+state per session, or NewConsoleUserFactory to also support switching
// settings from the browser UI.
type Console[S any] struct {
	// factory is called at the start of each new session (no user ID).
	factory func() (*agent.Engine[S], S)

	// userFactory is called at the start of each new session with the
	// settings chosen in the browser UI. Takes precedence over factory.
	userFactory func(cfg SessionConfig) (*agent.Engine[S], S)

	// defaultUser is the user ID shown in the browser on first load.
	defaultUser string
	// currentUser is the user ID used for the current/last session.
	currentUser string

	// Title is the display name shown in the browser header.
	// If empty, defaults to "Agent Debug Console".
	Title string

	// defaultModel is the model shown in the browser on first load.
	defaultModel string
	// currentModel is the model used for the current/last session.
	currentModel string

	// OnImage, if set, is called whenever the user sends an image.
	OnImage func(state S, imageData []byte, caption string)

	// OnAudio, if set, is called whenever the user sends an audio message.
	// audio is the raw bytes; mimeType is e.g. "audio/wav"; caption is any
	// accompanying text.
	OnAudio func(state S, audio []byte, mimeType, caption string)

	// TranscribeFunc, if set, is called instead of routing audio directly
	// through ReplyCh. The returned text is sent as a normal text Reply.
	// Use this to add STT (speech-to-text) before the agent sees the message.
	// If nil, audio is routed as Reply.AudioData (Path B — native Gemini).
	TranscribeFunc func(ctx context.Context, audio []byte, mimeType string) (string, error)

	// ReplyAggregateWindow, if > 0, causes WaitForReply to coalesce
	// multiple rapid user messages within this duration into a single reply.
	// Default: 0 (disabled).
	ReplyAggregateWindow time.Duration

	// ContextFunc, if set, is called at the start of each new session.
	ContextFunc func(state S) string

	// OnTextInput, if set, is called before user text is sent to the engine.
	// It can modify state and return a replacement text for the LLM.
	// Use this to intercept long pasted content, store it in state, and
	// return a short summary so the LLM context stays small.
	// If nil, text is passed through unchanged.
	OnTextInput func(state S, text string) string

	engine *agent.Engine[S]
	state  S

	broadcaster *broadcaster
	chat        *agent.ChannelChat
	running     bool
	mu          sync.Mutex

	totalInput      int
	totalOutput     int
	totalCacheRead  int
	totalCacheWrite int
}

// NewConsole creates a debug console wrapping the given engine.
// The engine is instrumented once and reused across sessions.
func NewConsole[S any](engine *agent.Engine[S], state S) *Console[S] {
	c := &Console[S]{
		engine:      engine,
		state:       state,
		broadcaster: newBroadcaster(500),
	}
	c.instrument(engine)
	return c
}

// NewConsoleFactory creates a debug console that calls factory() at the start
// of each new conversation session.
func NewConsoleFactory[S any](factory func() (*agent.Engine[S], S)) *Console[S] {
	c := &Console[S]{
		factory:     factory,
		broadcaster: newBroadcaster(500),
	}
	c.chat = &agent.ChannelChat{ReplyCh: make(chan agent.Reply, 1)}
	return c
}

// NewConsoleUserFactory creates a debug console that calls factory(cfg) at the
// start of each new session. The user ID, model, API key, and base URL are all
// editable in the browser UI and persisted in localStorage. defaultUser and
// defaultModel are pre-filled on first load.
//
// Example:
//
//	debug.NewConsoleUserFactory(func(cfg debug.SessionConfig) (*agent.Engine[*S], *S) {
//	    // use cfg.Model, cfg.APIKey, cfg.BaseURL to create provider
//	}, "debug", "claude-sonnet-4-6")
func NewConsoleUserFactory[S any](factory func(cfg SessionConfig) (*agent.Engine[S], S), defaultUser, defaultModel string) *Console[S] {
	c := &Console[S]{
		userFactory:  factory,
		defaultUser:  defaultUser,
		currentUser:  defaultUser,
		defaultModel: defaultModel,
		currentModel: defaultModel,
		broadcaster:  newBroadcaster(500),
	}
	c.chat = &agent.ChannelChat{ReplyCh: make(chan agent.Reply, 1)}
	return c
}

// chatSetter is an optional interface for tools that hold a ChatInterface
// reference. instrument() calls SetChat on any tool that implements it so the
// tool always uses the console's live chat rather than a stale construction-
// time reference (e.g. NullChat).
type chatSetter interface {
	SetChat(agent.ChatInterface)
}

// instrument wires the console into engine: replaces Chat, wraps the LLM
// provider and all tools, and intercepts OnUsage/OnToolDone.
// Called once in fixed mode (NewConsole) or once per session in factory mode.
func (c *Console[S]) instrument(engine *agent.Engine[S]) {
	c.chat = &agent.ChannelChat{
		SendFunc: func(ctx context.Context, text string) error {
			c.emit("chat", map[string]any{"role": "assistant", "text": text})
			return nil
		},
		AttachmentFunc: func(ctx context.Context, att agent.Attachment) error {
			ev := map[string]any{"role": "assistant", "text": att.Caption}
			if att.Type == "image" && len(att.Data) > 0 {
				ev["image_data"] = base64.StdEncoding.EncodeToString(att.Data)
				ev["image_media_type"] = imageMIME(att.Data, att.Name)
			} else if att.Type == "file" && len(att.Data) > 0 {
				mime := fileMIME(att.Data, att.Name)
				if strings.HasPrefix(mime, "audio/") || strings.HasPrefix(mime, "video/") {
					ev["file_data"] = base64.StdEncoding.EncodeToString(att.Data)
					ev["file_mime"] = mime
					ev["file_name"] = att.Name
				} else {
					ev["text"] = fmt.Sprintf("[attachment: %s]", att.Name)
				}
			} else {
				ev["text"] = fmt.Sprintf("[attachment: %s]", att.Name)
			}
			c.emit("chat", ev)
			return nil
		},
		ReplyCh:              make(chan agent.Reply, 1),
		ReplyAggregateWindow: c.ReplyAggregateWindow,
	}
	engine.Chat = c.chat

	// Interactive sessions need a high idle threshold so the agent doesn't
	// exit after a couple of text-only responses.
	if engine.IdleTurnsToExit < 100 {
		engine.IdleTurnsToExit = 100
	}

	engine.LLM = wrapProvider(engine.LLM, c)

	for i, t := range engine.Tools {
		if cs, ok := t.(chatSetter); ok {
			cs.SetChat(c.chat)
		}
		engine.Tools[i] = &debugTool[S]{inner: t, console: c}
	}
	for i, t := range engine.LazyTools {
		if cs, ok := t.(chatSetter); ok {
			cs.SetChat(c.chat)
		}
		engine.LazyTools[i] = &debugTool[S]{inner: t, console: c}
	}

	origOnToolDone := engine.OnToolDone
	engine.OnToolDone = func(state S, name string, result *agent.ToolResult) {
		if origOnToolDone != nil {
			origOnToolDone(state, name, result)
		}
	}

	origOnUsage := engine.OnUsage
	engine.OnUsage = func(u agent.Usage) {
		c.mu.Lock()
		c.totalInput += u.InputTokens
		c.totalOutput += u.OutputTokens
		c.totalCacheRead += u.CacheReadTokens
		c.totalCacheWrite += u.CacheWriteTokens
		c.mu.Unlock()
		c.emit("usage", map[string]any{
			"input_tokens":       u.InputTokens,
			"output_tokens":      u.OutputTokens,
			"cache_read":         u.CacheReadTokens,
			"cache_write":        u.CacheWriteTokens,
			"total_input":        c.totalInput,
			"total_output":       c.totalOutput,
			"total_cache_read":   c.totalCacheRead,
			"total_cache_write":  c.totalCacheWrite,
		})
		if origOnUsage != nil {
			origOnUsage(u)
		}
	}
}

// Handler returns an http.Handler for the debug console.
func (c *Console[S]) Handler() http.Handler {
	mux := http.NewServeMux()
	staticFS, _ := fs.Sub(consoleHTML, "frontend/dist")
	mux.Handle("GET /", http.FileServer(http.FS(staticFS)))
	mux.HandleFunc("GET /events", c.handleSSE)
	mux.HandleFunc("POST /send", c.handleSend)
	mux.HandleFunc("GET /user", c.handleGetUser)
	return mux
}

func (c *Console[S]) handleGetUser(w http.ResponseWriter, r *http.Request) {
	c.mu.Lock()
	cur := c.currentUser
	def := c.defaultUser
	c.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"current_user": cur,
		"default_user": def,
	})
}

// Serve is a one-line launcher for a fixed engine: wraps it, starts an HTTP
// server, and blocks.
func Serve[S any](ctx context.Context, engine *agent.Engine[S], state S, addr string) error {
	return serve(ctx, NewConsole(engine, state), addr)
}

// ServeFactory is a one-line launcher for a factory-based console.
func ServeFactory[S any](ctx context.Context, factory func() (*agent.Engine[S], S), addr string) error {
	return serve(ctx, NewConsoleFactory(factory), addr)
}

// ServeUserFactory is a one-line launcher for a user-and-model-aware console.
// The factory receives the SessionConfig chosen in the browser UI on each new
// session.
//
// Example:
//
//	debug.ServeUserFactory(ctx, myFactory, "debug", "claude-sonnet-4-6", ":9742")
func ServeUserFactory[S any](ctx context.Context, factory func(cfg SessionConfig) (*agent.Engine[S], S), defaultUser, defaultModel, addr string) error {
	return serve(ctx, NewConsoleUserFactory(factory, defaultUser, defaultModel), addr)
}

// Serve starts the HTTP server for this console and blocks until ctx is done.
func (c *Console[S]) Serve(ctx context.Context, addr string) error {
	srv := &http.Server{Addr: addr, Handler: c.Handler()}
	go func() {
		<-ctx.Done()
		srv.Close()
	}()
	log.Printf("[debug] console at http://localhost%s", addr)
	return srv.ListenAndServe()
}

func serve[S any](ctx context.Context, console *Console[S], addr string) error {
	return console.Serve(ctx, addr)
}

func (c *Console[S]) emit(eventType string, data any) {
	b, _ := json.Marshal(data)
	c.broadcaster.send(event{Type: eventType, Data: string(b)})
}

func (c *Console[S]) emitConfig() {
	if c.engine == nil {
		return
	}
	tools := make([]map[string]any, 0, len(c.engine.Tools)+len(c.engine.LazyTools))
	for _, t := range c.engine.Tools {
		tools = append(tools, map[string]any{
			"name":        t.Name(),
			"description": t.Description(),
			"parameters":  json.RawMessage(t.Parameters()),
		})
	}
	for _, t := range c.engine.LazyTools {
		tools = append(tools, map[string]any{
			"name":        t.Name(),
			"description": t.Description(),
			"parameters":  json.RawMessage(t.Parameters()),
			"lazy":        true,
		})
	}
	c.emit("config", map[string]any{
		"title":          c.Title,
		"system_prompt":  c.engine.SystemPrompt,
		"tools":          tools,
		"max_iterations": c.engine.MaxIterations,
		"current_user":   c.currentUser,
		"default_user":   c.defaultUser,
		"current_model":  c.currentModel,
		"default_model":  c.defaultModel,
	})
}

func (c *Console[S]) handleSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ch := c.broadcaster.subscribe()
	defer c.broadcaster.unsubscribe(ch)

	ctx := r.Context()
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				return
			}
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Type, ev.Data)
			flusher.Flush()
		case <-ctx.Done():
			return
		}
	}
}

func (c *Console[S]) handleSend(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Text           string `json:"text"`
		ImageData      string `json:"image_data"`       // base64-encoded
		ImageMediaType string `json:"image_media_type"` // e.g. "image/png"
		AudioData      string `json:"audio_data"`       // base64-encoded
		AudioMediaType string `json:"audio_media_type"` // e.g. "audio/wav"
		UserID         string `json:"user_id"`          // override user for new sessions
		Model          string `json:"model"`            // override model for new sessions
		APIKey         string `json:"api_key"`          // API key for new sessions
		BaseURL        string `json:"base_url"`         // custom base URL for new sessions
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad request", 400)
		return
	}

	var imageBytes []byte
	if body.ImageData != "" {
		var err error
		imageBytes, err = base64.StdEncoding.DecodeString(body.ImageData)
		if err != nil {
			http.Error(w, "invalid image_data base64", 400)
			return
		}
		log.Printf("[debug] received image (%d bytes, mime=%s) with text=%q", len(imageBytes), body.ImageMediaType, body.Text)
	}

	var audioBytes []byte
	if body.AudioData != "" {
		var err error
		audioBytes, err = base64.StdEncoding.DecodeString(body.AudioData)
		if err != nil {
			http.Error(w, "invalid audio_data base64", 400)
			return
		}
		log.Printf("[debug] received audio (%d bytes, mime=%s) with text=%q", len(audioBytes), body.AudioMediaType, body.Text)
	}

	c.mu.Lock()
	stopped := !c.running
	if stopped {
		// Determine user ID and model for this session.
		userID := body.UserID
		if userID == "" {
			userID = c.currentUser
		}
		if userID == "" {
			userID = c.defaultUser
		}
		c.currentUser = userID

		model := body.Model
		if model == "" {
			model = c.currentModel
		}
		if model == "" {
			model = c.defaultModel
		}
		c.currentModel = model

		// Spin up a fresh engine+state for each new session.
		if c.userFactory != nil {
			engine, state := c.userFactory(SessionConfig{
				UserID:  userID,
				Model:   model,
				APIKey:  body.APIKey,
				BaseURL: body.BaseURL,
			})
			c.engine = engine
			c.state = state
			c.totalInput = 0
			c.totalOutput = 0
			c.totalCacheRead = 0
			c.totalCacheWrite = 0
			c.instrument(engine)
		} else if c.factory != nil {
			engine, state := c.factory()
			c.engine = engine
			c.state = state
			c.totalInput = 0
			c.totalOutput = 0
			c.totalCacheRead = 0
			c.totalCacheWrite = 0
			c.instrument(engine)
		}
		c.running = true
	}
	// Route incoming image into state before the agent sees the reply.
	if len(imageBytes) > 0 && c.OnImage != nil && c.engine != nil {
		c.OnImage(c.state, imageBytes, body.Text)
	}
	if len(audioBytes) > 0 && c.OnAudio != nil && c.engine != nil {
		c.OnAudio(c.state, audioBytes, body.AudioMediaType, body.Text)
	}
	c.mu.Unlock()

	// Always echo the user message as a chat event so it is stored in the
	// broadcaster history and survives SSE reconnects.
	userEv := map[string]any{"role": "user", "text": body.Text}
	if len(imageBytes) > 0 {
		userEv["image_data"] = base64.StdEncoding.EncodeToString(imageBytes)
		userEv["image_media_type"] = imageMIME(imageBytes, "")
	}
	if len(audioBytes) > 0 {
		userEv["audio_data"] = base64.StdEncoding.EncodeToString(audioBytes)
		userEv["audio_media_type"] = body.AudioMediaType
	}

	if stopped {
		c.broadcaster.clear()
		c.emitConfig()
		c.emit("chat", userEv)
		c.emit("state", map[string]any{"running": true})

		go func() {
			taskText := body.Text
			if c.OnTextInput != nil {
				taskText = c.OnTextInput(c.state, taskText)
			}
			var err error
			if len(audioBytes) > 0 {
				// Include audio in the very first LLM call, not via pre-queue.
				err = c.engine.RunWithReply(context.Background(), c.state, agent.Reply{
					Text:           taskText,
					AudioData:      audioBytes,
					AudioMediaType: body.AudioMediaType,
				})
			} else {
				err = c.engine.Run(context.Background(), c.state, taskText)
			}
			c.mu.Lock()
			c.running = false
			c.mu.Unlock()
			if err != nil {
				c.emit("error", map[string]any{"error": err.Error()})
			}
			c.emit("state", map[string]any{"running": false})
		}()
	} else {
		c.emit("chat", userEv)
		if len(audioBytes) > 0 {
			if c.TranscribeFunc != nil {
				text, err := c.TranscribeFunc(r.Context(), audioBytes, body.AudioMediaType)
				if err != nil {
					log.Printf("[debug] TranscribeFunc error: %v", err)
					text = body.Text // fallback to caption
				}
				reply := agent.Reply{Text: text}
				select {
				case c.chat.ReplyCh <- reply:
				default:
					c.chat.BufferMessage(text)
				}
			} else {
				// Path B: native audio — route bytes directly.
				reply := agent.Reply{Text: body.Text, AudioData: audioBytes, AudioMediaType: body.AudioMediaType}
				select {
				case c.chat.ReplyCh <- reply:
				default:
					c.chat.BufferMessageWithAudio(body.Text, audioBytes, body.AudioMediaType)
				}
			}
		} else {
			replyText := body.Text
			if c.OnTextInput != nil {
				replyText = c.OnTextInput(c.state, replyText)
			}
			reply := agent.Reply{Text: replyText}
			select {
			case c.chat.ReplyCh <- reply:
			default:
				c.chat.BufferMessage(replyText)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "sent"})
}

// fileMIME detects the MIME type of audio/video files from magic bytes or filename.
func fileMIME(data []byte, name string) string {
	// MP3: ID3 tag or MPEG sync word
	if len(data) >= 3 && string(data[:3]) == "ID3" {
		return "audio/mpeg"
	}
	if len(data) >= 2 && data[0] == 0xFF && (data[1]&0xE0) == 0xE0 {
		return "audio/mpeg"
	}
	// OGG
	if len(data) >= 4 && string(data[:4]) == "OggS" {
		return "audio/ogg"
	}
	// WAV (RIFF....WAVE)
	if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE" {
		return "audio/wav"
	}
	// MP4/M4A/MOV (ftyp box)
	if len(data) >= 8 && string(data[4:8]) == "ftyp" {
		return "video/mp4"
	}
	// WebM (EBML header — Matroska/WebM)
	if len(data) >= 4 && data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
		return "video/webm"
	}
	// Fall back to extension
	if i := strings.LastIndex(name, "."); i >= 0 {
		switch strings.ToLower(name[i+1:]) {
		case "mp3":
			return "audio/mpeg"
		case "wav":
			return "audio/wav"
		case "ogg":
			return "audio/ogg"
		case "m4a", "aac":
			return "audio/mp4"
		case "mp4", "m4v":
			return "video/mp4"
		case "webm":
			return "video/webm"
		case "mov":
			return "video/quicktime"
		}
	}
	return "application/octet-stream"
}

// imageMIME detects the MIME type from magic bytes, falling back to the filename.
func imageMIME(data []byte, name string) string {
	if len(data) >= 2 && data[0] == 0x89 && data[1] == 0x50 {
		return "image/png"
	}
	if len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return "image/jpeg"
	}
	if len(data) >= 6 && string(data[:6]) == "GIF87a" || len(data) >= 6 && string(data[:6]) == "GIF89a" {
		return "image/gif"
	}
	if len(data) >= 4 && string(data[:4]) == "RIFF" {
		return "image/webp"
	}
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '.' {
			switch name[i+1:] {
			case "jpg", "jpeg":
				return "image/jpeg"
			case "gif":
				return "image/gif"
			case "webp":
				return "image/webp"
			}
			return "image/png"
		}
	}
	return "image/png"
}

// --- SSE Broadcaster ---

type event struct {
	Type string
	Data string
}

type broadcaster struct {
	mu      sync.RWMutex
	clients map[chan event]struct{}
	history []event
	maxHist int
}

func newBroadcaster(maxHistory int) *broadcaster {
	return &broadcaster{
		clients: make(map[chan event]struct{}),
		maxHist: maxHistory,
	}
}

func (b *broadcaster) send(ev event) {
	b.mu.Lock()
	b.history = append(b.history, ev)
	if len(b.history) > b.maxHist {
		b.history = b.history[len(b.history)-b.maxHist:]
	}
	clients := make([]chan event, 0, len(b.clients))
	for ch := range b.clients {
		clients = append(clients, ch)
	}
	b.mu.Unlock()

	for _, ch := range clients {
		select {
		case ch <- ev:
		default:
		}
	}
}

func (b *broadcaster) subscribe() chan event {
	ch := make(chan event, 64)
	b.mu.Lock()
	for _, ev := range b.history {
		ch <- ev
	}
	b.clients[ch] = struct{}{}
	b.mu.Unlock()
	return ch
}

func (b *broadcaster) clear() {
	b.mu.Lock()
	b.history = nil
	b.mu.Unlock()
}

func (b *broadcaster) unsubscribe(ch chan event) {
	b.mu.Lock()
	delete(b.clients, ch)
	b.mu.Unlock()
	close(ch)
}

// --- Debug Provider ---

type providerWrapper struct {
	inner agent.LLMProvider
	emit  func(string, any)
}

func (p *providerWrapper) Chat(ctx context.Context, messages []agent.Message, tools []agent.ToolDef) (*agent.Response, error) {
	return p.inner.Chat(ctx, messages, tools)
}

func (p *providerWrapper) StreamChat(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, onText func(chunk string)) (*agent.Response, error) {
	streamer, ok := p.inner.(agent.StreamingLLMProvider)
	if !ok {
		resp, err := p.inner.Chat(ctx, messages, tools)
		if err != nil {
			return nil, err
		}
		if resp.Content != "" {
			onText(resp.Content)
		}
		return resp, nil
	}
	wrapped := func(chunk string) {
		p.emit("stream", map[string]any{"text": chunk})
		onText(chunk)
	}
	return streamer.StreamChat(ctx, messages, tools, wrapped)
}

var _ agent.StreamingLLMProvider = (*providerWrapper)(nil)

func wrapProvider[S any](inner agent.LLMProvider, c *Console[S]) agent.LLMProvider {
	return &providerWrapper{inner: inner, emit: c.emit}
}

// --- Debug Tool ---

type debugTool[S any] struct {
	inner   agent.Tool[S]
	console *Console[S]
}

func (t *debugTool[S]) Name() string               { return t.inner.Name() }
func (t *debugTool[S]) Description() string        { return t.inner.Description() }
func (t *debugTool[S]) Parameters() json.RawMessage { return t.inner.Parameters() }

func (t *debugTool[S]) Execute(ctx context.Context, state S, args json.RawMessage) (*agent.ToolResult, error) {
	t.console.emit("tool_start", map[string]any{
		"name": t.inner.Name(),
		"args": json.RawMessage(args),
	})

	start := time.Now()
	result, err := t.inner.Execute(ctx, state, args)
	elapsed := time.Since(start)

	data := map[string]any{
		"name":        t.inner.Name(),
		"duration_ms": elapsed.Milliseconds(),
	}
	if err != nil {
		data["is_error"] = true
		data["summary"] = err.Error()
	} else if result != nil {
		data["summary"] = truncate(result.Summary, 500)
		data["is_error"] = result.IsError
	}
	t.console.emit("tool_done", data)

	return result, err
}

// --- Helpers ---

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
