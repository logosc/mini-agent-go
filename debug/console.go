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
	"sync"
	"time"

	agent "github.com/logosc/mini-agent-go"
)

//go:embed frontend/dist
var consoleHTML embed.FS

// Console wraps an Engine[S] with debug instrumentation and serves a web UI.
//
// Create with NewConsole for a fixed engine, NewConsoleFactory to get a fresh
// engine+state per session, or NewConsoleUserFactory to also support switching
// the user ID from the browser UI.
type Console[S any] struct {
	// factory is called at the start of each new session (no user ID).
	factory func() (*agent.Engine[S], S)

	// userFactory is called at the start of each new session with the user ID
	// and model chosen in the browser UI. Takes precedence over factory.
	userFactory func(userID, model string) (*agent.Engine[S], S)

	// defaultUser is the user ID shown in the browser on first load.
	defaultUser string
	// currentUser is the user ID used for the current/last session.
	currentUser string

	// defaultModel is the model shown in the browser on first load.
	defaultModel string
	// currentModel is the model used for the current/last session.
	currentModel string

	// OnImage, if set, is called whenever the user sends an image.
	OnImage func(state S, imageData []byte, caption string)

	// ContextFunc, if set, is called at the start of each new session.
	ContextFunc func(state S) string

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

// NewConsoleUserFactory creates a debug console that calls factory(userID, model)
// at the start of each new session. Both the user ID and model are editable in
// the browser UI. defaultUser and defaultModel are pre-filled on first load.
//
// Example:
//
//	debug.NewConsoleUserFactory(myAgent.NewDebugSession, "debug", "claude-sonnet-4-6")
func NewConsoleUserFactory[S any](factory func(userID, model string) (*agent.Engine[S], S), defaultUser, defaultModel string) *Console[S] {
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
			} else {
				ev["text"] = fmt.Sprintf("[attachment: %s]", att.Name)
			}
			c.emit("chat", ev)
			return nil
		},
		ReplyCh: make(chan agent.Reply, 1),
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
// The factory receives the user ID and model chosen in the browser UI on each
// new session.
//
// Example:
//
//	debug.ServeUserFactory(ctx, myAgent.NewDebugSession, "debug", "claude-sonnet-4-6", ":9742")
func ServeUserFactory[S any](ctx context.Context, factory func(userID, model string) (*agent.Engine[S], S), defaultUser, defaultModel, addr string) error {
	return serve(ctx, NewConsoleUserFactory(factory, defaultUser, defaultModel), addr)
}

func serve[S any](ctx context.Context, console *Console[S], addr string) error {
	srv := &http.Server{Addr: addr, Handler: console.Handler()}
	go func() {
		<-ctx.Done()
		srv.Close()
	}()
	log.Printf("[debug] console at http://localhost%s", addr)
	return srv.ListenAndServe()
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
		UserID         string `json:"user_id"`          // override user for new sessions
		Model          string `json:"model"`            // override model for new sessions
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
			engine, state := c.userFactory(userID, model)
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
	c.mu.Unlock()

	// Always echo the user message as a chat event so it is stored in the
	// broadcaster history and survives SSE reconnects.
	userEv := map[string]any{"role": "user", "text": body.Text}
	if len(imageBytes) > 0 {
		userEv["image_data"] = base64.StdEncoding.EncodeToString(imageBytes)
		userEv["image_media_type"] = imageMIME(imageBytes, "")
	}

	if stopped {
		c.broadcaster.clear()
		c.emitConfig()
		c.emit("chat", userEv)
		c.emit("state", map[string]any{"running": true})
		go func() {
			err := c.engine.Run(context.Background(), c.state, body.Text)
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
		reply := agent.Reply{Text: body.Text, ImageData: imageBytes}
		select {
		case c.chat.ReplyCh <- reply:
		default:
			if len(imageBytes) > 0 {
				c.chat.BufferMessageWithImage(body.Text, imageBytes)
			} else {
				c.chat.BufferMessage(body.Text)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "sent"})
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
