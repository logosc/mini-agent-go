package agent

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

// Reply holds the user's response.
type Reply struct {
	Text           string
	ImageData      []byte // Optional image data.
	AudioData      []byte // Optional raw audio bytes.
	AudioMediaType string // MIME type, e.g. "audio/wav", "audio/ogg".
}

// BufferedMessage holds a message that arrived while the agent was busy.
type BufferedMessage struct {
	Text           string
	ImageData      []byte
	AudioData      []byte
	AudioMediaType string
}

// ChatInterface bridges the agent with a user for interactive mode.
type ChatInterface interface {
	Send(ctx context.Context, text string) error
	SendAttachment(ctx context.Context, att Attachment) error
	WaitForReply(ctx context.Context) (Reply, error)
	DrainMessages() []BufferedMessage
}

// NullChat is a no-op ChatInterface for headless/background mode.
type NullChat struct{}

func (NullChat) Send(ctx context.Context, text string) error              { return nil }
func (NullChat) SendAttachment(ctx context.Context, att Attachment) error  { return nil }
func (NullChat) WaitForReply(ctx context.Context) (Reply, error) {
	return Reply{}, fmt.Errorf("non-interactive mode: no user available")
}
func (NullChat) DrainMessages() []BufferedMessage { return nil }

// ChannelChat implements ChatInterface using a send function and reply channel.
type ChannelChat struct {
	SendFunc       func(ctx context.Context, text string) error
	AttachmentFunc func(ctx context.Context, att Attachment) error
	ReplyCh        chan Reply

	// ReplyAggregateWindow, if > 0, causes WaitForReply to wait this long
	// after each incoming message for additional messages before returning.
	// Multiple texts are joined with "\n"; last image wins; last audio wins. Default: 0 (off).
	ReplyAggregateWindow time.Duration

	mu       sync.Mutex
	messages []BufferedMessage
}

func (c *ChannelChat) Send(ctx context.Context, text string) error {
	if c.SendFunc != nil {
		return c.SendFunc(ctx, text)
	}
	return nil
}

func (c *ChannelChat) SendAttachment(ctx context.Context, att Attachment) error {
	if c.AttachmentFunc != nil {
		return c.AttachmentFunc(ctx, att)
	}
	return nil
}

func (c *ChannelChat) WaitForReply(ctx context.Context) (Reply, error) {
	// Wait for the first message.
	var first Reply
	select {
	case first = <-c.ReplyCh:
	case <-ctx.Done():
		return Reply{}, ctx.Err()
	}

	window := c.ReplyAggregateWindow
	if window <= 0 {
		return first, nil
	}

	// Aggregate additional messages within the window.
	texts := []string{first.Text}
	imageData := first.ImageData
	audioData := first.AudioData
	audioMediaType := first.AudioMediaType
	timer := time.NewTimer(window)
	defer timer.Stop()
	for {
		select {
		case msg := <-c.ReplyCh:
			texts = append(texts, msg.Text)
			if len(msg.ImageData) > 0 {
				imageData = msg.ImageData // last image wins
			}
			if len(msg.AudioData) > 0 {
				audioData = msg.AudioData           // last audio wins
				audioMediaType = msg.AudioMediaType
			}
			timer.Reset(window)
		case <-timer.C:
			combined := strings.Join(texts, "\n")
			if len(texts) > 1 {
				log.Printf("[agent-chat] aggregated %d replies within %v window", len(texts), window)
			}
			return Reply{Text: combined, ImageData: imageData, AudioData: audioData, AudioMediaType: audioMediaType}, nil
		case <-ctx.Done():
			return Reply{}, ctx.Err()
		}
	}
}

// BufferMessage stores a text message that arrived while the agent was busy.
// Thread-safe.
func (c *ChannelChat) BufferMessage(text string) {
	c.mu.Lock()
	c.messages = append(c.messages, BufferedMessage{Text: text})
	c.mu.Unlock()
	log.Printf("[agent-chat] buffered message (text=%q), total buffered: %d", text, len(c.messages))
}

// BufferMessageWithImage stores a message with an image. Thread-safe.
func (c *ChannelChat) BufferMessageWithImage(text string, imageData []byte) {
	c.mu.Lock()
	c.messages = append(c.messages, BufferedMessage{Text: text, ImageData: imageData})
	c.mu.Unlock()
	log.Printf("[agent-chat] buffered message with image (%d bytes), total buffered: %d", len(imageData), len(c.messages))
}

// BufferMessageWithAudio stores a message with audio. Thread-safe.
func (c *ChannelChat) BufferMessageWithAudio(text string, audioData []byte, audioMediaType string) {
	c.mu.Lock()
	c.messages = append(c.messages, BufferedMessage{Text: text, AudioData: audioData, AudioMediaType: audioMediaType})
	c.mu.Unlock()
	log.Printf("[agent-chat] buffered message with audio (%d bytes, mime=%s), total buffered: %d", len(audioData), audioMediaType, len(c.messages))
}

// DrainMessages returns and clears all buffered messages. Thread-safe.
func (c *ChannelChat) DrainMessages() []BufferedMessage {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.messages) == 0 {
		return nil
	}
	msgs := c.messages
	c.messages = nil
	log.Printf("[agent-chat] drained %d buffered message(s)", len(msgs))
	return msgs
}
