package agent

import (
	"context"
	"fmt"
	"log"
	"sync"
)

// Reply holds the user's response.
type Reply struct {
	Text      string
	ImageData []byte // Optional image data.
}

// BufferedMessage holds a message that arrived while the agent was busy.
type BufferedMessage struct {
	Text      string
	ImageData []byte
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
	select {
	case msg := <-c.ReplyCh:
		return msg, nil
	case <-ctx.Done():
		return Reply{}, ctx.Err()
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
