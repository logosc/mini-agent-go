package agent

import (
	"context"
	"testing"
	"time"
)

func TestNullChatSendDoesNotBlock(t *testing.T) {
	chat := NullChat{}
	if err := chat.Send(context.Background(), "hello"); err != nil {
		t.Fatalf("Send: %v", err)
	}
}

func TestNullChatSendAttachmentDoesNotBlock(t *testing.T) {
	chat := NullChat{}
	if err := chat.SendAttachment(context.Background(), Attachment{Type: "file", Data: []byte("x")}); err != nil {
		t.Fatalf("SendAttachment: %v", err)
	}
}

func TestNullChatWaitForReplyReturnsError(t *testing.T) {
	chat := NullChat{}
	_, err := chat.WaitForReply(context.Background())
	if err == nil {
		t.Fatal("WaitForReply should return error in non-interactive mode")
	}
}

func TestNullChatDrainMessages(t *testing.T) {
	chat := NullChat{}
	if got := chat.DrainMessages(); got != nil {
		t.Errorf("NullChat.DrainMessages() = %v, want nil", got)
	}
}

func TestChannelChatSendAndReply(t *testing.T) {
	ch := make(chan Reply, 1)
	var sent []string
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error {
			sent = append(sent, text)
			return nil
		},
		ReplyCh: ch,
	}

	if err := chat.Send(context.Background(), "What style?"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if len(sent) != 1 || sent[0] != "What style?" {
		t.Errorf("sent = %v", sent)
	}

	ch <- Reply{Text: "watercolor"}
	reply, err := chat.WaitForReply(context.Background())
	if err != nil {
		t.Fatalf("WaitForReply: %v", err)
	}
	if reply.Text != "watercolor" {
		t.Errorf("reply = %q, want %q", reply.Text, "watercolor")
	}
}

func TestChannelChatContextCancellation(t *testing.T) {
	chat := &ChannelChat{
		SendFunc: func(ctx context.Context, text string) error { return nil },
		ReplyCh:  make(chan Reply),
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := chat.WaitForReply(ctx)
	if err == nil {
		t.Fatal("should return error on cancelled context")
	}
}

func TestChannelChatBufferMessage(t *testing.T) {
	chat := &ChannelChat{ReplyCh: make(chan Reply)}

	chat.BufferMessage("hello")
	chat.BufferMessage("world")

	msgs := chat.DrainMessages()
	if len(msgs) != 2 {
		t.Fatalf("DrainMessages() returned %d, want 2", len(msgs))
	}
	if msgs[0].Text != "hello" {
		t.Errorf("msgs[0].Text = %q", msgs[0].Text)
	}
	if msgs[1].Text != "world" {
		t.Errorf("msgs[1].Text = %q", msgs[1].Text)
	}

	// Second drain should be empty.
	if got := chat.DrainMessages(); len(got) != 0 {
		t.Errorf("second DrainMessages() returned %d, want 0", len(got))
	}
}

func TestChannelChatDrainMessagesEmpty(t *testing.T) {
	chat := &ChannelChat{ReplyCh: make(chan Reply)}
	if got := chat.DrainMessages(); got != nil {
		t.Errorf("DrainMessages() on empty = %v, want nil", got)
	}
}

func TestWaitForReplyAggregation(t *testing.T) {
	ch := make(chan Reply, 5)
	chat := &ChannelChat{
		ReplyCh:              ch,
		ReplyAggregateWindow: 2 * time.Second,
	}

	// Send 3 messages in rapid succession.
	ch <- Reply{Text: "hello"}
	ch <- Reply{Text: "world"}
	ch <- Reply{Text: "!", ImageData: []byte("img")}

	reply, err := chat.WaitForReply(context.Background())
	if err != nil {
		t.Fatalf("WaitForReply: %v", err)
	}
	if reply.Text != "hello\nworld\n!" {
		t.Errorf("reply.Text = %q, want %q", reply.Text, "hello\nworld\n!")
	}
	if string(reply.ImageData) != "img" {
		t.Errorf("reply.ImageData = %q, want %q", reply.ImageData, "img")
	}
}

func TestWaitForReplyNoWindow(t *testing.T) {
	ch := make(chan Reply, 5)
	chat := &ChannelChat{
		ReplyCh: ch,
		// ReplyAggregateWindow defaults to 0 — no aggregation.
	}

	ch <- Reply{Text: "first"}
	ch <- Reply{Text: "second"}

	reply, err := chat.WaitForReply(context.Background())
	if err != nil {
		t.Fatalf("WaitForReply: %v", err)
	}
	// Should return only the first message.
	if reply.Text != "first" {
		t.Errorf("reply.Text = %q, want %q", reply.Text, "first")
	}
}
