package agent

import "testing"

func TestToolResultWithError(t *testing.T) {
	r := ToolResult{}.WithError("something broke")
	if !r.IsError {
		t.Error("expected IsError=true")
	}
	if r.Summary != "something broke" {
		t.Errorf("summary = %q", r.Summary)
	}
}

func TestAttachmentFields(t *testing.T) {
	a := Attachment{Type: "file", Data: []byte("pdf"), Name: "story.pdf", Caption: "Your story"}
	if a.Type != "file" || a.Name != "story.pdf" {
		t.Errorf("attachment = %+v", a)
	}
}
