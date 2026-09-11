package api

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/telemetry"
)

type fakeFeedbackSink struct {
	mu       sync.Mutex
	received []*telemetry.Feedback
}

func (f *fakeFeedbackSink) Submit(fb *telemetry.Feedback) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.received = append(f.received, fb)
	return true
}

func (f *fakeFeedbackSink) Received() []*telemetry.Feedback {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]*telemetry.Feedback, len(f.received))
	copy(out, f.received)
	return out
}

func newFeedbackTestServer(t *testing.T) (*Server, *fakeFeedbackSink) {
	t.Helper()
	sink := &fakeFeedbackSink{}
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterFeedbackRoutes(srv.Router(), FeedbackDeps{Sink: sink})
	return srv, sink
}

func TestFeedback_AcceptsValidPayload(t *testing.T) {
	srv, sink := newFeedbackTestServer(t)

	body, _ := json.Marshal(map[string]any{
		"completion_id": "chatcmpl-abc",
		"stage_id":      "planning",
		"rating":        0.8,
		"labels":        []string{"accurate"},
		"notes":         "good answer",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusAccepted, rr.Code, rr.Body.String())
	got := sink.Received()
	require.Len(t, got, 1)
	assert.Equal(t, "chatcmpl-abc", got[0].CompletionID)
	assert.Equal(t, "planning", got[0].StageID)
	require.NotNil(t, got[0].Rating)
	assert.InDelta(t, 0.8, *got[0].Rating, 1e-9)
	assert.Equal(t, []string{"accurate"}, got[0].Labels)
}

func TestFeedback_RejectsMissingCompletionID(t *testing.T) {
	srv, _ := newFeedbackTestServer(t)
	body, _ := json.Marshal(map[string]any{
		"stage_id": "planning",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestFeedback_RejectsMissingStageID(t *testing.T) {
	srv, _ := newFeedbackTestServer(t)
	body, _ := json.Marshal(map[string]any{
		"completion_id": "chatcmpl-abc",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestFeedback_RejectsOutOfRangeRating(t *testing.T) {
	srv, _ := newFeedbackTestServer(t)
	for _, bad := range []float64{-0.1, 1.5} {
		body, _ := json.Marshal(map[string]any{
			"completion_id": "x",
			"stage_id":      "s",
			"rating":        bad,
		})
		req := httptest.NewRequest(http.MethodPost, "/v1/feedback", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		srv.Router().ServeHTTP(rr, req)
		assert.Equal(t, http.StatusBadRequest, rr.Code, "rating=%v should be rejected", bad)
	}
}

func TestFeedback_OmittedRatingIsNil(t *testing.T) {
	srv, sink := newFeedbackTestServer(t)
	body, _ := json.Marshal(map[string]any{
		"completion_id": "chatcmpl-x",
		"stage_id":      "s",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusAccepted, rr.Code)
	require.Len(t, sink.Received(), 1)
	assert.Nil(t, sink.Received()[0].Rating)
}
