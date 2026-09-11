package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/stages"
)

func newStageTestServer(t *testing.T) (*Server, *stages.FakeRegistry) {
	t.Helper()
	reg := stages.NewFakeRegistry()
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterStageRoutes(srv.Router(), reg)
	return srv, reg
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return b
}

func TestStageHandlers_PutCreatesStage(t *testing.T) {
	srv, reg := newStageTestServer(t)

	body := mustJSON(t, map[string]any{
		"backend":          "gpt-4o",
		"reasoning_effort": "high",
		"labels":           map[string]any{"owner": "core"},
	})
	req := httptest.NewRequest(http.MethodPut, "/api/v1/stages/planning", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)

	var got stages.Stage
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, "planning", got.ID)
	assert.Equal(t, "gpt-4o", got.Backend)
	assert.Equal(t, "high", got.ReasoningEffort)
	assert.Equal(t, "core", got.Labels["owner"])

	stored, err := reg.Get(context.Background(), "planning")
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", stored.Backend)
}

func TestStageHandlers_GetMissingReturns404(t *testing.T) {
	srv, _ := newStageTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/stages/missing", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusNotFound, rr.Code)
	var env errorEnvelope
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &env))
	assert.Equal(t, "not_found", env.Error.Type)
}

func TestStageHandlers_PatchPartialUpdate(t *testing.T) {
	srv, reg := newStageTestServer(t)

	_, err := reg.Replace(context.Background(), &stages.Stage{
		ID:              "planning",
		Backend:         "gpt-4o",
		ReasoningEffort: "high",
		Labels:          map[string]any{"a": "b"},
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"backend": "gpt-4o-mini"})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/stages/planning", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
	var got stages.Stage
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, "gpt-4o-mini", got.Backend)
	assert.Equal(t, "high", got.ReasoningEffort, "untouched field preserved")
	assert.Equal(t, "b", got.Labels["a"])
}

func TestStageHandlers_PatchMissingReturns404(t *testing.T) {
	srv, _ := newStageTestServer(t)

	body := mustJSON(t, map[string]any{"backend": "gpt-4o"})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/stages/missing", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusNotFound, rr.Code)
}

func TestStageHandlers_DeleteReturns204(t *testing.T) {
	srv, reg := newStageTestServer(t)

	_, err := reg.Replace(context.Background(), &stages.Stage{ID: "to-delete"})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodDelete, "/api/v1/stages/to-delete", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusNoContent, rr.Code)
	assert.Empty(t, rr.Body.Bytes())
}

func TestStageHandlers_DeleteMissingReturns404(t *testing.T) {
	srv, _ := newStageTestServer(t)
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/stages/missing", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNotFound, rr.Code)
}

func TestStageHandlers_ListReturnsAllStagesOrderedByID(t *testing.T) {
	srv, reg := newStageTestServer(t)

	for _, id := range []string{"zeta", "alpha", "mu"} {
		_, err := reg.Replace(context.Background(), &stages.Stage{ID: id})
		require.NoError(t, err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/stages", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
	var body struct {
		Stages []stages.Stage `json:"stages"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	require.Len(t, body.Stages, 3)
	assert.Equal(t, []string{"alpha", "mu", "zeta"},
		[]string{body.Stages[0].ID, body.Stages[1].ID, body.Stages[2].ID})
}

func TestStageHandlers_PutRejectsInvalidJSON(t *testing.T) {
	srv, _ := newStageTestServer(t)
	req := httptest.NewRequest(http.MethodPut, "/api/v1/stages/planning",
		bytes.NewReader([]byte("not json")))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestStageHandlers_PutRejectsUnknownFields(t *testing.T) {
	srv, _ := newStageTestServer(t)
	body := mustJSON(t, map[string]any{
		"backend":      "x",
		"unknown_attr": "y",
	})
	req := httptest.NewRequest(http.MethodPut, "/api/v1/stages/planning", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code,
		"DisallowUnknownFields should reject typos")
}

func TestStageHandlers_PutSetsPrompt(t *testing.T) {
	srv, reg := newStageTestServer(t)

	body := mustJSON(t, map[string]any{"backend": "gpt-4o", "prompt": "You are a careful assistant."})
	req := httptest.NewRequest(http.MethodPut, "/api/v1/stages/answer", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
	var got stages.Stage
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, "You are a careful assistant.", got.Prompt)

	stored, err := reg.Get(context.Background(), "answer")
	require.NoError(t, err)
	assert.Equal(t, "You are a careful assistant.", stored.Prompt)
}

func TestStageHandlers_PatchSetsPromptLeavingBackend(t *testing.T) {
	srv, reg := newStageTestServer(t)
	_, err := reg.Replace(context.Background(), &stages.Stage{ID: "answer", Backend: "gpt-4o"})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"prompt": "new prompt"})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/stages/answer", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
	var got stages.Stage
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, "new prompt", got.Prompt)
	assert.Equal(t, "gpt-4o", got.Backend, "untouched field preserved")
}

func TestStageHandlers_PatchClearsPrompt(t *testing.T) {
	srv, reg := newStageTestServer(t)
	_, err := reg.Replace(context.Background(), &stages.Stage{ID: "answer", Backend: "gpt-4o", Prompt: "old"})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"prompt": ""})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/stages/answer", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
	var got stages.Stage
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Empty(t, got.Prompt)
}

func TestStageHandlers_PutRejectsOversizePrompt(t *testing.T) {
	srv, _ := newStageTestServer(t)

	body := mustJSON(t, map[string]any{
		"backend": "gpt-4o",
		"prompt":  strings.Repeat("x", maxPromptLen+1),
	})
	req := httptest.NewRequest(http.MethodPut, "/api/v1/stages/answer", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestStageHandlers_PatchRejectsOversizePrompt(t *testing.T) {
	srv, reg := newStageTestServer(t)
	_, err := reg.Replace(context.Background(), &stages.Stage{ID: "answer", Backend: "gpt-4o"})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"prompt": strings.Repeat("x", maxPromptLen+1)})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/stages/answer", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadRequest, rr.Code)
}
