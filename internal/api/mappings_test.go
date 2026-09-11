package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/mappings"
	"github.com/harvard-cns/orla/internal/provider"
	"github.com/harvard-cns/orla/internal/scheduler"
	"github.com/harvard-cns/orla/internal/stages"

	"github.com/openai/openai-go"
)

func newMappingTestServer(t *testing.T) (*Server, *mappings.FakeRegistry) {
	t.Helper()
	reg := mappings.NewFakeRegistry()
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterMappingRoutes(srv.Router(), reg)
	return srv, reg
}

func TestMappingHandlers_PutReturns201AndStores(t *testing.T) {
	srv, reg := newMappingTestServer(t)

	body := mustJSON(t, map[string]any{
		"name":      "cand",
		"overrides": map[string]string{"queryBuilder": "gpt-4.1"},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/mappings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusCreated, rr.Code, rr.Body.String())
	var got mappings.Variant
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, "cand", got.Name)
	assert.Equal(t, map[string]string{"queryBuilder": "gpt-4.1"}, got.Overrides)

	stored, err := reg.Get(context.Background(), "cand")
	require.NoError(t, err)
	assert.Equal(t, "gpt-4.1", stored.Overrides["queryBuilder"])
}

func TestMappingHandlers_PutRejectsEmptyOverrides(t *testing.T) {
	srv, _ := newMappingTestServer(t)

	body := mustJSON(t, map[string]any{"name": "cand", "overrides": map[string]string{}})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/mappings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadRequest, rr.Code, rr.Body.String())
}

func TestMappingHandlers_PutRejectsMissingName(t *testing.T) {
	srv, _ := newMappingTestServer(t)

	body := mustJSON(t, map[string]any{"overrides": map[string]string{"queryBuilder": "gpt-4.1"}})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/mappings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadRequest, rr.Code, rr.Body.String())
}

func TestMappingHandlers_GetNotFound(t *testing.T) {
	srv, _ := newMappingTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/mappings/missing", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusNotFound, rr.Code)
}

func TestMappingHandlers_DeleteThenGone(t *testing.T) {
	srv, reg := newMappingTestServer(t)
	_, err := reg.Put(context.Background(), "cand", map[string]string{"queryBuilder": "gpt-4.1"})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodDelete, "/api/v1/mappings/cand", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNoContent, rr.Code)

	req = httptest.NewRequest(http.MethodDelete, "/api/v1/mappings/cand", nil)
	rr = httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNotFound, rr.Code)
}

// newVariantProxyEnv wires a proxy with two backends and a mappings
// registry so a request can be routed by the live mapping or a variant.
func newVariantProxyEnv(t *testing.T) (*Server, *stages.FakeRegistry, *mappings.FakeRegistry) {
	t.Helper()
	mock := func(name string) *provider.MockProvider {
		return provider.NewMockProvider().WithName(name).
			WithResponse(&openai.ChatCompletion{
				ID:      "chatcmpl-stub",
				Choices: []openai.ChatCompletionChoice{{FinishReason: "stop", Message: openai.ChatCompletionMessage{Role: "assistant", Content: "hi"}}},
			})
	}
	sched := scheduler.New(
		func(b *backends.Backend) provider.Backend { return mock(b.Name) },
		slog.New(slog.NewTextHandler(io.Discard, nil)),
	)
	sched.Register(&backends.Backend{Name: "live", Endpoint: "x", ModelID: new("openai:live"), MaxConcurrency: 2})
	sched.Register(&backends.Backend{Name: "shadow", Endpoint: "x", ModelID: new("openai:shadow"), MaxConcurrency: 2})
	t.Cleanup(func() { _ = sched.Shutdown(context.Background()) })

	stageReg := stages.NewFakeRegistry()
	_, err := stageReg.Replace(context.Background(), &stages.Stage{ID: "planning", Backend: "live"})
	require.NoError(t, err)

	mapReg := mappings.NewFakeRegistry()

	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterProxyRoutes(srv.Router(), ProxyDeps{Stages: stageReg, Mappings: mapReg, Scheduler: sched})
	return srv, stageReg, mapReg
}

func dispatchModel(t *testing.T, srv *Server, mapping string) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyForChat("q")))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "planning")
	if mapping != "" {
		req.Header.Set(HeaderMapping, mapping)
	}
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	var resp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &resp))
	return resp.Model
}

func TestProxy_NoMappingUsesLiveBackend(t *testing.T) {
	srv, _, _ := newVariantProxyEnv(t)
	assert.Equal(t, "live", dispatchModel(t, srv, ""))
}

func TestProxy_MappingVariantOverridesBackend(t *testing.T) {
	srv, _, mapReg := newVariantProxyEnv(t)
	_, err := mapReg.Put(context.Background(), "cand", map[string]string{"planning": "shadow"})
	require.NoError(t, err)
	assert.Equal(t, "shadow", dispatchModel(t, srv, "cand"))
}

func TestProxy_MappingVariantFallsThroughForUnmappedStage(t *testing.T) {
	srv, _, mapReg := newVariantProxyEnv(t)
	// The variant overrides a different stage, so "planning" falls
	// through to its live backend.
	_, err := mapReg.Put(context.Background(), "cand", map[string]string{"other": "shadow"})
	require.NoError(t, err)
	assert.Equal(t, "live", dispatchModel(t, srv, "cand"))
}
