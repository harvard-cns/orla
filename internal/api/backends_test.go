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
)

func newBackendTestServer(t *testing.T) (*Server, *backends.FakeRegistry) {
	t.Helper()
	reg := backends.NewFakeRegistry()
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterBackendRoutes(srv.Router(), BackendDeps{Registry: reg})
	return srv, reg
}

func TestBackendHandlers_CreateReturns201(t *testing.T) {
	srv, reg := newBackendTestServer(t)

	body := mustJSON(t, map[string]any{
		"name":            "gpt4o",
		"endpoint":        "https://api.openai.com/v1",
		"model_id":        "openai:gpt-4o",
		"api_key_env_var": "OPENAI_API_KEY",
		"max_concurrency": 8,
		"quality":         0.85,
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusCreated, rr.Code, rr.Body.String())
	var got backends.Backend
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, "gpt4o", got.Name)
	assert.Equal(t, int32(8), got.MaxConcurrency)
	require.NotNil(t, got.Quality)
	assert.InDelta(t, 0.85, *got.Quality, 1e-9)

	stored, err := reg.Get(context.Background(), "gpt4o")
	require.NoError(t, err)
	require.NotNil(t, stored.ModelID)
	assert.Equal(t, "openai:gpt-4o", *stored.ModelID)
}

func TestBackendHandlers_CreateDuplicateReturns409(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "gpt4o", "endpoint": "x", "model_id": "openai:gpt-4o", "max_concurrency": 1,
	})
	for i := range 2 {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		srv.Router().ServeHTTP(rr, req)
		if i == 0 {
			require.Equal(t, http.StatusCreated, rr.Code)
		} else {
			assert.Equal(t, http.StatusConflict, rr.Code)
		}
	}
}

func TestBackendHandlers_CreateRejectsMissingFields(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	tests := []struct {
		name string
		body map[string]any
	}{
		{"no name", map[string]any{"endpoint": "x", "model_id": "y", "max_concurrency": 1}},
		{"no endpoint", map[string]any{"name": "x", "model_id": "y", "max_concurrency": 1}},
		{"no model_id", map[string]any{"name": "x", "endpoint": "y", "max_concurrency": 1}},
		{"zero concurrency", map[string]any{"name": "x", "endpoint": "y", "model_id": "z", "max_concurrency": 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := mustJSON(t, tt.body)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)
			assert.Equal(t, http.StatusBadRequest, rr.Code)
		})
	}
}

// TestBackendHandlers_RejectsNonJSONContentType patches a backend's
// endpoint with Content-Type text/plain and asserts both a 415
// response and that the endpoint is left unchanged.
func TestBackendHandlers_RejectsNonJSONContentType(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "x", Endpoint: "trusted.example", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"endpoint": "http://attacker.example"})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/x", bytes.NewReader(body))
	req.Header.Set("Content-Type", "text/plain")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusUnsupportedMediaType, rr.Code)
	stored, err := reg.Get(context.Background(), "x")
	require.NoError(t, err)
	assert.Equal(t, "trusted.example", stored.Endpoint, "the forged write must never reach the handler")
}

func TestBackendHandlers_GetMissing404(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends/missing", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNotFound, rr.Code)
}

func TestBackendHandlers_PatchOnlyChangesProvidedFields(t *testing.T) {
	srv, reg := newBackendTestServer(t)

	q := 0.85
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "gpt4o", Endpoint: "x", ModelID: new("openai:gpt-4o"),
		MaxConcurrency: 4, Quality: &q,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"max_concurrency": 16})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/gpt4o", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	var got backends.Backend
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, int32(16), got.MaxConcurrency)
	require.NotNil(t, got.Quality)
	assert.InDelta(t, 0.85, *got.Quality, 1e-9, "untouched")
}

func TestBackendHandlers_PatchZeroConcurrencyReturns400(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "x", Endpoint: "y", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"max_concurrency": 0})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/x", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestBackendHandlers_DeleteReturns204(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "x", Endpoint: "y", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodDelete, "/api/v1/backends/x", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNoContent, rr.Code)
}

func TestBackendHandlers_CreateRejectsRatesOnLLM(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "gpt4o", "endpoint": "https://api.openai.com/v1",
		"model_id": "openai:gpt-4o", "max_concurrency": 1,
		"rates": map[string]float64{"gpu_seconds": 0.001},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
	assert.Contains(t, rr.Body.String(), "rates is only valid for kind=tool")
}

func TestBackendHandlers_CreateRejectsNegativeRate(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "boltz", "endpoint": "http://boltz.local", "kind": "tool",
		"tool_kind": "structure-prediction", "max_concurrency": 1,
		"rates": map[string]float64{"gpu_seconds": -1.0},
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
	assert.Contains(t, rr.Body.String(), "non-negative finite")
}

func TestBackendHandlers_PatchRatesUpdatesFakeRegistry(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	toolKind := "structure-prediction"
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "boltz", Endpoint: "http://boltz.local", MaxConcurrency: 1,
		Kind: backends.KindTool, ToolKind: &toolKind,
		Rates: map[string]float64{"gpu_seconds": 0.001},
	})
	require.NoError(t, err)

	newRates := map[string]float64{"gpu_seconds": 0.002}
	body := mustJSON(t, map[string]any{"rates": newRates})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/boltz", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())

	got, err := reg.Get(context.Background(), "boltz")
	require.NoError(t, err)
	assert.InDelta(t, 0.002, got.Rates["gpu_seconds"], 1e-12)
}

func TestFakeRegistry_GetReturnsIndependentRatesCopy(t *testing.T) {
	reg := backends.NewFakeRegistry()
	toolKind := "structure-prediction"
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "boltz", Endpoint: "x", MaxConcurrency: 1,
		Kind: backends.KindTool, ToolKind: &toolKind,
		Rates: map[string]float64{"gpu_seconds": 0.001},
	})
	require.NoError(t, err)
	first, err := reg.Get(context.Background(), "boltz")
	require.NoError(t, err)
	first.Rates["gpu_seconds"] = 999.0 // poison the returned map

	second, err := reg.Get(context.Background(), "boltz")
	require.NoError(t, err)
	assert.InDelta(t, 0.001, second.Rates["gpu_seconds"], 1e-12,
		"a caller's mutation must not leak into the registry's stored map")
}

type fakeManager struct{ states map[string]string }

func (f *fakeManager) CircuitState(name string) string {
	if s, ok := f.states[name]; ok {
		return s
	}
	return "closed"
}

func newBackendTestServerWithManager(t *testing.T, m LLMBackendManager) (*Server, *backends.FakeRegistry) {
	t.Helper()
	reg := backends.NewFakeRegistry()
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterBackendRoutes(srv.Router(), BackendDeps{Registry: reg, Manager: m})
	return srv, reg
}

func TestBackendHandlers_ListCircuitStateClosed(t *testing.T) {
	srv, reg := newBackendTestServerWithManager(t, &fakeManager{states: map[string]string{"b": "closed"}})
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "b", Endpoint: "x", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code)

	var body struct {
		Backends []struct {
			CircuitBreaker string `json:"circuit_breaker"`
		} `json:"backends"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	require.Len(t, body.Backends, 1)
	assert.Equal(t, "closed", body.Backends[0].CircuitBreaker)
}

func TestBackendHandlers_ListCircuitStateOpen(t *testing.T) {
	srv, reg := newBackendTestServerWithManager(t, &fakeManager{states: map[string]string{"b": "open"}})
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "b", Endpoint: "x", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code)

	var body struct {
		Backends []struct {
			CircuitBreaker string `json:"circuit_breaker"`
		} `json:"backends"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	require.Len(t, body.Backends, 1)
	assert.Equal(t, "open", body.Backends[0].CircuitBreaker)
}

func TestBackendHandlers_GetCircuitStateHalfOpen(t *testing.T) {
	srv, reg := newBackendTestServerWithManager(t, &fakeManager{states: map[string]string{"b": "half-open"}})
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "b", Endpoint: "x", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends/b", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code)

	var body struct {
		CircuitBreaker string `json:"circuit_breaker"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	assert.Equal(t, "half-open", body.CircuitBreaker)
}

func TestBackendHandlers_ListDefaultsCircuitStateToClosedWithoutManager(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "b", Endpoint: "x", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code)

	var body struct {
		Backends []struct {
			CircuitBreaker string `json:"circuit_breaker"`
		} `json:"backends"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	require.Len(t, body.Backends, 1)
	assert.Equal(t, "closed", body.Backends[0].CircuitBreaker)
}

func TestBackendHandlers_ListOrderedByName(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	for _, n := range []string{"zeta", "alpha", "mu"} {
		_, err := reg.Insert(context.Background(), &backends.Backend{
			Name: n, Endpoint: "y", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
		})
		require.NoError(t, err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code)

	var body struct {
		Backends []backends.Backend `json:"backends"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	require.Len(t, body.Backends, 3)
	assert.Equal(t, []string{"alpha", "mu", "zeta"},
		[]string{body.Backends[0].Name, body.Backends[1].Name, body.Backends[2].Name})
}

func TestBackendHandlers_CreateRejectsCacheReadCostOnTool(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "boltz", "endpoint": "http://boltz.local", "kind": "tool",
		"tool_kind": "structure-prediction", "max_concurrency": 1,
		"rates":                      map[string]float64{"gpu_seconds": 0.001},
		"cache_read_cost_per_mtoken": 0.30,
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
	assert.Contains(t, rr.Body.String(), "cache_read_cost_per_mtoken is only valid for kind=llm")
}

func TestBackendHandlers_CreateRejectsNegativeCacheReadCost(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "gpt4o", "endpoint": "https://api.openai.com/v1",
		"model_id": "openai:gpt-4o", "max_concurrency": 1,
		"cache_read_cost_per_mtoken": -0.30,
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
	assert.Contains(t, rr.Body.String(), "cache_read_cost_per_mtoken must be a non-negative finite number")
}

func TestBackendHandlers_PatchCacheReadCost(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "gpt4o", Endpoint: "https://api.openai.com/v1", MaxConcurrency: 1,
		Kind: backends.KindLLM, ModelID: new("openai:gpt-4o"),
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"cache_read_cost_per_mtoken": 0.30})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/gpt4o", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())

	got, err := reg.Get(context.Background(), "gpt4o")
	require.NoError(t, err)
	require.NotNil(t, got.CacheReadCostPerMtoken)
	assert.InDelta(t, 0.30, *got.CacheReadCostPerMtoken, 1e-9)
}

func TestBackendHandlers_PatchRejectsCacheReadCostOnTool(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	toolKind := "structure-prediction"
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "boltz", Endpoint: "http://boltz.local", MaxConcurrency: 1,
		Kind: backends.KindTool, ToolKind: &toolKind,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"cache_read_cost_per_mtoken": 0.30})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/boltz", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
	assert.Contains(t, rr.Body.String(), "cache_read_cost_per_mtoken is only valid for kind=llm")
}
