package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/backends"
)

func TestBackendHandlers_CreateStoresCostSource(t *testing.T) {
	srv, reg := newBackendTestServer(t)

	body := mustJSON(t, map[string]any{
		"name":            "gpt4o",
		"endpoint":        "https://api.openai.com/v1",
		"model_id":        "openai:gpt-4o",
		"max_concurrency": 1,
		"cost_source":     "http://localhost:9090/price",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusCreated, rr.Code, rr.Body.String())

	stored, err := reg.Get(context.Background(), "gpt4o")
	require.NoError(t, err)
	require.NotNil(t, stored.CostSource)
	assert.Equal(t, "http://localhost:9090/price", *stored.CostSource)
}

// The URL rules themselves are covered in TestValidateHTTPURL. This
// checks that create rejects a bad value rather than storing it.
func TestBackendHandlers_CreateRejectsInvalidCostSource(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "b", "endpoint": "x", "model_id": "openai:gpt-4o",
		"max_concurrency": 1, "cost_source": "ftp://example.com/price",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code, rr.Body.String())
}

func TestBackendHandlers_CreateRejectsCostSourceOnTool(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "boltz", "endpoint": "x", "kind": "tool", "tool_kind": "structure-prediction",
		"max_concurrency": 1, "cost_source": "http://localhost:9090/price",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code, rr.Body.String())
}

func TestBackendHandlers_PatchSetsAndClearsCostSource(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	model := "openai:gpt-4o"
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "gpt4o", Endpoint: "x", Kind: backends.KindLLM, ModelID: &model, MaxConcurrency: 1,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"cost_source": "http://localhost:9090/price"})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/gpt4o", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())

	var got backends.Backend
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	require.NotNil(t, got.CostSource)
	assert.Equal(t, "http://localhost:9090/price", *got.CostSource)

	body = mustJSON(t, map[string]any{"cost_source": ""})
	req = httptest.NewRequest(http.MethodPatch, "/api/v1/backends/gpt4o", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr = httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())

	stored, err := reg.Get(context.Background(), "gpt4o")
	require.NoError(t, err)
	assert.Nil(t, stored.CostSource, "empty cost_source must clear the field")
}

func TestBackendHandlers_PatchRejectsCostSourceOnTool(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	toolKind := "structure-prediction"
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "boltz", Endpoint: "x", Kind: backends.KindTool, ToolKind: &toolKind, MaxConcurrency: 1,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"cost_source": "http://localhost:9090/price"})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/boltz", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code, rr.Body.String())
}
