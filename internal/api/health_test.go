package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestServer(t *testing.T, ready ReadyFunc) *Server {
	t.Helper()
	return NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Ready:         ready,
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
}

func TestHealthz_AlwaysOK(t *testing.T) {
	srv := newTestServer(t, func(ctx context.Context) error {
		return errors.New("db down, should be irrelevant for healthz")
	})

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
	var body map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	assert.Equal(t, "ok", body["status"])
}

func TestReadyz_OKWhenReadyFuncReturnsNil(t *testing.T) {
	srv := newTestServer(t, func(ctx context.Context) error { return nil })

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
}

func TestReadyz_OKWhenReadyFuncIsNil(t *testing.T) {
	srv := newTestServer(t, nil)

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
}

func TestReadyz_ServiceUnavailableOnError(t *testing.T) {
	srv := newTestServer(t, func(ctx context.Context) error {
		return errors.New("database not reachable")
	})

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusServiceUnavailable, rr.Code)
	var body map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	assert.Equal(t, "not_ready", body["status"])
	assert.Contains(t, body["error"], "database not reachable")
}

// oversizedBody is a JSON object larger than the 10-byte cap the body
// limit tests configure.
var oversizedBody = `{"padding": "` + strings.Repeat("x", 100) + `"}`

func registerEchoRoute(r chi.Router) {
	r.Post("/echo", func(w http.ResponseWriter, r *http.Request) {
		var v map[string]any
		if decodeJSON(w, r, &v) {
			w.WriteHeader(http.StatusOK)
		}
	})
}

func TestBodyLimit_RejectsOversizedBodiesWith413(t *testing.T) {
	tests := []struct {
		name     string
		register func(r chi.Router)
		path     string
		body     string
	}{
		{
			name:     "control plane decodeJSON",
			register: registerEchoRoute,
			path:     "/echo",
			body:     oversizedBody,
		},
		{
			name:     "control plane decodeJSON with trailing bytes past the cap",
			register: registerEchoRoute,
			path:     "/echo",
			body:     `{}` + strings.Repeat(" ", 100),
		},
		{
			name:     "chat completions proxy",
			register: func(r chi.Router) { RegisterProxyRoutes(r, ProxyDeps{}) },
			path:     "/v1/chat/completions",
			body:     oversizedBody,
		},
		{
			name:     "tool invocation",
			register: func(r chi.Router) { RegisterToolRoutes(r, ToolDeps{}) },
			path:     "/v1/tools/predict",
			body:     oversizedBody,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := NewServer(ServerConfig{
				ListenAddress:   "127.0.0.1:0",
				MaxRequestBytes: 10,
				Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
			})
			tt.register(srv.Router())

			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			// The tool route checks the stage before it reads the body.
			req.Header.Set(HeaderStage, "test-stage")
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)

			assert.Equal(t, http.StatusRequestEntityTooLarge, rr.Code)
			var env errorEnvelope
			require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &env))
			assert.Equal(t, "request body exceeds the 10 byte limit", env.Error.Message)
			assert.Equal(t, "invalid_request_error", env.Error.Type)
		})
	}
}

func TestDecodeJSON_StatusUnderTheCap(t *testing.T) {
	tests := []struct {
		name string
		body string
		want int
	}{
		{name: "single object", body: `{"a": 1}`, want: http.StatusOK},
		{name: "trailing whitespace", body: "{\"a\": 1}\n  ", want: http.StatusOK},
		{name: "malformed JSON", body: `{not json`, want: http.StatusBadRequest},
		{name: "second JSON value", body: `{}{}`, want: http.StatusBadRequest},
		{name: "trailing junk", body: `{} junk`, want: http.StatusBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := NewServer(ServerConfig{
				ListenAddress:   "127.0.0.1:0",
				MaxRequestBytes: 1024,
				Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
			})
			registerEchoRoute(srv.Router())

			req := httptest.NewRequest(http.MethodPost, "/echo", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)

			assert.Equal(t, tt.want, rr.Code)
		})
	}
}
