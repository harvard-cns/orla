package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

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

func TestBodyLimit_RejectsOversizedBodies(t *testing.T) {
	srv := NewServer(ServerConfig{
		ListenAddress:   "127.0.0.1:0",
		MaxRequestBytes: 10,
		Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	srv.Router().Post("/echo", func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.ReadAll(r.Body); err != nil {
			http.Error(w, "too big", http.StatusRequestEntityTooLarge)
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/echo", bytes.NewReader(bytes.Repeat([]byte("x"), 100)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusRequestEntityTooLarge, rr.Code)
}
