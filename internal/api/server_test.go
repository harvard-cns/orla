package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// freePort returns an OS-assigned free TCP port as "127.0.0.1:N".
func freePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := l.Addr().String()
	require.NoError(t, l.Close())
	return addr
}

func TestServer_StartShutdown(t *testing.T) {
	addr := freePort(t)
	srv := NewServer(ServerConfig{
		ListenAddress: addr,
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		ReadTimeout:   1 * time.Second,
		WriteTimeout:  1 * time.Second,
	})

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Start() }()

	// Poll until the listener accepts.
	deadline := time.Now().Add(2 * time.Second)
	var resp *http.Response
	var err error
	for time.Now().Before(deadline) {
		resp, err = http.Get("http://" + addr + "/healthz")
		if err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	require.NoError(t, err, "server never became reachable")
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.NoError(t, resp.Body.Close())

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	require.NoError(t, srv.Shutdown(shutdownCtx))

	select {
	case err := <-errCh:
		assert.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("Start did not return after Shutdown")
	}
}

func TestServer_RequestIDPresent(t *testing.T) {
	addr := freePort(t)
	srv := NewServer(ServerConfig{
		ListenAddress: addr,
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Start() }()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
		<-errCh
	})

	deadline := time.Now().Add(2 * time.Second)
	var resp *http.Response
	var err error
	for time.Now().Before(deadline) {
		resp, err = http.Get("http://" + addr + "/healthz")
		if err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	assert.NotEmpty(t, resp.Header.Get("X-Request-Id"),
		"chi middleware should set X-Request-Id")
}

// TestServer_AccessLogRecordsCaller covers the client address on the
// access log, which is the forensic half of the control-plane audit.
// Orla authenticates nobody, so both fields are whatever the caller
// claimed.
func TestServer_AccessLogRecordsCaller(t *testing.T) {
	var buf bytes.Buffer
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(&buf, nil)),
	})
	srv.Router().Put("/api/v1/stages/{id}", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPut, "/api/v1/stages/planning", nil)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Forwarded-For", "10.1.2.3")
	srv.Router().ServeHTTP(httptest.NewRecorder(), req)

	assert.Contains(t, buf.String(), "remote_addr="+req.RemoteAddr)
	assert.Contains(t, buf.String(), "forwarded_for=10.1.2.3")
}

func TestRequireJSONMiddleware_EnforcesContentTypeOnWrites(t *testing.T) {
	tests := []struct {
		name        string
		method      string
		contentType string
		setHeader   bool
		wantStatus  int
	}{
		{name: "missing header", method: http.MethodPost, setHeader: false, wantStatus: http.StatusUnsupportedMediaType},
		{name: "text/plain", method: http.MethodPost, contentType: "text/plain", setHeader: true, wantStatus: http.StatusUnsupportedMediaType},
		{name: "form-urlencoded", method: http.MethodPost, contentType: "application/x-www-form-urlencoded", setHeader: true, wantStatus: http.StatusUnsupportedMediaType},
		{name: "multipart", method: http.MethodPost, contentType: "multipart/form-data; boundary=x", setHeader: true, wantStatus: http.StatusUnsupportedMediaType},
		{name: "json", method: http.MethodPost, contentType: "application/json", setHeader: true, wantStatus: http.StatusOK},
		{name: "json with charset", method: http.MethodPost, contentType: "application/json; charset=utf-8", setHeader: true, wantStatus: http.StatusOK},
		{name: "put missing header", method: http.MethodPut, setHeader: false, wantStatus: http.StatusUnsupportedMediaType},
		{name: "patch missing header", method: http.MethodPatch, setHeader: false, wantStatus: http.StatusUnsupportedMediaType},
		{name: "get ignores header", method: http.MethodGet, setHeader: false, wantStatus: http.StatusOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := NewServer(ServerConfig{
				ListenAddress: "127.0.0.1:0",
				Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
			})
			ok := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }
			srv.Router().Post("/echo", ok)
			srv.Router().Put("/echo", ok)
			srv.Router().Patch("/echo", ok)
			srv.Router().Get("/echo", ok)

			req := httptest.NewRequest(tt.method, "/echo", bytes.NewReader([]byte(`{}`)))
			if tt.setHeader {
				req.Header.Set("Content-Type", tt.contentType)
			}
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)

			assert.Equal(t, tt.wantStatus, rr.Code, rr.Body.String())
		})
	}
}

// TestRoutingPath_MirrorsChiPathSelection builds its requests by hand,
// since httptest.NewRequest cannot produce a URL whose RawPath and
// Path are both empty.
func TestRoutingPath_MirrorsChiPathSelection(t *testing.T) {
	tests := []struct {
		name    string
		rawPath string
		path    string
		want    string
	}{
		{name: "raw path set", rawPath: "/api/v1/stages/foo%2Fbar", path: "/api/v1/stages/foo/bar", want: "/api/v1/stages/foo%2Fbar"},
		{name: "raw path empty, path set", rawPath: "", path: "/api/v1/stages/planning", want: "/api/v1/stages/planning"},
		{name: "both empty", rawPath: "", path: "", want: "/"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &http.Request{URL: &url.URL{RawPath: tt.rawPath, Path: tt.path}}
			assert.Equal(t, tt.want, routingPath(r))
		})
	}
}

// TestRequireJSONMiddleware_NamesTheDeclaredType covers the rejection
// body, which quotes the caller's own header back. The encoder escapes
// it, so a header carrying quotes stays inside the JSON envelope.
func TestRequireJSONMiddleware_NamesTheDeclaredType(t *testing.T) {
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	srv.Router().Post("/echo", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	for _, declared := range []string{"text/plain", `a"b`} {
		req := httptest.NewRequest(http.MethodPost, "/echo", bytes.NewReader([]byte(`{}`)))
		req.Header.Set("Content-Type", declared)
		rr := httptest.NewRecorder()
		srv.Router().ServeHTTP(rr, req)

		require.Equal(t, http.StatusUnsupportedMediaType, rr.Code)
		var body errorEnvelope
		require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
		assert.Contains(t, body.Error.Message, strconv.Quote(declared))
		assert.Equal(t, "invalid_request_error", body.Error.Type)
	}
}
