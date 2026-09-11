package api

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeControlPlaneAuditMetrics struct {
	mu  sync.Mutex
	got []string // "resource|method|outcome"
}

func (f *fakeControlPlaneAuditMetrics) IncControlPlaneMutation(resource, method, outcome string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.got = append(f.got, resource+"|"+method+"|"+outcome)
}

func (f *fakeControlPlaneAuditMetrics) snapshot() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.got...)
}

// newAuditTestServer enables the audit through the real NewServer and
// registers one route per shape the control plane uses.
func newAuditTestServer(t *testing.T, logger *slog.Logger) (*Server, *fakeControlPlaneAuditMetrics) {
	t.Helper()
	m := &fakeControlPlaneAuditMetrics{}
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        logger,
		AuditMetrics:  m,
	})
	ok := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }
	r := srv.Router()
	r.Route("/api/v1/stages/{id}", func(r chi.Router) {
		r.Get("/", ok)
		r.Head("/", ok)
		r.Put("/", ok)
		r.Patch("/", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNotFound) })
		r.Delete("/", func(http.ResponseWriter, *http.Request) { panic("boom") })
		// net/http sends 200 for a handler that writes nothing.
		r.Post("/", func(http.ResponseWriter, *http.Request) {})
	})
	r.Post("/api/v1/scheduler/policy", ok)
	r.Post("/v1/chat/completions", ok)
	return srv, m
}

func TestControlPlaneResource_Outcomes(t *testing.T) {
	tests := []struct {
		name    string
		pattern string
		want    string
		wantOK  bool
	}{
		{name: "resource with id", pattern: "/api/v1/stages/{id}", want: "stages", wantOK: true},
		{name: "list, no id", pattern: "/api/v1/stages", want: "stages", wantOK: true},
		{name: "list, trailing slash", pattern: "/api/v1/stages/", want: "stages", wantOK: true},
		{name: "backends resource", pattern: "/api/v1/backends/{name}", want: "backends", wantOK: true},
		{name: "hyphenated resource", pattern: "/api/v1/stage-mapper/", want: "stage-mapper", wantOK: true},
		{name: "nested scheduler policy path", pattern: "/api/v1/scheduler/policy", want: "scheduler", wantOK: true},
		{name: "wildcard under a resource", pattern: "/api/v1/stages/{id}/*", want: "stages", wantOK: true},
		{name: "not a control-plane path", pattern: "/v1/chat/completions", want: "", wantOK: false},
		{name: "prefix only, no resource", pattern: "/api/v1/", want: "", wantOK: false},
		{name: "empty first segment", pattern: "/api/v1//policy", want: "", wantOK: false},
		{name: "unrouted request, no pattern", pattern: "", want: "", wantOK: false},
		{name: "root", pattern: "/", want: "", wantOK: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := controlPlaneResource(tt.pattern)
			assert.Equal(t, tt.want, got)
			assert.Equal(t, tt.wantOK, ok)
		})
	}
}

func TestAuditControlPlaneMutations_SkipsReads(t *testing.T) {
	srv, m := newAuditTestServer(t, slog.New(slog.NewTextHandler(io.Discard, nil)))

	for _, method := range []string{http.MethodGet, http.MethodHead} {
		rr := httptest.NewRecorder()
		srv.Router().ServeHTTP(rr, httptest.NewRequest(method, "/api/v1/stages/planning", nil))
		require.Equal(t, http.StatusOK, rr.Code)
	}

	assert.Empty(t, m.snapshot())
}

func TestAuditControlPlaneMutations_RecordsMutations(t *testing.T) {
	tests := []struct {
		name   string
		method string
		path   string
		want   string
	}{
		{name: "put stage", method: http.MethodPut, path: "/api/v1/stages/planning", want: "stages|PUT|success"},
		// Every orla handler writes a status. A handler that does not
		// counts as an error, since the recorded status stays at zero.
		{name: "handler writes no status", method: http.MethodPost, path: "/api/v1/stages/planning", want: "stages|POST|error"},
		{name: "post scheduler policy", method: http.MethodPost, path: "/api/v1/scheduler/policy", want: "scheduler|POST|success"},
		{name: "rejected patch", method: http.MethodPatch, path: "/api/v1/stages/planning", want: "stages|PATCH|error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, m := newAuditTestServer(t, slog.New(slog.NewTextHandler(io.Discard, nil)))

			req := httptest.NewRequest(tt.method, tt.path, nil)
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)

			assert.Equal(t, []string{tt.want}, m.snapshot())
		})
	}
}

// TestAuditControlPlaneMutations_RecordsPanicAsError covers a handler
// that panics mid-mutation. The audit runs outside Recoverer, so it
// records the attempt with the 500 the recoverer produced.
func TestAuditControlPlaneMutations_RecordsPanicAsError(t *testing.T) {
	var buf bytes.Buffer
	srv, m := newAuditTestServer(t, slog.New(slog.NewTextHandler(&buf, nil)))

	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, httptest.NewRequest(http.MethodDelete, "/api/v1/stages/planning", nil))

	assert.Equal(t, http.StatusInternalServerError, rr.Code)
	assert.Equal(t, []string{"stages|DELETE|error"}, m.snapshot())
	assert.Contains(t, buf.String(), "status=500")
}

func TestAuditControlPlaneMutations_IgnoresDataPlane(t *testing.T) {
	srv, m := newAuditTestServer(t, slog.New(slog.NewTextHandler(io.Discard, nil)))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)
	assert.Empty(t, m.snapshot())
}

// TestAuditControlPlaneMutations_IgnoresUnroutedPaths keeps the metric's
// label set bounded. The resource label comes from the URL, so counting
// requests that matched no route would let any caller mint an unbounded
// number of label series by POSTing junk paths.
func TestAuditControlPlaneMutations_IgnoresUnroutedPaths(t *testing.T) {
	srv, m := newAuditTestServer(t, slog.New(slog.NewTextHandler(io.Discard, nil)))

	for _, path := range []string{"/api/v1/junk", "/api/v1/other", "/api/v1/"} {
		req := httptest.NewRequest(http.MethodPost, path, nil)
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		srv.Router().ServeHTTP(rr, req)
		require.Equal(t, http.StatusNotFound, rr.Code, path)
	}

	assert.Empty(t, m.snapshot())
}

// TestAuditControlPlaneMutations_RecordsContentTypeRejections covers a
// write requireJSONMiddleware rejects before chi ever resolves a route
// pattern, using router.Find instead. A real control-plane resource is
// still counted, a junk path and a data-plane route are still not.
func TestAuditControlPlaneMutations_RecordsContentTypeRejections(t *testing.T) {
	tests := []struct {
		name   string
		method string
		path   string
		want   []string
	}{
		{name: "real control-plane resource", method: http.MethodPut, path: "/api/v1/stages/planning", want: []string{"stages|PUT|error"}},
		{name: "unrouted junk path", method: http.MethodPost, path: "/api/v1/junk", want: nil},
		{name: "data-plane route", method: http.MethodPost, path: "/v1/chat/completions", want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, m := newAuditTestServer(t, slog.New(slog.NewTextHandler(io.Discard, nil)))

			req := httptest.NewRequest(tt.method, tt.path, nil)
			req.Header.Set("Content-Type", "text/plain")
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)

			assert.Equal(t, http.StatusUnsupportedMediaType, rr.Code)
			assert.Equal(t, tt.want, m.snapshot())
		})
	}
}

// TestAuditControlPlaneMutations_RecordsContentTypeRejectionsOnEncodedPaths
// covers a percent-encoded path segment (e.g. a literal "%2F"), which
// decodes differently in URL.Path than in URL.RawPath. Real chi
// dispatch always prefers RawPath, so resolving the rejection's
// would-be pattern must walk the same string or it can miss a real
// resource, or worse, phantom-match one a request never actually
// reached. Each case also asserts exactly one audit entry, not two,
// guarding against the rejection's speculative route lookup leaking
// state into the shared route context loggingMiddleware reads later.
func TestAuditControlPlaneMutations_RecordsContentTypeRejectionsOnEncodedPaths(t *testing.T) {
	tests := []struct {
		name string
		path string
		want []string
	}{
		{name: "encoded slash within a real resource's id", path: "/api/v1/stages/foo%2Fbar", want: []string{"stages|PUT|error"}},
		{name: "encoded slash makes the path literally unrouted", path: "/api/v1/stages%2Fplanning", want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, m := newAuditTestServer(t, slog.New(slog.NewTextHandler(io.Discard, nil)))

			req := httptest.NewRequest(http.MethodPut, tt.path, nil)
			req.Header.Set("Content-Type", "text/plain")
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)

			assert.Equal(t, http.StatusUnsupportedMediaType, rr.Code)
			assert.Equal(t, tt.want, m.snapshot())
		})
	}
}

// TestAuditControlPlaneMutations_ServesWithoutMetrics covers a server
// built without an audit sink, which must serve control-plane writes
// rather than dereference the absent interface.
func TestAuditControlPlaneMutations_ServesWithoutMetrics(t *testing.T) {
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	srv.Router().Put("/api/v1/stages/{id}", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPut, "/api/v1/stages/planning", nil)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)

	// A rejection must not dereference the absent audit sink either.
	badReq := httptest.NewRequest(http.MethodPut, "/api/v1/stages/planning", nil)
	badReq.Header.Set("Content-Type", "text/plain")
	badRR := httptest.NewRecorder()
	srv.Router().ServeHTTP(badRR, badReq)

	assert.Equal(t, http.StatusUnsupportedMediaType, badRR.Code)
}
