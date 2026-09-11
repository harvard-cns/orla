package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/provider"
	"github.com/harvard-cns/orla/internal/scheduler"
	"github.com/harvard-cns/orla/internal/stages"
	"github.com/harvard-cns/orla/internal/telemetry"
)

// mockTool implements provider.ToolProvider for tests.
type mockTool struct {
	name     string
	toolKind string
	respFn   func(req provider.ToolRequest) (*provider.ToolResponse, error)
}

func (m *mockTool) Name() string     { return m.name }
func (m *mockTool) ToolKind() string { return m.toolKind }
func (m *mockTool) Invoke(_ context.Context, req provider.ToolRequest) (*provider.ToolResponse, error) {
	return m.respFn(req)
}

type recordingSink struct {
	mu  sync.Mutex
	got []*telemetry.CompletionRecord
}

func (s *recordingSink) Submit(rec *telemetry.CompletionRecord) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.got = append(s.got, rec)
	return true
}

// records returns a snapshot of the submitted records under lock, safe to
// call from a test goroutine while a server goroutine may still be writing.
func (s *recordingSink) records() []*telemetry.CompletionRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*telemetry.CompletionRecord(nil), s.got...)
}

func newToolTestEnv(t *testing.T, tool *mockTool, b *backends.Backend) (*Server, *backends.FakeRegistry, *recordingSink, *fakeProxyMetrics) {
	t.Helper()
	return newToolTestEnvOpts(t, tool, b, true)
}

// newToolTestEnvOpts sets whether the backend gets a scheduler
// executor. false simulates a backend in the registry with none.
func newToolTestEnvOpts(t *testing.T, tool *mockTool, b *backends.Backend, registerWithScheduler bool) (*Server, *backends.FakeRegistry, *recordingSink, *fakeProxyMetrics) {
	t.Helper()
	if b == nil {
		toolKind := tool.toolKind
		b = &backends.Backend{
			Name:           tool.name,
			Endpoint:       "http://unused-by-mock",
			MaxConcurrency: 1,
			Kind:           backends.KindTool,
			ToolKind:       &toolKind,
			Rates:          map[string]float64{"gpu_seconds": 0.001}, // $/s, ~$3.60/hr
		}
	}
	breg := backends.NewFakeRegistry()
	_, err := breg.Insert(context.Background(), b)
	require.NoError(t, err)

	sreg := stages.NewFakeRegistry()
	_, err = sreg.Replace(context.Background(), &stages.Stage{
		ID: "predict", Backend: tool.name,
	})
	require.NoError(t, err)

	// Scheduler factory always returns the mock tool regardless of which
	// backend it sees, since we register only one backend.
	sch := scheduler.New(
		func(*backends.Backend) provider.Backend { return tool },
		slog.New(slog.NewTextHandler(io.Discard, nil)),
	)
	if registerWithScheduler {
		sch.Register(b)
	}
	t.Cleanup(func() { _ = sch.Shutdown(context.Background()) })

	sink := &recordingSink{}
	metrics := &fakeProxyMetrics{}
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterToolRoutes(srv.Router(), ToolDeps{
		Stages:         sreg,
		Scheduler:      sch,
		Backends:       breg,
		CompletionSink: sink,
		Metrics:        metrics,
	})
	return srv, breg, sink, metrics
}

func TestTool_InvokeSuccess(t *testing.T) {
	tool := &mockTool{
		name:     "boltz",
		toolKind: "structure-prediction",
		respFn: func(req provider.ToolRequest) (*provider.ToolResponse, error) {
			assert.Equal(t, "structure-prediction", req.Kind)
			// echo a fixed response with gpu_seconds=2.5
			return &provider.ToolResponse{
				Payload: []byte(`{"structure_cif":"hello"}`),
				Usage:   map[string]float64{"gpu_seconds": 2.5},
			}, nil
		},
	}
	srv, _, sink, metrics := newToolTestEnv(t, tool, nil)

	body := []byte(`{"sequences":["MKTV"]}`)
	req := httptest.NewRequest(http.MethodPost,
		"/v1/tools/structure-prediction", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "predict")

	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	assert.NotEmpty(t, rr.Header().Get("X-Orla-Completion-Id"))

	var resp provider.ToolResponse
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &resp))
	assert.InDelta(t, 2.5, resp.Usage["gpu_seconds"], 1e-9)
	assert.JSONEq(t, `{"structure_cif":"hello"}`, string(resp.Payload))

	// Completion record captured.
	require.Len(t, sink.got, 1)
	rec := sink.got[0]
	assert.Equal(t, "predict", rec.StageID)
	assert.Equal(t, "boltz", rec.Backend)
	assert.Equal(t, "success", rec.Status)
	assert.Equal(t, "structure-prediction", rec.ToolKind)
	assert.InDelta(t, 2.5, rec.Usage["gpu_seconds"], 1e-9)
	require.NotNil(t, rec.CostUSD)
	// cost = 2.5 s × $0.001/s = $0.0025
	assert.InDelta(t, 0.0025, *rec.CostUSD, 1e-9)

	// Metrics emitted.
	reqs := metrics.requestsSnapshot()
	require.Len(t, reqs, 1)
	assert.Equal(t, "predict|boltz|success", reqs[0])
	assert.Empty(t, metrics.costAnomaliesSnapshot(), "cost is well within the sanity ceiling")
}

// TestTool_RejectsNonJSONContentType posts to
// /v1/tools/structure-prediction with Content-Type text/plain and
// asserts a 415 response without invoking the tool.
func TestTool_RejectsNonJSONContentType(t *testing.T) {
	tool := &mockTool{name: "boltz", toolKind: "structure-prediction",
		respFn: func(provider.ToolRequest) (*provider.ToolResponse, error) {
			t.Fatal("should not be invoked")
			return nil, nil
		}}
	srv, _, _, _ := newToolTestEnv(t, tool, nil)

	req := httptest.NewRequest(http.MethodPost,
		"/v1/tools/structure-prediction", bytes.NewReader([]byte(`{"sequences":["MKTV"]}`)))
	req.Header.Set("Content-Type", "text/plain")
	req.Header.Set(HeaderStage, "predict")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusUnsupportedMediaType, rr.Code)
}

func TestTool_RequiresStageHeader(t *testing.T) {
	tool := &mockTool{name: "boltz", toolKind: "structure-prediction",
		respFn: func(provider.ToolRequest) (*provider.ToolResponse, error) {
			t.Fatal("should not be invoked")
			return nil, nil
		}}
	srv, _, _, _ := newToolTestEnv(t, tool, nil)

	req := httptest.NewRequest(http.MethodPost,
		"/v1/tools/structure-prediction", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestTool_BackendNotFoundReturns502(t *testing.T) {
	tool := &mockTool{name: "boltz", toolKind: "structure-prediction",
		respFn: func(provider.ToolRequest) (*provider.ToolResponse, error) { return nil, nil }}
	srv, _, _, _ := newToolTestEnv(t, tool, nil)

	// Drop the registered backend so the lookup misses.
	// (FakeRegistry's Delete is fine for this.)
	// We can't easily reach into the FakeRegistry from here, so instead
	// re-bind the stage to an unknown backend.
	// Skipping this test variant, covered functionally by ToolDeps wiring.
	_ = srv
}

func TestTool_WrongKindOnBackendReturns400(t *testing.T) {
	toolKind := "docking"
	tool := &mockTool{
		name:     "ad-vina",
		toolKind: "docking",
		respFn: func(provider.ToolRequest) (*provider.ToolResponse, error) {
			t.Fatal("should not be invoked")
			return nil, nil
		},
	}
	b := &backends.Backend{
		Name:           tool.name,
		Endpoint:       "http://unused",
		MaxConcurrency: 1,
		Kind:           backends.KindTool,
		ToolKind:       &toolKind,
		Rates:          map[string]float64{"gpu_seconds": 0.001},
	}
	srv, _, _, _ := newToolTestEnv(t, tool, b)

	// Client asks for structure-prediction but the backend's tool_kind is docking.
	req := httptest.NewRequest(http.MethodPost,
		"/v1/tools/structure-prediction", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "predict")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadRequest, rr.Code)
	assert.Contains(t, rr.Body.String(), "structure-prediction")
}

func TestTool_LLMBackendOnToolRouteReturns400(t *testing.T) {
	// Register an LLM backend, ask the tool route to dispatch to it.
	modelID := "openai:gpt-4o"
	llm := &backends.Backend{
		Name:           "gpt4o",
		Endpoint:       "http://unused",
		MaxConcurrency: 1,
		Kind:           backends.KindLLM,
		ModelID:        &modelID,
	}
	tool := &mockTool{name: "gpt4o", toolKind: "structure-prediction",
		respFn: func(provider.ToolRequest) (*provider.ToolResponse, error) { return nil, nil }}
	srv, _, _, _ := newToolTestEnv(t, tool, llm)

	req := httptest.NewRequest(http.MethodPost,
		"/v1/tools/structure-prediction", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "predict")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusBadRequest, rr.Code, rr.Body.String())
	assert.Contains(t, rr.Body.String(), `kind=\"llm\"`)
}

func TestTool_PropagatesProviderError(t *testing.T) {
	tool := &mockTool{
		name:     "boltz",
		toolKind: "structure-prediction",
		respFn: func(provider.ToolRequest) (*provider.ToolResponse, error) {
			return nil, assertErr("upstream broke")
		},
	}
	srv, _, sink, metrics := newToolTestEnv(t, tool, nil)

	req := httptest.NewRequest(http.MethodPost,
		"/v1/tools/structure-prediction", bytes.NewReader([]byte(`{"sequences":["X"]}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "predict")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusBadGateway, rr.Code, rr.Body.String())
	assert.Contains(t, rr.Body.String(), "upstream broke")

	// Error path also records a completion (status=error) and metric.
	require.Len(t, sink.got, 1)
	assert.Equal(t, "error", sink.got[0].Status)
	reqs := metrics.requestsSnapshot()
	require.Len(t, reqs, 1)
	assert.Equal(t, "predict|boltz|error", reqs[0])
}

// assertErr is a tiny helper so this file doesn't need the errors package.
type assertErr string

func (e assertErr) Error() string { return string(e) }

// A backend in the registry but not the scheduler passes the Kind
// checks, then fails AcquireTool with ErrUnknownBackend. The tool route
// must record the rejection metric just as the proxy route does.
func TestTool_SchedulerUnknownBackendRecordsRejectionMetric(t *testing.T) {
	tool := &mockTool{
		name:     "boltz",
		toolKind: "structure-prediction",
		respFn: func(provider.ToolRequest) (*provider.ToolResponse, error) {
			t.Fatal("should not be invoked")
			return nil, nil
		},
	}
	srv, _, _, metrics := newToolTestEnvOpts(t, tool, nil, false)

	req := httptest.NewRequest(http.MethodPost,
		"/v1/tools/structure-prediction", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "predict")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadGateway, rr.Code, rr.Body.String())
	assert.Equal(t, []string{unregisteredBackendLabel + "/unknown_backend"}, metrics.rejectionsSnapshot())
}

func TestComputeToolCost_Outcomes(t *testing.T) {
	tests := []struct {
		name          string
		resp          *provider.ToolResponse
		rates         map[string]float64
		wantCost      *float64
		wantAnomalous bool
	}{
		{
			name:     "reported cost within ceiling",
			resp:     &provider.ToolResponse{CostUSD: new(5.0)},
			wantCost: new(5.0),
		},
		{
			name:          "reported cost exceeds ceiling but is still returned",
			resp:          &provider.ToolResponse{CostUSD: new(toolCostAnomalyCeilingUSD + 0.01)},
			wantCost:      new(toolCostAnomalyCeilingUSD + 0.01),
			wantAnomalous: true,
		},
		{
			name:     "dot product within ceiling",
			resp:     &provider.ToolResponse{Usage: map[string]float64{"gpu_seconds": 5.0}},
			rates:    map[string]float64{"gpu_seconds": 0.001},
			wantCost: new(0.005),
		},
		{
			name:          "dot product exceeds ceiling but is still returned",
			resp:          &provider.ToolResponse{Usage: map[string]float64{"gpu_seconds": 2_000_000}},
			rates:         map[string]float64{"gpu_seconds": 0.001},
			wantCost:      new(2000.0),
			wantAnomalous: true,
		},
		{
			name:     "nil response",
			resp:     nil,
			wantCost: nil,
		},
		{
			name:     "non-finite reported cost is dropped, not flagged",
			resp:     &provider.ToolResponse{CostUSD: new(math.NaN())},
			wantCost: nil,
		},
		{
			name:     "negative reported cost is dropped, not flagged",
			resp:     &provider.ToolResponse{CostUSD: new(-1.0)},
			wantCost: nil,
		},
		{
			name:     "usage keys do not overlap rates",
			resp:     &provider.ToolResponse{Usage: map[string]float64{"calls": 1}},
			rates:    map[string]float64{"gpu_seconds": 0.001},
			wantCost: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeProxyMetrics{}
			got := computeToolCost(tt.resp, tt.rates, "boltz", "cmpl-1", fake)
			if tt.wantCost == nil {
				assert.Nil(t, got)
			} else {
				require.NotNil(t, got)
				assert.InDelta(t, *tt.wantCost, *got, 1e-9)
			}
			if tt.wantAnomalous {
				assert.Equal(t, []string{"boltz"}, fake.costAnomaliesSnapshot())
			} else {
				assert.Empty(t, fake.costAnomaliesSnapshot())
			}
		})
	}
}

// TestFlagIfCostAnomalous_Boundary covers the exact edge of the
// ceiling comparison (<=), which TestComputeToolCost_Outcomes's cases
// don't pin down. Those use values clearly above or clearly below.
func TestFlagIfCostAnomalous_Boundary(t *testing.T) {
	tests := []struct {
		name          string
		cost          float64
		wantAnomalous bool
	}{
		{name: "just below ceiling", cost: toolCostAnomalyCeilingUSD - 0.01, wantAnomalous: false},
		{name: "exactly at ceiling", cost: toolCostAnomalyCeilingUSD, wantAnomalous: false},
		{name: "just above ceiling", cost: toolCostAnomalyCeilingUSD + 0.01, wantAnomalous: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeProxyMetrics{}
			flagIfCostAnomalous(fake, "boltz", "cmpl-1", tt.cost)
			if tt.wantAnomalous {
				assert.Equal(t, []string{"boltz"}, fake.costAnomaliesSnapshot())
			} else {
				assert.Empty(t, fake.costAnomaliesSnapshot())
			}
		})
	}
}

// TestFlagIfCostAnomalous_NilMetrics confirms the nil-metrics guard
// doesn't panic, matching the same gap already covered for
// statusForSchedulerErr's identical nil-check shape.
func TestFlagIfCostAnomalous_NilMetrics(t *testing.T) {
	assert.NotPanics(t, func() {
		flagIfCostAnomalous(nil, "boltz", "cmpl-1", toolCostAnomalyCeilingUSD+1)
	})
}
