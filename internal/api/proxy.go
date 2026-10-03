package api

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/costs"
	"github.com/harvard-cns/orla/internal/mappings"
	"github.com/harvard-cns/orla/internal/scheduler"
	"github.com/harvard-cns/orla/internal/stages"
	"github.com/harvard-cns/orla/internal/telemetry"
)

// Header names. Lookups are case-insensitive because chi and net/http
// normalize incoming headers to canonical case.
const (
	HeaderStage       = "X-Orla-Stage"
	HeaderWorkflowRun = "X-Orla-Workflow-Run"
	HeaderMapping     = "X-Orla-Mapping"
	HeaderTagPrefix   = "X-Orla-Tag-"
)

// metadata key fallbacks for SDKs that can't easily set headers.
const (
	metaStage       = "orla.stage"
	metaWorkflowRun = "orla.workflow_run"
	metaMapping     = "orla.mapping"
)

// CompletionSink receives one record per dispatched chat completion.
// Implementations are typically wrapping telemetry.CompletionWriter;
// nil is acceptable for tests that don't care about records.
type CompletionSink interface {
	Submit(rec *telemetry.CompletionRecord) bool
}

// ProxyMetrics is the subset of metrics.Metrics consumed by the proxy
// hot path. Nil is allowed for tests that don't care about metrics.
type ProxyMetrics interface {
	IncRequest(stage, backend, status string)
	ObserveBackendLatency(backend string, seconds float64)
	IncSchedulerRejection(backend, reason string)
	IncToolCostAnomaly(backend string)
	IncStageMapperDecision(outcome string)
	ObserveStageMapperDecision(seconds float64)
}

// LiveCosts serves the current polled price for a backend. Implemented
// by costs.Store.
type LiveCosts interface {
	Get(name string) (costs.Price, bool)
}

// ProxyDeps bundles the dependencies of the proxy handler. Mappings
// may be nil, then no request can select a variant and the live stage
// mapping always resolves. Costs may be nil, then every backend prices
// through its static columns. StageMapper may be nil, then every
// stage routes by its static mapping.
type ProxyDeps struct {
	Stages         stages.Registry
	Mappings       mappings.Registry
	Scheduler      *scheduler.Scheduler
	CompletionSink CompletionSink
	Metrics        ProxyMetrics
	Costs          LiveCosts
	StageMapper    *mappings.MapperHolder
}

// RegisterProxyRoutes mounts POST /v1/chat/completions.
func RegisterProxyRoutes(r chi.Router, deps ProxyDeps) {
	h := &proxyHandler{deps: deps}
	r.Post("/v1/chat/completions", h.chatCompletions)
}

type proxyHandler struct {
	deps ProxyDeps
}

// requestContext aggregates the identity metadata we extract from
// headers + body fallbacks. Stages and tags are persisted later by the
// completion-records writer.
type requestContext struct {
	Stage       string
	WorkflowRun string
	Mapping     string
	Tags        map[string]string

	CaptureIO      bool
	RequestContent string
}

func (h *proxyHandler) chatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeBodyError(w, fmt.Errorf("read body: %w", err))
		return
	}

	// Peek for the stream flag, openai.ChatCompletionNewParams doesn't
	// carry it, client-side it's controlled by which method (New vs
	// NewStreaming) is called.
	var peek struct {
		Stream bool `json:"stream"`
	}
	if len(body) > 0 {
		_ = json.Unmarshal(body, &peek)
	}

	var params openai.ChatCompletionNewParams
	if err := json.Unmarshal(body, &params); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("decode chat completion params: %w", err))
		return
	}
	if len(params.Messages) == 0 {
		writeErrorMsg(w, http.StatusBadRequest, "messages is required and must not be empty")
		return
	}
	if err := reattachJSONSchema(&params, body); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	rc := extractRequestContext(r, params.Metadata)
	if rc.Stage == "" {
		writeErrorMsg(w, http.StatusBadRequest, "stage is required (set X-Orla-Stage or metadata.orla.stage)")
		return
	}

	stage, err := h.deps.Stages.GetOrCreate(r.Context(), rc.Stage)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	backendName := stage.Backend
	// A shadow request selects a mapping variant. When the variant
	// overrides this stage, its backend wins over the live mapping and
	// the stage mapper, since a variant is an explicit per-request
	// pin. A stage absent from the variant falls through.
	variantOverride := false
	if rc.Mapping != "" && h.deps.Mappings != nil {
		if override, ok := h.deps.Mappings.Resolve(r.Context(), rc.Mapping, rc.Stage); ok {
			backendName = override
			variantOverride = true
		}
	}
	if !variantOverride {
		if decided, ok := h.askStageMapper(r.Context(), rc, backendName); ok {
			backendName = decided
		}
	}
	if backendName == "" {
		// Fall back to the client-supplied model field if the stage
		// has no mapping yet.
		backendName = string(params.Model)
	}
	if backendName == "" {
		writeErrorMsg(w, http.StatusBadRequest,
			fmt.Sprintf("stage %q has no backend mapping and request did not specify model", rc.Stage))
		return
	}

	if stage.CaptureIO {
		rc.CaptureIO = true
		rc.RequestContent = string(body)
	}

	// Apply stage-level inference policy.
	if stage.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(stage.ReasoningEffort)
	}
	if stage.Prompt != "" {
		applyStagePrompt(&params, stage.Prompt)
	}

	if peek.Stream {
		h.serveStreaming(w, r, rc, backendName, params)
		return
	}
	h.serveNonStreaming(w, r, rc, backendName, params)
}

// askStageMapper asks the stage mapper which backend should serve
// this request and reports whether it chose one. Any failure falls
// back to the static mapping, so a broken mapper service degrades
// routing rather than availability. A decision naming a backend the
// proxy did not offer is treated the same way.
func (h *proxyHandler) askStageMapper(ctx context.Context, rc *requestContext, current string) (string, bool) {
	if h.deps.StageMapper == nil {
		return "", false
	}
	mapper := h.deps.StageMapper.Get()
	if mapper == nil {
		return "", false
	}

	candidates := h.mapperCandidates()
	if len(candidates) == 0 {
		return "", false
	}

	start := time.Now()
	decided, err := mapper.Decide(ctx, mappings.DecideRequest{
		Stage:      rc.Stage,
		Tags:       rc.Tags,
		Current:    current,
		Candidates: candidates,
	})
	h.observeMapperDecision(time.Since(start).Seconds())

	switch {
	case errors.Is(err, context.DeadlineExceeded):
		h.incMapperDecision("fallback_timeout")
	case err != nil:
		h.incMapperDecision("fallback_error")
	case decided == "":
		h.incMapperDecision("declined")
		return "", false
	case !slices.ContainsFunc(candidates, func(c mappings.Candidate) bool { return c.Name == decided }):
		h.incMapperDecision("fallback_invalid")
		err = fmt.Errorf("mapper chose %q, which was not offered", decided)
	default:
		h.incMapperDecision("ok")
		return decided, true
	}
	slog.Default().Warn("proxy: stage mapper fell back to the static mapping",
		"stage", rc.Stage,
		"error", err,
	)
	return "", false
}

// mapperCandidates lists every LLM backend the mapper may choose,
// priced the way the completion will be billed. A live polled price
// wins over the static columns.
func (h *proxyHandler) mapperCandidates() []mappings.Candidate {
	stats := h.deps.Scheduler.Stats()
	candidates := make([]mappings.Candidate, 0, len(stats))
	for _, s := range stats {
		b, ok := h.deps.Scheduler.BackendOf(s.Backend)
		if !ok || b.Kind != backends.KindLLM {
			continue
		}
		in, out := b.InputCostPerMtoken, b.OutputCostPerMtoken
		if h.deps.Costs != nil {
			if p, ok := h.deps.Costs.Get(s.Backend); ok {
				in, out = &p.InputPerMtoken, &p.OutputPerMtoken
			}
		}
		candidates = append(candidates, mappings.Candidate{
			Name:                s.Backend,
			Quality:             b.Quality,
			InputCostPerMtoken:  in,
			OutputCostPerMtoken: out,
			QueueDepth:          s.QueueDepth,
			InFlight:            s.InFlight,
			Capacity:            s.Capacity,
			Circuit:             s.CircuitState,
		})
	}
	return candidates
}

// incMapperDecision and observeMapperDecision are no-ops when
// ProxyMetrics is nil.
func (h *proxyHandler) incMapperDecision(outcome string) {
	if h.deps.Metrics != nil {
		h.deps.Metrics.IncStageMapperDecision(outcome)
	}
}

func (h *proxyHandler) observeMapperDecision(seconds float64) {
	if h.deps.Metrics != nil {
		h.deps.Metrics.ObserveStageMapperDecision(seconds)
	}
}

// reattachJSONSchema restores a json_schema response format onto params.
// The SDK param union keeps the format type when it unmarshals the
// request body but drops the nested schema, so a structured-output
// request would otherwise reach the backend without its schema.
// Rebuilding the concrete param from the raw body preserves it.
func reattachJSONSchema(params *openai.ChatCompletionNewParams, body []byte) error {
	var raw struct {
		ResponseFormat json.RawMessage `json:"response_format"`
	}
	if err := json.Unmarshal(body, &raw); err != nil || len(raw.ResponseFormat) == 0 {
		return nil
	}
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw.ResponseFormat, &head); err != nil {
		return fmt.Errorf("decode response_format type: %w", err)
	}
	if head.Type != "json_schema" {
		return nil
	}
	var js shared.ResponseFormatJSONSchemaParam
	if err := json.Unmarshal(raw.ResponseFormat, &js); err != nil {
		return fmt.Errorf("decode response_format json_schema: %w", err)
	}
	params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &js}
	return nil
}

// applyStagePrompt substitutes the stage's prompt for the request's
// leading instruction message, the system or developer message SDKs use
// for instructions. It replaces that message in place and keeps its
// role, or prepends a system message when the first message is neither.
// The rest of the conversation is left untouched, so a tool-calling loop
// keeps its scratchpad and only the instructions change. The caller
// guarantees prompt is non-empty.
func applyStagePrompt(params *openai.ChatCompletionNewParams, prompt string) {
	if len(params.Messages) > 0 {
		switch {
		case params.Messages[0].OfSystem != nil:
			params.Messages[0] = openai.SystemMessage(prompt)
			return
		case params.Messages[0].OfDeveloper != nil:
			params.Messages[0] = openai.DeveloperMessage(prompt)
			return
		}
	}
	params.Messages = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(prompt)}, params.Messages...)
}

func (h *proxyHandler) serveNonStreaming(w http.ResponseWriter, r *http.Request, rc *requestContext, backendName string, params openai.ChatCompletionNewParams) {
	p, release, err := h.deps.Scheduler.AcquireLLM(r.Context(), backendName, schedulerRequestInfo(rc, params))
	if err != nil {
		statusForSchedulerErr(w, err, backendName, h.deps.Metrics)
		return
	}
	defer release()

	start := time.Now()
	resp, err := p.Chat(r.Context(), params)
	latencyMs := int(time.Since(start) / time.Millisecond)
	h.deps.Scheduler.ReportOutcome(backendName, err)

	if err != nil {
		h.recordCompletion(&completionInputs{
			completionID: uuid.NewString(),
			rc:           rc,
			backend:      backendName,
			status:       "error",
			latencyMs:    &latencyMs,
		})
		h.emitMetrics(rc.Stage, backendName, "error", latencyMs)
		writeUpstreamError(w, err)
		return
	}

	// The body's "model" field is the canonical source of truth for
	// which backend ran the call. Overwrite with the resolved backend
	// name so the developer can correlate against /api/v1/backends.
	resp.Model = backendName

	completionID := resp.ID
	if completionID == "" {
		completionID = uuid.NewString()
		resp.ID = completionID
	}
	prompt := int(resp.Usage.PromptTokens)
	completion := int(resp.Usage.CompletionTokens)
	cached := int(resp.Usage.PromptTokensDetails.CachedTokens)
	costUSD := h.computeLLMCost(backendName, prompt, completion, cached, completionID)
	var responseContent string
	if rc.CaptureIO {
		if b, err := json.Marshal(resp); err == nil {
			responseContent = string(b)
		}
	}
	h.recordCompletion(&completionInputs{
		completionID:     completionID,
		rc:               rc,
		backend:          backendName,
		status:           "success",
		promptTokens:     &prompt,
		completionTokens: &completion,
		cachedTokens:     &cached,
		latencyMs:        &latencyMs,
		costUSD:          costUSD,
		responseContent:  responseContent,
	})
	h.emitMetrics(rc.Stage, backendName, "success", latencyMs)

	writeJSON(w, http.StatusOK, resp)
}

func (h *proxyHandler) serveStreaming(w http.ResponseWriter, r *http.Request, rc *requestContext, backendName string, params openai.ChatCompletionNewParams) {
	p, release, err := h.deps.Scheduler.AcquireLLM(r.Context(), backendName, schedulerRequestInfo(rc, params))
	if err != nil {
		statusForSchedulerErr(w, err, backendName, h.deps.Metrics)
		return
	}
	defer release()

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErrorMsg(w, http.StatusInternalServerError, "streaming not supported by this server")
		return
	}

	modelJSON, err := json.Marshal(backendName)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("encode model name: %w", err))
		return
	}

	start := time.Now()
	var completionID string
	var promptTokens, completionTokens, cachedTokens int
	var captured strings.Builder

	stream := p.ChatStream(r.Context(), params)
	defer func() { _ = stream.Close() }()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	for stream.Next() {
		chunk := stream.Current()
		if completionID == "" && chunk.ID != "" {
			completionID = chunk.ID
		}
		// Some upstreams (notably OpenAI) only attach Usage to the
		// final chunk. We capture whichever value lands last.
		if chunk.Usage.PromptTokens > 0 {
			promptTokens = int(chunk.Usage.PromptTokens)
		}
		if chunk.Usage.CompletionTokens > 0 {
			completionTokens = int(chunk.Usage.CompletionTokens)
		}
		if chunk.Usage.PromptTokensDetails.CachedTokens > 0 {
			cachedTokens = int(chunk.Usage.PromptTokensDetails.CachedTokens)
		}
		if rc.CaptureIO && len(chunk.Choices) > 0 {
			captured.WriteString(chunk.Choices[0].Delta.Content)
		}
		data := rechunkWithModel(chunk.RawJSON(), modelJSON)
		if _, werr := fmt.Fprintf(w, "data: %s\n\n", data); werr != nil {
			return
		}
		flusher.Flush()
	}

	latencyMs := int(time.Since(start) / time.Millisecond)

	streamErr := stream.Err()
	if errors.Is(streamErr, io.EOF) {
		streamErr = nil
	}
	h.deps.Scheduler.ReportOutcome(backendName, streamErr)

	if streamErr != nil {
		errEnv := errorEnvelope{Error: errorBody{Message: streamErr.Error(), Type: "server_error"}}
		if data, mErr := json.Marshal(errEnv); mErr == nil {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
		if completionID == "" {
			completionID = uuid.NewString()
		}
		h.recordCompletion(&completionInputs{
			completionID: completionID,
			rc:           rc,
			backend:      backendName,
			status:       "error",
			latencyMs:    &latencyMs,
		})
		h.emitMetrics(rc.Stage, backendName, "error", latencyMs)
		return
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()

	if completionID == "" {
		completionID = uuid.NewString()
	}
	costUSD := h.computeLLMCost(backendName, promptTokens, completionTokens, cachedTokens, completionID)
	h.recordCompletion(&completionInputs{
		completionID:     completionID,
		rc:               rc,
		backend:          backendName,
		status:           "success",
		promptTokens:     reportedCount(promptTokens),
		completionTokens: reportedCount(completionTokens),
		cachedTokens:     reportedCount(cachedTokens),
		latencyMs:        &latencyMs,
		costUSD:          costUSD,
		responseContent:  captured.String(),
	})
	h.emitMetrics(rc.Stage, backendName, "success", latencyMs)
}

// rechunkWithModel returns the upstream chunk JSON with its model field
// replaced by modelJSON and every other field forwarded unchanged.
// Re-encoding the decoded openai-go struct would emit zero-value fields
// such as an empty delta.role that strict OpenAI clients reject.
func rechunkWithModel(raw string, modelJSON json.RawMessage) string {
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return raw
	}
	m["model"] = modelJSON
	out, err := json.Marshal(m)
	if err != nil {
		return raw
	}
	return string(out)
}

// computeLLMCost rolls token counts and the backend's per-million-
// token rates into a dollar amount. A live polled price takes
// precedence over the backend's static columns. cachedTokens is the
// share of promptTokens the provider served from its cache, priced at
// the backend's cache read rate when it declares one and at the input
// rate otherwise. Returns nil if the backend has no configured rates
// or the scheduler does not know about this backend name. A non-finite
// result is dropped with a log line so a configuration error cannot
// poison cost aggregates.
func (h *proxyHandler) computeLLMCost(backendName string, promptTokens, completionTokens, cachedTokens int, completionID string) *float64 {
	b, ok := h.deps.Scheduler.BackendOf(backendName)
	if !ok {
		return nil
	}
	in, out := b.InputCostPerMtoken, b.OutputCostPerMtoken
	if h.deps.Costs != nil {
		if p, ok := h.deps.Costs.Get(backendName); ok {
			in, out = &p.InputPerMtoken, &p.OutputPerMtoken
		}
	}
	if in == nil && out == nil {
		return nil
	}
	var cost float64
	if in != nil {
		// A provider can report more cached tokens than prompt tokens.
		// The clamp holds the uncached share at zero or above.
		cached := min(max(cachedTokens, 0), promptTokens)
		cacheRate := *in
		if b.CacheReadCostPerMtoken != nil {
			cacheRate = *b.CacheReadCostPerMtoken
		}
		cost += float64(promptTokens-cached) * (*in) / 1_000_000.0
		cost += float64(cached) * cacheRate / 1_000_000.0
	}
	if out != nil {
		cost += float64(completionTokens) * (*out) / 1_000_000.0
	}
	if math.IsNaN(cost) || math.IsInf(cost, 0) || cost < 0 {
		slog.Default().Warn("proxy: LLM cost computation produced non-finite or negative value",
			"backend", backendName,
			"completion_id", completionID,
			"cost_usd", cost,
		)
		return nil
	}
	return &cost
}

type completionInputs struct {
	completionID     string
	rc               *requestContext
	backend          string
	status           string
	promptTokens     *int
	completionTokens *int
	cachedTokens     *int
	latencyMs        *int
	costUSD          *float64
	responseContent  string
}

// reportedCount returns a pointer to n, or nil when n is zero. A streamed
// response can finish without ever carrying a usage block, so a zero count
// there means the upstream reported nothing and records as NULL.
func reportedCount(n int) *int {
	if n == 0 {
		return nil
	}
	return &n
}

// emitMetrics is a no-op when ProxyMetrics is nil.
func (h *proxyHandler) emitMetrics(stage, backend, status string, latencyMs int) {
	if h.deps.Metrics == nil {
		return
	}
	h.deps.Metrics.IncRequest(stage, backend, status)
	h.deps.Metrics.ObserveBackendLatency(backend, float64(latencyMs)/1000.0)
}

func (h *proxyHandler) recordCompletion(in *completionInputs) {
	if h.deps.CompletionSink == nil {
		return
	}
	rec := &telemetry.CompletionRecord{
		CompletionID:     in.completionID,
		StageID:          in.rc.Stage,
		WorkflowRun:      in.rc.WorkflowRun,
		Backend:          in.backend,
		Status:           in.status,
		PromptTokens:     in.promptTokens,
		CompletionTokens: in.completionTokens,
		CachedTokens:     in.cachedTokens,
		LatencyMs:        in.latencyMs,
		CostUSD:          in.costUSD,
		Tags:             in.rc.Tags,
		Mapping:          in.rc.Mapping,
		CreatedAt:        time.Now(),
	}
	if in.rc.CaptureIO {
		rec.IO = &telemetry.CapturedIO{
			Request:  in.rc.RequestContent,
			Response: in.responseContent,
		}
	}
	_ = h.deps.CompletionSink.Submit(rec)
}

// extractRequestContext parses X-Orla-* headers and metadata fallbacks.
// Header values win over body metadata when both are set.
func extractRequestContext(r *http.Request, metadata shared.Metadata) *requestContext {
	rc := &requestContext{Tags: make(map[string]string)}

	rc.Stage = cmp.Or(r.Header.Get(HeaderStage), metadata[metaStage])
	rc.WorkflowRun = cmp.Or(r.Header.Get(HeaderWorkflowRun), metadata[metaWorkflowRun])
	rc.Mapping = cmp.Or(r.Header.Get(HeaderMapping), metadata[metaMapping])

	for name, values := range r.Header {
		if !strings.HasPrefix(name, HeaderTagPrefix) {
			continue
		}
		if len(values) == 0 {
			continue
		}
		key := strings.ToLower(strings.TrimPrefix(name, HeaderTagPrefix))
		if key == "" {
			continue
		}
		rc.Tags[key] = values[0]
	}
	return rc
}

// schedulerRequestInfo builds the metadata the scheduler forwards to a
// scheduling policy. The model is the client-requested model, which may
// differ from the resolved backend name.
func schedulerRequestInfo(rc *requestContext, params openai.ChatCompletionNewParams) scheduler.RequestInfo {
	return scheduler.RequestInfo{
		Stage: rc.Stage,
		Model: string(params.Model),
		Tags:  rc.Tags,
	}
}

// unregisteredBackendLabel replaces an unknown backend name in the
// rejection metric. The name can be arbitrary client input, so using it
// verbatim would let a client mint unbounded Prometheus series.
const unregisteredBackendLabel = "unregistered"

// statusForSchedulerErr classifies a scheduler acquire error into an
// HTTP response and a metric reason.
func statusForSchedulerErr(w http.ResponseWriter, err error, backendName string, metrics ProxyMetrics) {
	// Every branch below except unknown_backend labels the metric with
	// backendName verbatim. That is safe only because Scheduler.Acquire
	// returns ErrUnknownBackend from the registry lookup before it ever
	// consults the context or the executor, so any other error implies a
	// registered backend and a bounded label. An arbitrary client-supplied
	// model can reach here only as unknown_backend, which uses the fixed
	// unregisteredBackendLabel. Preserve that ordering in the scheduler or
	// this becomes an unbounded-cardinality hole.
	reason := "internal_error"
	metricBackend := backendName
	_, isCircuitOpen := errors.AsType[*scheduler.CircuitOpenError](err)
	switch {
	case errors.Is(err, scheduler.ErrUnknownBackend):
		reason = "unknown_backend"
		metricBackend = unregisteredBackendLabel
		writeError(w, http.StatusBadGateway, fmt.Errorf("backend %q is not registered", backendName))
	case isCircuitOpen:
		reason = "circuit_open"
		// Backend is failing fast. Signal "retry later" with 503 rather than
		// a generic 500. Retry-After mirrors the breaker's open window.
		w.Header().Set("Retry-After", "60")
		writeError(w, http.StatusServiceUnavailable, err)
	case errors.Is(err, scheduler.ErrWrongKind):
		reason = "wrong_kind"
		writeError(w, http.StatusInternalServerError, err)
	case errors.Is(err, context.Canceled):
		reason = "canceled"
		writeError(w, http.StatusRequestTimeout, err)
	case errors.Is(err, context.DeadlineExceeded):
		reason = "deadline_exceeded"
		writeError(w, http.StatusRequestTimeout, err)
	default:
		writeError(w, http.StatusInternalServerError, err)
	}
	if metrics != nil {
		metrics.IncSchedulerRejection(metricBackend, reason)
	}
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	if apiErr, ok := errors.AsType[*openai.Error](err); ok {
		// Mirror the upstream status when sensible (4xx). 5xx surfaces
		// as a 502 to make "orla failed" distinguishable from "client
		// asked for something silly".
		if apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 {
			writeError(w, apiErr.StatusCode, err)
			return
		}
	}
	writeError(w, http.StatusBadGateway, err)
}
