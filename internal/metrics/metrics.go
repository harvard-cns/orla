// Package metrics owns the Prometheus metric definitions and the
// custom collectors that pull point-in-time gauges from the scheduler
// and the BatchWriter instances.
//
// Hot-path metrics (request counter, latency histogram) are
// push-style: handlers call methods on Metrics directly.
// Sampled metrics (queue depth, in-flight, batch writer drops) are
// pull-style: a custom Collector emits them at scrape time.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

// Metrics is the set of push-style counters/histograms emitted by
// proxy and feedback handlers.
type Metrics struct {
	RequestsTotal            *prometheus.CounterVec
	BackendLatency           *prometheus.HistogramVec
	FeedbackTotal            *prometheus.CounterVec
	SchedulerRejectionsTotal *prometheus.CounterVec
	PolicyDecisionsTotal     *prometheus.CounterVec
	PolicyDecisionSeconds    *prometheus.HistogramVec
	ToolCostAnomalyTotal     *prometheus.CounterVec
	CostFetchFailuresTotal   *prometheus.CounterVec

	ControlPlaneMutationsTotal *prometheus.CounterVec
	RequestBodyTooLargeTotal   *prometheus.CounterVec

	StageMapperDecisionsTotal  *prometheus.CounterVec
	StageMapperDecisionSeconds prometheus.Histogram
}

// New constructs and registers all push-style metrics on reg.
func New(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		RequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "requests_total",
				Help:      "Total chat completion requests dispatched, by stage, backend, and status (success|error).",
			},
			[]string{"stage", "backend", "status"},
		),
		BackendLatency: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "orla",
				Name:      "backend_latency_seconds",
				Help:      "Latency of upstream backend dispatch in seconds.",
				// 50ms .. 60s
				Buckets: prometheus.ExponentialBuckets(0.05, 2, 11),
			},
			[]string{"backend"},
		),
		FeedbackTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "feedback_total",
				Help:      "Total feedback submissions, by stage.",
			},
			[]string{"stage"},
		),
		SchedulerRejectionsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "scheduler_rejections_total",
				Help:      "Requests that failed to acquire a scheduler slot, by backend and reason.",
			},
			[]string{"backend", "reason"},
		),
		PolicyDecisionsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "scheduler_policy_decisions_total",
				Help:      "Scheduling policy decisions, by backend and outcome (ok|fallback_timeout|fallback_error|fallback_invalid).",
			},
			[]string{"backend", "outcome"},
		),
		PolicyDecisionSeconds: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "orla",
				Name:      "scheduler_policy_decision_seconds",
				Help:      "Latency of a scheduling policy decision in seconds.",
				// 1ms .. ~1s
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 11),
			},
			[]string{"backend"},
		),
		ControlPlaneMutationsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "control_plane_mutations_total",
				Help:      "Mutating requests to the /api/v1 control plane, by resource, method, and outcome (success|error).",
			},
			[]string{"resource", "method", "outcome"},
		),
		RequestBodyTooLargeTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "request_body_too_large_total",
				Help:      "Requests whose body ran past ORLA_MAX_REQUEST_BYTES, by matched route pattern.",
			},
			[]string{"route"},
		),
		ToolCostAnomalyTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "tool_cost_anomaly_total",
				Help:      "Tool dispatches whose reported cost exceeded the sanity ceiling, by backend.",
			},
			[]string{"backend"},
		),
		CostFetchFailuresTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "cost_fetch_failures_total",
				Help:      "Failed reads of a backend's cost source, by backend. A rising count means completions are priced from a stale price.",
			},
			[]string{"backend"},
		),
		StageMapperDecisionsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "orla",
				Name:      "stage_mapper_decisions_total",
				Help:      "Stage mapper decisions by outcome: ok, declined, or a fallback_* reason.",
			},
			[]string{"outcome"},
		),
		StageMapperDecisionSeconds: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: "orla",
				Name:      "stage_mapper_decision_seconds",
				Help:      "Latency of a stage mapper decision in seconds.",
				// 1ms .. ~1s
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 11),
			},
		),
	}
	reg.MustRegister(
		m.RequestsTotal,
		m.BackendLatency,
		m.FeedbackTotal,
		m.SchedulerRejectionsTotal,
		m.PolicyDecisionsTotal,
		m.PolicyDecisionSeconds,
		m.ControlPlaneMutationsTotal,
		m.RequestBodyTooLargeTotal,
		m.ToolCostAnomalyTotal,
		m.CostFetchFailuresTotal,
		m.StageMapperDecisionsTotal,
		m.StageMapperDecisionSeconds,
	)
	return m
}

// IncRequest is the api.ProxyMetrics adapter.
func (m *Metrics) IncRequest(stage, backend, status string) {
	m.RequestsTotal.WithLabelValues(stage, backend, status).Inc()
}

// ObserveBackendLatency is the api.ProxyMetrics adapter.
func (m *Metrics) ObserveBackendLatency(backend string, seconds float64) {
	m.BackendLatency.WithLabelValues(backend).Observe(seconds)
}

// IncFeedback is the api.FeedbackMetrics adapter.
func (m *Metrics) IncFeedback(stage string) {
	m.FeedbackTotal.WithLabelValues(stage).Inc()
}

// IncSchedulerRejection is the api.ProxyMetrics adapter.
func (m *Metrics) IncSchedulerRejection(backend, reason string) {
	m.SchedulerRejectionsTotal.WithLabelValues(backend, reason).Inc()
}

// IncPolicyDecision is the scheduler.PolicyMetrics adapter.
func (m *Metrics) IncPolicyDecision(backend, outcome string) {
	m.PolicyDecisionsTotal.WithLabelValues(backend, outcome).Inc()
}

// ObservePolicyDecision is the scheduler.PolicyMetrics adapter.
func (m *Metrics) ObservePolicyDecision(backend string, seconds float64) {
	m.PolicyDecisionSeconds.WithLabelValues(backend).Observe(seconds)
}

// IncControlPlaneMutation is the api.ControlPlaneAuditMetrics adapter.
func (m *Metrics) IncControlPlaneMutation(resource, method, outcome string) {
	m.ControlPlaneMutationsTotal.WithLabelValues(resource, method, outcome).Inc()
}

// IncRequestBodyTooLarge is the api.BodyLimitMetrics adapter.
func (m *Metrics) IncRequestBodyTooLarge(route string) {
	m.RequestBodyTooLargeTotal.WithLabelValues(route).Inc()
}

// IncToolCostAnomaly is the api.ProxyMetrics adapter.
func (m *Metrics) IncToolCostAnomaly(backend string) {
	m.ToolCostAnomalyTotal.WithLabelValues(backend).Inc()
}

// IncCostFetchFailure is the costs.PollerMetrics adapter.
func (m *Metrics) IncCostFetchFailure(backend string) {
	m.CostFetchFailuresTotal.WithLabelValues(backend).Inc()
}

// IncStageMapperDecision is the api.ProxyMetrics adapter.
func (m *Metrics) IncStageMapperDecision(outcome string) {
	m.StageMapperDecisionsTotal.WithLabelValues(outcome).Inc()
}

// ObserveStageMapperDecision is the api.ProxyMetrics adapter.
func (m *Metrics) ObserveStageMapperDecision(seconds float64) {
	m.StageMapperDecisionSeconds.Observe(seconds)
}
