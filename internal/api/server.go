// Package api hosts the HTTP server, middleware stack, and request
// handlers. The server uses chi for routing because middleware
// composition (RequestID, Logger, Recoverer, Timeout, BodyLimit) is the
// shape we need and stdlib mux doesn't provide.
package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ReadyFunc reports whether the daemon is fully ready to serve traffic.
// /readyz invokes it with a short timeout, returning a non-nil error
// renders a 503. ReadyFunc may be nil (then /readyz always reports OK).
type ReadyFunc func(ctx context.Context) error

// ServerConfig is the input to NewServer. All durations have sensible
// defaults, the only required field is ListenAddress.
type ServerConfig struct {
	ListenAddress     string
	ReadTimeout       time.Duration
	ReadHeaderTimeout time.Duration
	WriteTimeout      time.Duration
	IdleTimeout       time.Duration

	// MaxRequestBytes caps incoming JSON body size. 0 disables.
	MaxRequestBytes int64

	// Ready is called by /readyz. May be nil.
	Ready ReadyFunc

	// Logger receives request and lifecycle events.
	Logger *slog.Logger

	// PromRegistry, if non-nil, mounts /metrics using a handler that
	// scrapes this registry.
	PromRegistry prometheus.Gatherer

	// AuditMetrics, if non-nil, counts control-plane mutations.
	AuditMetrics ControlPlaneAuditMetrics
}

// Server wraps an *http.Server and the chi router. Construct with
// NewServer and run with Start/Shutdown.
type Server struct {
	httpServer *http.Server
	router     chi.Router
	logger     *slog.Logger
}

// NewServer assembles the middleware stack and registers the health
// endpoints. Feature routes are added by callers via Router() before
// Start is called.
func NewServer(cfg ServerConfig) *Server {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(loggingMiddleware(cfg.Logger, cfg.AuditMetrics))
	r.Use(middleware.Recoverer)
	r.Use(requireJSONMiddleware(r, cfg.AuditMetrics))
	if cfg.MaxRequestBytes > 0 {
		r.Use(bodyLimitMiddleware(cfg.MaxRequestBytes))
	}

	registerHealthRoutes(r, cfg.Ready)
	if cfg.PromRegistry != nil {
		r.Handle("/metrics", promhttp.HandlerFor(cfg.PromRegistry, promhttp.HandlerOpts{}))
	}

	return &Server{
		router: r,
		logger: cfg.Logger,
		httpServer: &http.Server{
			Addr:              cfg.ListenAddress,
			Handler:           r,
			ReadTimeout:       cfg.ReadTimeout,
			ReadHeaderTimeout: cfg.ReadHeaderTimeout,
			WriteTimeout:      cfg.WriteTimeout,
			IdleTimeout:       cfg.IdleTimeout,
		},
	}
}

// Router exposes the underlying chi.Router so feature packages can
// register their routes before Start.
func (s *Server) Router() chi.Router { return s.router }

// Addr returns the resolved listen address (useful in tests that use
// :0 to pick an ephemeral port, only valid after Start).
func (s *Server) Addr() string { return s.httpServer.Addr }

// Start blocks until the HTTP server stops. Returns nil when stopped
// via Shutdown, otherwise the underlying error.
func (s *Server) Start() error {
	s.logger.Info("http server listening", "address", s.httpServer.Addr)
	if err := s.httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("http server: %w", err)
	}
	return nil
}

// Shutdown gracefully drains in-flight requests, bounded by ctx.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func bodyLimitMiddleware(limit int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.Body = http.MaxBytesReader(w, r.Body, limit)
			next.ServeHTTP(w, r)
		})
	}
}

// requireJSONMiddleware returns 415 for any POST, PUT, or PATCH whose
// Content-Type is not application/json. GET, HEAD, DELETE, and every
// other method pass through unchecked. A rejection is still audited,
// using router.Find to resolve the route pattern without dispatching.
func requireJSONMiddleware(router chi.Router, audit ControlPlaneAuditMetrics) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.Method {
			case http.MethodPost, http.MethodPut, http.MethodPatch:
			default:
				next.ServeHTTP(w, r)
				return
			}
			mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
			if err != nil || mediaType != "application/json" {
				if audit != nil {
					// A fresh scratch context, never the request's own:
					// Find mutates it in place, and the request's shared
					// context is still read by loggingMiddleware below,
					// which would otherwise see this speculative lookup's
					// leftover state and double-count the mutation.
					pattern := router.Find(chi.NewRouteContext(), r.Method, routingPath(r))
					auditControlPlaneMutation(audit, r, pattern, http.StatusUnsupportedMediaType)
				}
				writeErrorMsg(w, http.StatusUnsupportedMediaType, "content-type must be application/json")
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// routingPath returns the path chi's own dispatcher would route on,
// preferring RawPath the way routeHTTP does. A percent-encoded
// segment (e.g. a literal "%2F") decodes differently in Path than in
// RawPath, so resolving a route pattern for an unrouted request must
// walk the same string real dispatch would or it can match a
// different route entirely, or none at all.
func routingPath(r *http.Request) string {
	if r.URL.RawPath != "" {
		return r.URL.RawPath
	}
	if r.URL.Path == "" {
		return "/"
	}
	return r.URL.Path
}

func loggingMiddleware(logger *slog.Logger, audit ControlPlaneAuditMetrics) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			reqID := middleware.GetReqID(r.Context())
			if reqID != "" {
				// Echo to clients so they can quote it in bug reports.
				w.Header().Set("X-Request-Id", reqID)
			}
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)

			// Health endpoints are noisy on prod log aggregators.
			level := slog.LevelInfo
			if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
				level = slog.LevelDebug
			}
			logger.LogAttrs(r.Context(), level, "http request",
				slog.String("method", r.Method),
				slog.String("path", r.URL.Path),
				slog.Int("status", ww.Status()),
				slog.Int("bytes", ww.BytesWritten()),
				slog.Duration("duration", time.Since(start)),
				slog.String("request_id", reqID),
				slog.String("remote_addr", r.RemoteAddr),
				slog.String("forwarded_for", r.Header.Get("X-Forwarded-For")),
			)

			// Mounted outside Recoverer, so a panicking handler is
			// audited with the 500 the recoverer turned it into.
			if audit != nil {
				pattern := chi.RouteContext(r.Context()).RoutePattern()
				auditControlPlaneMutation(audit, r, pattern, ww.Status())
			}
		})
	}
}

// ControlPlaneAuditMetrics counts mutating requests to the control
// plane.
type ControlPlaneAuditMetrics interface {
	IncControlPlaneMutation(resource, method, outcome string)
}

// controlPlaneResource returns the resource segment of a matched
// control-plane route pattern and reports whether pattern is one.
// Taking the segment from the matched pattern bounds the metric's
// label set by the routes orla registers, since an unrouted request
// carries no pattern.
func controlPlaneResource(pattern string) (string, bool) {
	rest, ok := strings.CutPrefix(pattern, "/api/v1/")
	if !ok {
		return "", false
	}
	resource, _, _ := strings.Cut(rest, "/")
	return resource, resource != ""
}

// auditControlPlaneMutation counts a write to the control plane by
// method and route pattern, ignoring a GET or HEAD. The pattern may
// be one the request would have matched rather than one it did, for
// a rejection recorded ahead of routing.
func auditControlPlaneMutation(metrics ControlPlaneAuditMetrics, r *http.Request, pattern string, status int) {
	if r.Method == http.MethodGet || r.Method == http.MethodHead {
		return
	}
	resource, ok := controlPlaneResource(pattern)
	if !ok {
		return
	}

	outcome := "success"
	if status < 200 || status >= 300 {
		outcome = "error"
	}
	metrics.IncControlPlaneMutation(resource, r.Method, outcome)
}
