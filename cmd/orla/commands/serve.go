package commands

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"

	"github.com/harvard-cns/orla/internal/api"
	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/config"
	"github.com/harvard-cns/orla/internal/costs"
	"github.com/harvard-cns/orla/internal/mappings"
	"github.com/harvard-cns/orla/internal/metrics"
	"github.com/harvard-cns/orla/internal/provider"
	"github.com/harvard-cns/orla/internal/scheduler"
	"github.com/harvard-cns/orla/internal/settings"
	"github.com/harvard-cns/orla/internal/stages"
	"github.com/harvard-cns/orla/internal/storage"
	"github.com/harvard-cns/orla/internal/telemetry"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Run the orla daemon",
	RunE:  runServe,
}

func runServe(cmd *cobra.Command, _ []string) error {
	ctx, stop := signal.NotifyContext(cmd.Context(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	cfg, err := config.Load(ctx)
	if err != nil {
		return err
	}

	logger := newLogger(cfg.LogFormat, cfg.LogLevel)
	slog.SetDefault(logger)

	logger.Info("orla starting",
		"listen_address", cfg.ListenAddress,
		"log_format", cfg.LogFormat,
		"log_level", cfg.LogLevel,
	)

	store, err := storage.Open(ctx, storage.OpenConfig{
		DatabaseURL:     cfg.DatabaseURL,
		MaxOpenConns:    cfg.DBMaxOpenConns,
		ConnMaxLifetime: cfg.DBConnMaxLifetime,
		Logger:          logger,
	})
	if err != nil {
		return err
	}
	defer store.Close()

	stageRegistry := stages.NewPostgresRegistry(store.Pool())
	backendRegistry := backends.NewPostgresRegistry(store.Pool())
	mappingRegistry := mappings.NewPostgresRegistry(store.Pool())

	promReg := prometheus.NewRegistry()
	promReg.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)
	m := metrics.New(promReg)

	// Provider factory. LLM backends use the OpenAI provider. No tool
	// providers are registered, so a tool backend resolves to the OpenAI
	// provider, which the tool dispatch path rejects at AcquireTool. Add
	// a KindTool case here when a ToolProvider implementation lands.
	factory := func(b *backends.Backend) provider.Backend {
		return provider.NewOpenAI(b)
	}
	sched := scheduler.New(factory, logger, scheduler.WithPolicyMetrics(m))

	// Restore the scheduling policy last set through orlactl.
	policyStore := settings.NewPostgresStore(store.Pool())
	policyCfg, err := policyStore.Get(ctx)
	if err != nil {
		return fmt.Errorf("load scheduler policy: %w", err)
	}
	api.ApplySchedulerPolicy(sched, policyCfg)
	if policyCfg.Enabled() {
		logger.Info("scheduling policy enabled", "url", policyCfg.URL, "timeout", policyCfg.Timeout)
	}

	// Rehydrate scheduler with existing backends from the DB.
	existing, err := backendRegistry.List(ctx)
	if err != nil {
		return err
	}
	for _, b := range existing {
		sched.Register(b)
	}
	promReg.MustRegister(metrics.NewSchedulerCollector(sched))

	srv := api.NewServer(api.ServerConfig{
		ListenAddress:     cfg.ListenAddress,
		ReadTimeout:       cfg.ReadTimeout,
		ReadHeaderTimeout: cfg.ReadHeaderTimeout,
		WriteTimeout:      cfg.WriteTimeout,
		IdleTimeout:       cfg.IdleTimeout,
		MaxRequestBytes:   cfg.MaxRequestBytes,
		Logger:            logger,
		Ready:             store.Ping,
		PromRegistry:      promReg,
		AuditMetrics:      m,
		BodyLimitMetrics:  m,
	})
	api.RegisterStageRoutes(srv.Router(), stageRegistry)
	api.RegisterMappingRoutes(srv.Router(), mappingRegistry)
	api.RegisterSchedulerRoutes(srv.Router(), api.SchedulerDeps{
		Store:     policyStore,
		Scheduler: sched,
	})
	api.RegisterBackendRoutes(srv.Router(), api.BackendDeps{
		Registry:  backendRegistry,
		Lifecycle: sched,
		Manager:   sched,
	})
	costPolicyStore := settings.NewPostgresCostStore(store.Pool())
	api.RegisterCostRoutes(srv.Router(), api.CostDeps{Store: costPolicyStore})

	// Restore the stage mapper last set through orlactl.
	stageMapperStore := settings.NewPostgresMapperStore(store.Pool())
	stageMapperHolder := &mappings.MapperHolder{}
	mapperCfg, err := stageMapperStore.Get(ctx)
	if err != nil {
		return fmt.Errorf("load stage mapper: %w", err)
	}
	api.ApplyStageMapper(stageMapperHolder, mapperCfg)
	if mapperCfg.Enabled() {
		logger.Info("stage mapper enabled", "url", mapperCfg.URL, "timeout", mapperCfg.Timeout)
	}
	api.RegisterStageMapperRoutes(srv.Router(), api.StageMapperDeps{
		Store:  stageMapperStore,
		Holder: stageMapperHolder,
	})
	completionWriter := telemetry.NewCompletionWriter(telemetry.CompletionWriterConfig{
		Pool:   store.Pool(),
		Logger: logger,
	})
	feedbackWriter := telemetry.NewFeedbackWriter(telemetry.FeedbackWriterConfig{
		Pool:   store.Pool(),
		Logger: logger,
	})
	promReg.MustRegister(metrics.NewBatchWriterCollector(map[string]metrics.BatchWriterStats{
		"completion_records": completionWriter,
		"feedback":           feedbackWriter,
	}))
	promReg.MustRegister(metrics.NewCompletionIOCollector(completionWriter))

	costStore := costs.NewStore()
	costPoller := costs.NewPoller(costs.PollerConfig{
		Registry: backendRegistry,
		Store:    costStore,
		Policy:   costPolicyStore,
		Metrics:  m,
		Logger:   logger,
	})
	costPoller.Start()
	promReg.MustRegister(metrics.NewCostCollector(costStore))

	api.RegisterProxyRoutes(srv.Router(), api.ProxyDeps{
		Stages:         stageRegistry,
		Mappings:       mappingRegistry,
		Scheduler:      sched,
		CompletionSink: completionWriter,
		Metrics:        m,
		Costs:          costStore,
		StageMapper:    stageMapperHolder,
	})
	api.RegisterFeedbackRoutes(srv.Router(), api.FeedbackDeps{
		Sink:    feedbackWriter,
		Metrics: m,
	})
	api.RegisterMapperRoutes(srv.Router(), api.MapperDeps{
		Reader: telemetry.NewReader(store.Pool()),
	})
	api.RegisterToolRoutes(srv.Router(), api.ToolDeps{
		Stages:         stageRegistry,
		Scheduler:      sched,
		Backends:       backendRegistry,
		CompletionSink: completionWriter,
		Metrics:        m,
	})

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Start() }()

	select {
	case <-ctx.Done():
		logger.Info("shutdown signal received")
	case err := <-errCh:
		if err != nil {
			logger.Error("http server stopped unexpectedly", "error", err)
			return err
		}
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		logger.Error("graceful shutdown error", "error", err)
	}
	// Drain the start goroutine if it hasn't already returned.
	if startErr := <-errCh; startErr != nil {
		logger.Error("http server final error", "error", startErr)
	}
	if err := sched.Shutdown(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		logger.Error("scheduler shutdown error", "error", err)
	}
	if err := costPoller.Close(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		logger.Error("cost poller shutdown error", "error", err)
	}
	if err := completionWriter.Close(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		logger.Error("completion writer shutdown error", "error", err)
	}
	if err := feedbackWriter.Close(shutdownCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		logger.Error("feedback writer shutdown error", "error", err)
	}

	logger.Info("orla stopped")
	return nil
}

// newLogger constructs a slog.Logger writing to stderr. format is
// either "text" or "json". level is one of "debug", "info", "warn",
// or "error" and is matched case-insensitively. Unknown values fall
// back to text format at info level.
func newLogger(format, level string) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "debug":
		lvl = slog.LevelDebug
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error", "err":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lvl}
	var h slog.Handler
	if strings.EqualFold(format, "json") {
		h = slog.NewJSONHandler(os.Stderr, opts)
	} else {
		h = slog.NewTextHandler(os.Stderr, opts)
	}
	return slog.New(h)
}
