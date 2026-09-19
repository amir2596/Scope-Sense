// cmd/api is the long-running service: it exposes the pricing
// pipeline over HTTP so a real client (e.g. the Flutter app) can call
// it. This is deliberately a separate program from cmd/seed and
// cmd/evaluate — "always running, serving users" is a different job
// from "run once, do a task."
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"scopesense/internal/agent"
	"scopesense/internal/embedding"
	"scopesense/internal/explainer"
	"scopesense/internal/pricing"
	"scopesense/internal/vectorstore"
)

// evaluateTimeout bounds how long one request is allowed to run end
// to end (clarify + embed + search). Without this, a hung Ollama call
// would block a request (and its goroutine) forever instead of
// failing cleanly — the exact concurrency-correctness gap we flagged
// earlier.
//
// Set generously (90s) for CPU-only local development, where a
// model's first call after Ollama/the model isn't already loaded in
// memory can itself take 15-20+ seconds before generation even
// starts. Lower this (e.g. back to 30s) for a GPU-backed deployment,
// or once you're consistently warming models before testing (see
// README) and want tighter failure behavior.
const evaluateTimeout = 90 * time.Second

// maxConcurrentEvaluations bounds how many requests hit the local
// models/database at once. This is a plain Go semaphore: a buffered
// channel used purely for its capacity, not for the values it holds.
// Without this, a burst of traffic would fire unlimited concurrent
// calls at a single local Ollama instance, which has no queueing of
// its own and would degrade badly under load.
const maxConcurrentEvaluations = 4

type server struct {
	svc *pricing.Service
	sem chan struct{}
}

type evaluateRequest struct {
	Description string  `json:"description"`
	Budget      float64 `json:"budget"`
}

func (s *server) handleEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req evaluateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Description == "" {
		http.Error(w, "description is required", http.StatusBadRequest)
		return
	}

	log.Printf("evaluate request: budget=%.0f description=%q", req.Budget, truncateForLog(req.Description, 200))

	// Bounded concurrency: block here (not reject) if we're already at
	// capacity, up to the request's own timeout. A simple, honest
	// choice for a portfolio project; a production system under real
	// load might instead reject immediately with 503 and let a client
	// retry, rather than making the caller wait in a queue.
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-r.Context().Done():
		http.Error(w, "request cancelled while waiting for capacity", http.StatusServiceUnavailable)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), evaluateTimeout)
	defer cancel()

	result, err := s.svc.Evaluate(ctx, req.Description, req.Budget)
	if err != nil {
		log.Printf("evaluate error: %v", err)
		http.Error(w, "evaluation failed", http.StatusInternalServerError)
		return
	}

	// Log the full response server-side, independent of what the
	// client actually receives — useful for debugging the pipeline
	// (clarifier output, verdict, comparables) without needing to
	// inspect the Flutter app or add a debugger.
	if resultJSON, err := json.Marshal(result); err != nil {
		log.Printf("marshal result for logging: %v", err)
	} else {
		log.Printf("evaluate response: %s", resultJSON)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Printf("encode response error: %v", err)
	}
}

// truncateForLog cuts s to at most n runes, with a marker if it was
// cut, so a very long client description doesn't flood the log.
func truncateForLog(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "...(truncated)"
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

func main() {
	ctx := context.Background()

	pool, err := pgxpool.New(ctx, mustGetenv("DATABASE_URL"))
	if err != nil {
		log.Fatalf("connect to postgres: %v", err)
	}
	defer pool.Close()

	svc := &pricing.Service{
		Clarifier:   agent.NewOllamaClarifier("http://localhost:11434", "qwen2.5:7b-instruct"),
		Embedder:    embedding.NewOllamaEmbedder("http://localhost:11434", "bge-m3"),
		VectorStore: vectorstore.NewPgvectorStore(pool),
		Explainer:   explainer.NewOllamaExplainer("http://localhost:11434", "qwen2.5:7b-instruct"),
		TopK:        5,
	}

	srv := &server{
		svc: svc,
		sem: make(chan struct{}, maxConcurrentEvaluations),
	}

	warmUpModels(ctx, svc)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/evaluate", srv.handleEvaluate)
	mux.HandleFunc("/health", handleHealth)

	addr := ":8080"
	log.Printf("listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

// warmUpModels sends one throwaway request to each Ollama model
// before the server starts accepting real traffic. Without this, the
// first real request after Ollama (or the machine) restarts — or
// after either model sits idle past Ollama's default 5-minute
// keep-alive — pays the full cold-load cost (can be 30-60+ seconds
// on CPU) inline, which is exactly the slow first-request behavior
// we kept hitting during testing. Startup blocking on this is a
// deliberate tradeoff: a slightly slower `go run` is preferable to a
// demo/recruiter's first click taking a minute.
//
// Errors here are logged, not fatal — if Ollama isn't reachable yet,
// real requests will surface a clearer error at call time anyway.
func warmUpModels(ctx context.Context, svc *pricing.Service) {
	warmCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	log.Print("warming up clarifier model...")
	start := time.Now()
	if _, err := svc.Clarifier.Clarify(warmCtx, "test"); err != nil {
		log.Printf("clarifier warm-up failed (will retry on first real request): %v", err)
	} else {
		log.Printf("clarifier model warm (%s)", time.Since(start).Round(time.Second))
	}

	log.Print("warming up embedding model...")
	start = time.Now()
	if _, err := svc.Embedder.Embed(warmCtx, "test"); err != nil {
		log.Printf("embedder warm-up failed (will retry on first real request): %v", err)
	} else {
		log.Printf("embedding model warm (%s)", time.Since(start).Round(time.Second))
	}
	// Explainer uses the same underlying model as the clarifier
	// (qwen2.5:7b-instruct), so warming the clarifier already warms
	// it too — no separate call needed.
}

func mustGetenv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("missing required environment variable: %s", key)
	}
	return v
}
