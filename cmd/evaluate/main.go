// cmd/evaluate is a small test harness: paste a client description and
// a stated budget, see the full pipeline run (clarify -> embed ->
// retrieve -> verdict).
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"

	"scopesense/internal/agent"
	"scopesense/internal/embedding"
	"scopesense/internal/explainer"
	"scopesense/internal/pricing"
	"scopesense/internal/vectorstore"
)

func main() {
	description := flag.String("description", "", "client's raw project description")
	budget := flag.Float64("budget", 0, "client's stated budget in USD")
	flag.Parse()

	if *description == "" {
		log.Fatal("usage: go run ./cmd/evaluate -description \"...\" -budget 100")
	}

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

	result, err := svc.Evaluate(ctx, *description, *budget)
	if err != nil {
		log.Fatalf("evaluate: %v", err)
	}

	fmt.Println("--- Structured spec ---")
	fmt.Printf("Category: %s\nScope: %s\nTech stack: %v\nDeliverables: %v\n",
		result.Spec.Category, result.Spec.Scope, result.Spec.TechStack, result.Spec.Deliverables)
	if len(result.Spec.ClarifyingQuestions) > 0 {
		fmt.Println("Clarifying questions:")
		for _, q := range result.Spec.ClarifyingQuestions {
			fmt.Println(" -", q)
		}
	}

	fmt.Println("\n--- Price comparison ---")
	fmt.Printf("Client stated budget: %.0f\n", result.ClientStatedBudget)
	fmt.Printf("Verdict: %s\n", result.Verdict)
	if result.Explanation != "" {
		fmt.Printf("Explanation: %s\n", result.Explanation)
	}
	if result.Verdict != "no_data" {
		fmt.Printf("Similar projects ranged %.0f - %.0f (median %.0f)\n",
			result.MinPrice, result.MaxPrice, result.MedianPrice)
		fmt.Println("Comparable projects:")
		for _, c := range result.Comparables {
			fmt.Printf(" - %s: %.0f %s\n", c.Title, c.AgreedPrice, c.Currency)
		}
	}
}

func mustGetenv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("missing required environment variable: %s", key)
	}
	return v
}
