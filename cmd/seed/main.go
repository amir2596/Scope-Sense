// cmd/seed is a one-off/occasional tool: run it to (re)populate the
// database with the seed dataset. It is NOT the long-running API
// server — that's cmd/api, kept deliberately separate.
package main

import (
	"context"
	"encoding/json"
	"log"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"

	"scopesense/internal/embedding"
	"scopesense/internal/project"
	"scopesense/internal/vectorstore"
)

func main() {
	ctx := context.Background()

	dbURL := mustGetenv("DATABASE_URL")

	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Fatalf("connect to postgres: %v", err)
	}
	defer pool.Close()

	// Ollama runs locally — no API key, no billing, no geo-restriction.
	// bge-m3 (not nomic-embed-text) because it's multilingual, which
	// matters since client descriptions arrive in both English and
	// Persian. Start Ollama first with: ollama serve
	embedder := embedding.NewOllamaEmbedder("http://localhost:11434", "bge-m3")
	store := vectorstore.NewPgvectorStore(pool)

	projects, err := loadProjects("seed_projects.json")
	if err != nil {
		log.Fatalf("load seed data: %v", err)
	}

	for i, p := range projects {
		vec, err := embedder.Embed(ctx, p.EmbeddingText())
		if err != nil {
			log.Fatalf("embed project %s: %v", p.ID, err)
		}

		if err := store.Upsert(ctx, p.ID, vec, p.Metadata()); err != nil {
			log.Fatalf("upsert project %s: %v", p.ID, err)
		}

		log.Printf("seeded %d/%d: %s", i+1, len(projects), p.Title)
	}

	log.Printf("done: seeded %d projects", len(projects))
}

func loadProjects(path string) ([]project.Project, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var projects []project.Project
	if err := json.Unmarshal(data, &projects); err != nil {
		return nil, err
	}
	return projects, nil
}

func mustGetenv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("missing required environment variable: %s", key)
	}
	return v
}
