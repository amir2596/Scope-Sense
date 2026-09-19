package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"

	"scopesense/internal/project"
)

// PgvectorStore implements project.VectorStore backed by Postgres + the
// pgvector extension. It's the only file in the codebase that knows
// Postgres exists — everything else talks to the project.VectorStore
// interface instead.
type PgvectorStore struct {
	pool *pgxpool.Pool
}

func NewPgvectorStore(pool *pgxpool.Pool) *PgvectorStore {
	return &PgvectorStore{pool: pool}
}

// compile-time check that PgvectorStore actually satisfies the interface
var _ project.VectorStore = (*PgvectorStore)(nil)

func (s *PgvectorStore) Upsert(ctx context.Context, id string, embedding []float32, metadata map[string]any) error {
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("marshal metadata: %w", err)
	}

	_, err = s.pool.Exec(ctx, `
		INSERT INTO projects (id, embedding, metadata)
		VALUES ($1, $2, $3)
		ON CONFLICT (id) DO UPDATE
		SET embedding = EXCLUDED.embedding,
		    metadata  = EXCLUDED.metadata
	`, id, pgvector.NewVector(embedding), metaJSON)
	if err != nil {
		return fmt.Errorf("upsert project %s: %w", id, err)
	}
	return nil
}

func (s *PgvectorStore) Search(ctx context.Context, queryEmbedding []float32, topK int, filter map[string]any) ([]project.SearchResult, error) {
	// price_status and category are the two filters this project
	// needs today. Passing nil for either means "no filter" on that
	// field.
	var priceStatus *string
	if v, ok := filter["price_status"].(string); ok {
		priceStatus = &v
	}
	var category *string
	if v, ok := filter["category"].(string); ok {
		category = &v
	}

	rows, err := s.pool.Query(ctx, `
		SELECT id, metadata, embedding <=> $1 AS distance
		FROM projects
		WHERE ($2::text IS NULL OR metadata->>'price_status' = $2)
		  AND ($3::text IS NULL OR metadata->>'category' = $3)
		ORDER BY embedding <=> $1
		LIMIT $4
	`, pgvector.NewVector(queryEmbedding), priceStatus, category, topK)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	defer rows.Close()

	var results []project.SearchResult
	for rows.Next() {
		var (
			id       string
			metaJSON []byte
			distance float32
		)
		if err := rows.Scan(&id, &metaJSON, &distance); err != nil {
			return nil, fmt.Errorf("scan search row: %w", err)
		}

		var metadata map[string]any
		if err := json.Unmarshal(metaJSON, &metadata); err != nil {
			return nil, fmt.Errorf("unmarshal metadata for %s: %w", id, err)
		}

		results = append(results, project.SearchResult{
			ID:       id,
			Score:    distance,
			Metadata: metadata,
		})
	}
	return results, rows.Err()
}

func (s *PgvectorStore) Delete(ctx context.Context, id string) error {
	_, err := s.pool.Exec(ctx, `DELETE FROM projects WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete project %s: %w", id, err)
	}
	return nil
}
