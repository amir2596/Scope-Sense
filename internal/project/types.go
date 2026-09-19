package project

import "context"

// VectorStore is implemented by whatever database actually stores
// embeddings. internal/vectorstore/pgvector.go is the current
// implementation, but nothing outside that file needs to know that.
type VectorStore interface {
	Upsert(ctx context.Context, id string, embedding []float32, metadata map[string]any) error
	Search(ctx context.Context, queryEmbedding []float32, topK int, filter map[string]any) ([]SearchResult, error)
	Delete(ctx context.Context, id string) error
}

type SearchResult struct {
	ID       string
	Score    float32
	Metadata map[string]any
}

// Embedder is implemented by whatever turns text into a vector.
// internal/embedding/openai.go is the current implementation.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// Project mirrors one row of seed_projects.json / the projects table.
type Project struct {
	ID                  string   `json:"id"`
	Title               string   `json:"title"`
	RawDescription      string   `json:"raw_description"`
	ClientStatedBudget  float64  `json:"client_stated_budget"`
	Currency            string   `json:"currency"`
	Scope               string   `json:"scope"`
	TechStack           []string `json:"tech_stack"`
	Deliverables        []string `json:"deliverables"`
	Category            string   `json:"category"`
	AgreedPrice         float64  `json:"agreed_price"`
	PriceStatus         string   `json:"price_status"`
	DurationDays        int      `json:"duration_days"`
}

// EmbeddingText is what actually gets embedded. Using the structured
// fields (not the raw description) gives a cleaner signal, per our
// earlier design decision.
func (p Project) EmbeddingText() string {
	return p.Title + ". " + p.Scope
}

// Metadata is what gets stored alongside the vector, and is what
// Search() filters/returns.
func (p Project) Metadata() map[string]any {
	return map[string]any{
		"title":                 p.Title,
		"client_stated_budget":  p.ClientStatedBudget,
		"currency":              p.Currency,
		"agreed_price":          p.AgreedPrice,
		"price_status":          p.PriceStatus,
		"category":              p.Category,
	}
}
