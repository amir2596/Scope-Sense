package embedding

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"

	"scopesense/internal/project"
)

type OpenAIEmbedder struct {
	client *openai.Client
	model  openai.EmbeddingModel
}

func NewOpenAIEmbedder(apiKey string) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		client: openai.NewClient(apiKey),
		model:  openai.SmallEmbedding3, // "text-embedding-3-small", 1536 dimensions
	}
}

// compile-time check that OpenAIEmbedder actually satisfies the interface
var _ project.Embedder = (*OpenAIEmbedder)(nil)

func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{text},
		Model: e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("create embedding: %w", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("openai returned no embedding data")
	}
	return resp.Data[0].Embedding, nil
}
