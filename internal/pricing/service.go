package pricing

import (
	"context"
	"fmt"
	"sort"

	"scopesense/internal/agent"
	"scopesense/internal/project"
)

// Comparable is one retrieved past project used to justify a verdict.
type Comparable struct {
	Title       string  `json:"title"`
	AgreedPrice float64 `json:"agreed_price"`
	Currency    string  `json:"currency"`
}

// Result is the full output for a client's submission: the structured
// spec extracted from their description, the price range found from
// similar past projects, and a plain-language verdict.
type Result struct {
	Spec               *agent.ClarificationResult `json:"spec"`
	ClientStatedBudget float64                    `json:"client_stated_budget"`
	MinPrice           float64                    `json:"min_price"`
	MedianPrice        float64                    `json:"median_price"`
	MaxPrice           float64                    `json:"max_price"`
	Comparables        []Comparable               `json:"comparables"`
	Verdict            string                     `json:"verdict"`               // "realistic" | "underpriced" | "overpriced" | "no_data"
	Explanation        string                     `json:"explanation,omitempty"` // written justification, if an Explainer is configured
}

// Explainer turns a computed Result into a short written explanation
// a real client would read. Defined here in pricing (the consumer),
// not in the package that implements it — same pattern as Embedder
// and VectorStore. This also avoids a circular import: the concrete
// Ollama implementation lives in internal/explainer, which imports
// pricing for these types, rather than pricing importing it back.
type Explainer interface {
	Explain(ctx context.Context, r *Result) (string, error)
}

// Service wires the three core interfaces together, plus an optional
// Explainer. It has no idea whether any of them are backed by Ollama,
// OpenAI, Postgres, or anything else.
type Service struct {
	Clarifier   agent.ClarificationAgent
	Embedder    project.Embedder
	VectorStore project.VectorStore
	Explainer   Explainer // optional — nil means "skip the explanation step"
	TopK        int       // how many comparable projects to retrieve, e.g. 5
}

func (s *Service) Evaluate(ctx context.Context, rawDescription string, clientStatedBudget float64) (*Result, error) {
	// Step 1: turn the vague description into a structured spec.
	spec, err := s.Clarifier.Clarify(ctx, rawDescription)
	if err != nil {
		return nil, fmt.Errorf("clarify description: %w", err)
	}

	// Step 2: embed the structured spec (not the raw text) — same
	// choice we made for the seed data, for a cleaner, more comparable
	// signal.
	embeddingText := spec.Scope + ". Category: " + spec.Category
	vec, err := s.Embedder.Embed(ctx, embeddingText)
	if err != nil {
		return nil, fmt.Errorf("embed spec: %w", err)
	}

	// Step 3: retrieve similar past projects. Prefer real closed deals
	// over rough estimates, and prefer the same project category, per
	// our category-mismatch finding from testing.
	results, err := s.VectorStore.Search(ctx, vec, s.TopK, map[string]any{
		"price_status": "closed",
		"category":     spec.Category,
	})
	if err != nil {
		return nil, fmt.Errorf("search comparables: %w", err)
	}
	// Fall back to an unfiltered-by-category search if the strict
	// filter found nothing — better to compare against a looser match
	// than to return no data at all.
	if len(results) == 0 {
		results, err = s.VectorStore.Search(ctx, vec, s.TopK, map[string]any{
			"price_status": "closed",
		})
		if err != nil {
			return nil, fmt.Errorf("search comparables (fallback): %w", err)
		}
	}

	result := &Result{
		Spec:               spec,
		ClientStatedBudget: clientStatedBudget,
	}

	if len(results) == 0 {
		result.Verdict = "no_data"
		return result, nil
	}

	// Step 4: pure math — no LLM needed for this part, exactly the
	// distinction we discussed earlier.
	prices := make([]float64, 0, len(results))
	for _, r := range results {
		price, _ := r.Metadata["agreed_price"].(float64)
		title, _ := r.Metadata["title"].(string)
		currency, _ := r.Metadata["currency"].(string)

		result.Comparables = append(result.Comparables, Comparable{
			Title:       title,
			AgreedPrice: price,
			Currency:    currency,
		})
		prices = append(prices, price)
	}
	sort.Float64s(prices)
	result.MinPrice = prices[0]
	result.MaxPrice = prices[len(prices)-1]
	result.MedianPrice = median(prices)

	// Step 5: verdict — simple rule-based math, no LLM needed here.
	switch {
	case clientStatedBudget < result.MinPrice*0.7:
		result.Verdict = "underpriced"
	case clientStatedBudget > result.MaxPrice*1.3:
		result.Verdict = "overpriced"
	default:
		result.Verdict = "realistic"
	}

	// Step 6 (optional): generate a written explanation. This is a
	// deliberately separate, optional step — if it fails, the caller
	// still gets a fully usable numeric result. A missing explanation
	// is a lesser failure than a missing price verdict, so we log and
	// continue rather than returning an error here.
	if s.Explainer != nil {
		explanation, err := s.Explainer.Explain(ctx, result)
		if err != nil {
			result.Explanation = ""
		} else {
			result.Explanation = explanation
		}
	}

	return result, nil
}

func median(sorted []float64) float64 {
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}
