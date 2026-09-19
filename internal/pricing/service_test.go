package pricing

import (
	"context"
	"errors"
	"testing"

	"scopesense/internal/agent"
	"scopesense/internal/project"
)

// --- Fakes ---
//
// Each fake is a minimal, hand-written implementation of one of the
// interfaces Service depends on. They exist only for tests: no
// network calls, no real model, no real database — just whatever
// canned behavior each test case needs. This is only possible because
// Service depends on interfaces, not on OllamaClarifier/PgvectorStore
// directly.

// fakeClarifier always returns the same spec, or an error if one is set.
type fakeClarifier struct {
	spec *agent.ClarificationResult
	err  error
}

func (f *fakeClarifier) Clarify(ctx context.Context, rawDescription string) (*agent.ClarificationResult, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.spec, nil
}

// fakeEmbedder returns a fixed vector regardless of input text.
type fakeEmbedder struct {
	err error
}

func (f *fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	return []float32{0.1, 0.2, 0.3}, nil
}

// fakeVectorStore returns a fixed, pre-configured set of search
// results — this is how each test controls what "past projects" the
// pricing logic sees, without needing a real database.
type fakeVectorStore struct {
	results []project.SearchResult
	err     error
}

func (f *fakeVectorStore) Upsert(ctx context.Context, id string, embedding []float32, metadata map[string]any) error {
	return nil // unused by Evaluate; present only to satisfy the interface
}

func (f *fakeVectorStore) Search(ctx context.Context, queryEmbedding []float32, topK int, filter map[string]any) ([]project.SearchResult, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.results, nil
}

func (f *fakeVectorStore) Delete(ctx context.Context, id string) error {
	return nil // unused by Evaluate
}

// fakeExplainer returns a fixed sentence, or an error if configured to.
type fakeExplainer struct {
	text string
	err  error
}

func (f *fakeExplainer) Explain(ctx context.Context, r *Result) (string, error) {
	if f.err != nil {
		return "", f.err
	}
	return f.text, nil
}

// comparable builds one fake past-project search result, matching the
// shape Metadata() produces in internal/project/types.go.
func comparable(title string, price float64) project.SearchResult {
	return project.SearchResult{
		ID: title,
		Metadata: map[string]any{
			"title":        title,
			"agreed_price": price,
			"currency":     "USD",
			"price_status": "closed",
		},
	}
}

// --- Tests ---

func TestEvaluate_Verdicts(t *testing.T) {
	// Table-driven: each case is one row describing inputs and the
	// expected verdict, run through the same test body. This is the
	// standard Go pattern for testing several related scenarios
	// without duplicating setup code for each one.
	cases := []struct {
		name           string
		budget         float64
		comparables    []project.SearchResult
		wantVerdict    string
		wantComparable int // expected len(result.Comparables)
	}{
		{
			name:   "clearly underpriced",
			budget: 40,
			comparables: []project.SearchResult{
				comparable("salon app", 1200),
				comparable("gym app", 1800),
			},
			wantVerdict:    "underpriced",
			wantComparable: 2,
		},
		{
			name:   "clearly overpriced",
			budget: 50000,
			comparables: []project.SearchResult{
				comparable("salon app", 1200),
				comparable("gym app", 1800),
			},
			wantVerdict:    "overpriced",
			wantComparable: 2,
		},
		{
			name:   "realistic, within range",
			budget: 1500,
			comparables: []project.SearchResult{
				comparable("salon app", 1200),
				comparable("gym app", 1800),
			},
			wantVerdict:    "realistic",
			wantComparable: 2,
		},
		{
			name:           "no comparables found",
			budget:         500,
			comparables:    []project.SearchResult{},
			wantVerdict:    "no_data",
			wantComparable: 0,
		},
	}

	for _, tc := range cases {
		// t.Run creates a named subtest: failures report which case
		// failed by name, and cases run independently of each other.
		t.Run(tc.name, func(t *testing.T) {
			svc := &Service{
				Clarifier: &fakeClarifier{spec: &agent.ClarificationResult{
					Category: "mobile-app",
					Scope:    "a test project",
				}},
				Embedder:    &fakeEmbedder{},
				VectorStore: &fakeVectorStore{results: tc.comparables},
				TopK:        5,
			}

			result, err := svc.Evaluate(context.Background(), "some raw description", tc.budget)
			if err != nil {
				t.Fatalf("Evaluate returned unexpected error: %v", err)
			}

			if result.Verdict != tc.wantVerdict {
				t.Errorf("Verdict = %q, want %q", result.Verdict, tc.wantVerdict)
			}
			if len(result.Comparables) != tc.wantComparable {
				t.Errorf("len(Comparables) = %d, want %d", len(result.Comparables), tc.wantComparable)
			}
		})
	}
}

func TestEvaluate_CategoryFallback(t *testing.T) {
	// The fake here can't distinguish which filter it was called
	// with, so this test verifies the outcome (a fallback search
	// still returns usable data), not the internal call sequence —
	// good enough to prove the fallback path doesn't silently break.
	svc := &Service{
		Clarifier: &fakeClarifier{spec: &agent.ClarificationResult{
			Category: "web-backend", // a category with zero seed entries, in this test
			Scope:    "a test project",
		}},
		Embedder: &fakeEmbedder{},
		VectorStore: &fakeVectorStore{
			results: []project.SearchResult{comparable("some other project", 900)},
		},
		TopK: 5,
	}

	result, err := svc.Evaluate(context.Background(), "desc", 900)
	if err != nil {
		t.Fatalf("Evaluate returned unexpected error: %v", err)
	}
	if len(result.Comparables) != 1 {
		t.Fatalf("expected fallback to still return comparables, got %d", len(result.Comparables))
	}
}

func TestEvaluate_ClarifierError(t *testing.T) {
	// When Clarify fails, Evaluate should fail too — the whole
	// pipeline can't proceed without a structured spec. This confirms
	// errors propagate rather than being silently swallowed.
	svc := &Service{
		Clarifier:   &fakeClarifier{err: errors.New("model unavailable")},
		Embedder:    &fakeEmbedder{},
		VectorStore: &fakeVectorStore{},
		TopK:        5,
	}

	_, err := svc.Evaluate(context.Background(), "desc", 100)
	if err == nil {
		t.Fatal("expected an error when Clarify fails, got nil")
	}
}

func TestEvaluate_ExplainerFailureIsNonFatal(t *testing.T) {
	// This is the specific behavior we designed deliberately: a
	// failing Explainer should not fail the whole Evaluate call, only
	// leave Explanation empty. This test exists to protect that
	// design decision from being accidentally broken later.
	svc := &Service{
		Clarifier: &fakeClarifier{spec: &agent.ClarificationResult{
			Category: "mobile-app",
			Scope:    "a test project",
		}},
		Embedder: &fakeEmbedder{},
		VectorStore: &fakeVectorStore{
			results: []project.SearchResult{comparable("salon app", 1200)},
		},
		Explainer: &fakeExplainer{err: errors.New("explainer model down")},
		TopK:      5,
	}

	result, err := svc.Evaluate(context.Background(), "desc", 1200)
	if err != nil {
		t.Fatalf("Evaluate should not fail when only the Explainer fails, got: %v", err)
	}
	if result.Explanation != "" {
		t.Errorf("expected empty Explanation on Explainer failure, got %q", result.Explanation)
	}
	if result.Verdict == "" {
		t.Error("expected a verdict to still be computed despite Explainer failure")
	}
}

func TestMedian(t *testing.T) {
	// median is unexported, but directly testable since this test
	// file lives in the same package (package pricing, not
	// pricing_test) — a normal, common choice for testing internal
	// helper functions alongside the exported behavior that uses them.
	cases := []struct {
		name   string
		sorted []float64
		want   float64
	}{
		{name: "odd count", sorted: []float64{100, 200, 300}, want: 200},
		{name: "even count", sorted: []float64{100, 200, 300, 400}, want: 250},
		{name: "single value", sorted: []float64{500}, want: 500},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := median(tc.sorted)
			if got != tc.want {
				t.Errorf("median(%v) = %v, want %v", tc.sorted, got, tc.want)
			}
		})
	}
}
