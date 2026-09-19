package agent

import "context"

// ClarificationResult is what the agent produces from a raw client
// description: a structured spec the rest of the pipeline can embed
// and compare, plus any questions worth asking the client back.
type ClarificationResult struct {
	Scope               string   `json:"scope"`
	TechStack           []string `json:"tech_stack"`
	Deliverables        []string `json:"deliverables"`
	Category            string   `json:"category"`
	ClarifyingQuestions []string `json:"clarifying_questions"`
}

// ClarificationAgent turns a raw, possibly vague client description
// into a ClarificationResult. Kept as an interface (like Embedder and
// VectorStore) so the concrete LLM/prompt implementation can change
// without touching whatever calls it.
type ClarificationAgent interface {
	Clarify(ctx context.Context, rawDescription string) (*ClarificationResult, error)
}
