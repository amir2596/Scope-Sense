// Package explainer implements pricing.Explainer using a local Ollama
// chat model. It's a separate package (not internal/agent) so it can
// import internal/pricing for the Result/Comparable types without
// creating a cycle — pricing already imports agent for
// ClarificationResult, so agent importing pricing back would be
// circular; explainer avoids that by depending only on pricing.
package explainer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"scopesense/internal/pricing"
)

type OllamaExplainer struct {
	baseURL string
	model   string
	client  *http.Client
}

func NewOllamaExplainer(baseURL, model string) *OllamaExplainer {
	return &OllamaExplainer{baseURL: baseURL, model: model, client: &http.Client{}}
}

var _ pricing.Explainer = (*OllamaExplainer)(nil)

const explainerSystemPrompt = `You write a short, plain-language explanation (2-3 sentences) for a freelance client about whether their stated project budget is realistic, based on similar past projects. The user message below will explicitly tell you which language to write in — follow that instruction exactly. Be direct but polite. Reference at least one specific comparable project and its price. Do not use markdown formatting. Respond with plain text only, no JSON.`

// persianScript matches Persian/Arabic script characters. Same
// detection as internal/agent's clarifier — duplicated here rather
// than shared, since explainer deliberately depends only on
// internal/pricing (see the package comment above) to avoid an
// import cycle.
var persianScript = regexp.MustCompile(`[\x{0600}-\x{06FF}\x{0750}-\x{077F}\x{FB50}-\x{FEFF}]`)

func looksPersian(s string) bool {
	return persianScript.MatchString(s)
}

type ollamaChatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Think    bool          `json:"think"`
	Stream   bool          `json:"stream"`
	Options  ollamaOptions `json:"options"`
}

type ollamaOptions struct {
	// NumPredict caps generated tokens. A 2-3 sentence explanation
	// needs far fewer tokens than the clarifier's structured JSON, so
	// this is capped tighter — mainly a safety net against the model
	// rambling past what was asked for, which would cost latency for
	// no benefit.
	NumPredict int `json:"num_predict"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatResponse struct {
	Message chatMessage `json:"message"`
}

func (e *OllamaExplainer) Explain(ctx context.Context, r *pricing.Result) (string, error) {
	userContent := buildPrompt(r)

	reqBody, err := json.Marshal(ollamaChatRequest{
		Model: e.model,
		Messages: []chatMessage{
			{Role: "system", Content: explainerSystemPrompt},
			{Role: "user", Content: userContent},
		},
		Think:   false,
		Stream:  false,
		Options: ollamaOptions{NumPredict: 150},
	})
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		e.baseURL+"/api/chat", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("call ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	var chatResp ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	return strings.TrimSpace(chatResp.Message.Content), nil
}

// buildPrompt turns a Result's numbers into a plain description the
// model can write a sentence about — deliberately not JSON, since
// the output here is a sentence, not structured data.
func buildPrompt(r *pricing.Result) string {
	var b strings.Builder
	if r.Spec != nil && r.Spec.Scope != "" {
		fmt.Fprintf(&b, "Project description: %s\n", r.Spec.Scope)
	}
	fmt.Fprintf(&b, "Client stated budget: %.0f %s\n", r.ClientStatedBudget, currencyOf(r))
	fmt.Fprintf(&b, "Verdict: %s\n", r.Verdict)
	fmt.Fprintf(&b, "Comparable projects (real closed deals):\n")
	for _, c := range r.Comparables {
		fmt.Fprintf(&b, "- %s: %.0f %s\n", c.Title, c.AgreedPrice, c.Currency)
	}

	if r.Spec != nil && looksPersian(r.Spec.Scope) {
		b.WriteString("\nWrite your explanation in Persian (Farsi) — do not use English.\n")
	} else {
		b.WriteString("\nWrite your explanation in English — do not use Persian.\n")
	}

	return b.String()
}

func currencyOf(r *pricing.Result) string {
	if len(r.Comparables) > 0 {
		return r.Comparables[0].Currency
	}
	return "USD"
}
