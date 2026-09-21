# ScopeSense

![ScopeSense demo](docs/ScopeSense.gif)

Freelance clients on platforms like Upwork routinely write vague, one-line
project descriptions and attach unrealistic budgets — and freelancers waste
real time going back and forth to figure out whether a lead is even worth
pursuing. ScopeSense is a small agentic pipeline that takes a raw client
description and turns it into a structured spec, a data-backed price
range from similar past projects, and a plain-language verdict on whether
the stated budget is realistic.

It runs entirely on local, open-source models via Ollama — no cloud API
key, no per-request cost, and no dependency on a provider that may not be
reachable from every region.

## How it works

1. **Clarify** — an LLM agent reads the raw description and extracts a
   structured spec: scope, tech stack, deliverables, category, and any
   clarifying questions a real freelancer would ask before quoting.
2. **Embed + retrieve** — the structured spec is embedded and used to
   search a Postgres/pgvector store of past closed projects for the
   closest comparable deals, preferring same-category matches with a
   fallback to a looser search if none are found.
3. **Price check** — min/median/max price and a verdict
   (`realistic` / `underpriced` / `overpriced` / `no_data`) are computed
   with plain arithmetic against the retrieved comparables — no LLM
   involved in this step, deliberately, since it doesn't need one.
4. **Explain** (optional) — a second, smaller LLM call turns the verdict
   and numbers into a short, plain-language explanation a client could
   actually read.

## Engineering decisions worth knowing about

- **Prompt-injection defense.** The client's raw text is untrusted input —
  it's wrapped in `<client_description>` tags and the system prompt
  explicitly instructs the model to treat everything inside as data, never
  as instructions, even if it contains phrases like "ignore previous
  instructions." It's also capped at 2000 characters before it ever
  reaches the model.
- **Category-aware retrieval with a fallback.** Vector search filters by
  the clarifier's detected category first (comparing a mobile app against
  other mobile apps, not against unrelated web scraping gigs), and only
  falls back to an unfiltered search if that strict match returns nothing
  — better a looser comparison than no data at all.
- **Bounded concurrency, not unbounded goroutines.** The API caps
  concurrent evaluations at 4 via a semaphore, so a burst of requests
  queues instead of overwhelming the local Ollama instance; a request
  waiting for capacity respects its own cancellation instead of blocking
  forever.
- **Automatic model warm-up at startup.** Ollama unloads an idle model
  from memory after 5 minutes by default, and a cold load can add 30-60+
  seconds to whichever request happens to arrive first. The server pays
  that cost once at startup instead of passing it on to a real user.
- **Tuned for CPU inference, not just correctness.** Output length is
  explicitly capped (`num_predict`) per call, since generation is
  token-by-token and dominates latency on CPU far more than model size or
  prompt length do.
- **Model choice was revised, not just picked once.** Started with
  `gemma4`, which turned out to be a multimodal build (vision + audio
  encoders) adding real load time for capabilities this project never
  uses; switched to the text-only `qwen2.5:7b-instruct` for faster,
  lighter inference with no loss of JSON-following reliability.
- **The verdict math is separate from the LLM calls on purpose** — a
  wrong price threshold is a one-line fix and fully testable; keeping it
  out of a prompt keeps the one part of the pipeline that must be exactly
  right deterministic.

## Project layout

- `cmd/api` — the HTTP server (`/api/evaluate`, `/health`): request
  validation, bounded concurrency, per-request timeout, startup model
  warm-up, and request/response logging.
- `cmd/evaluate` — a CLI entry point for running one evaluation directly,
  without going through the HTTP server — useful for quick manual testing.
- `cmd/seed` — one-off script that reads `seed_projects.json` (25 sample
  closed projects), embeds each entry, and loads it into Postgres. Run
  once before anything else.
- `internal/project` — domain types and the `VectorStore`/`Embedder`
  interfaces. No framework or database dependency lives here.
- `internal/agent` — the `ClarificationAgent` interface and its Ollama
  implementation (the "Clarify" step above).
- `internal/embedding` — `OllamaEmbedder` (local, active) and
  `OpenAIEmbedder` (kept for reference, unused while cloud API access
  isn't assumed to be available).
- `internal/vectorstore` — the Postgres + pgvector implementation of
  `VectorStore`.
- `internal/pricing` — the `Service` that wires the above together: the
  actual pipeline orchestration, the price-math, and the verdict logic.
- `internal/explainer` — the optional "Explain" step's Ollama
  implementation, kept as its own package (rather than folded into
  `pricing`) to avoid a circular import between it and `agent`.
- `migrations` — SQL to create the `projects` table.

## Setup

1. **Install Ollama** — https://ollama.com, then pull the two local
   models this project uses:
   ```
   ollama pull bge-m3               # embeddings
   ollama pull qwen2.5:7b-instruct  # clarification + explanation agent
   ```
   Start the server if it isn't already running as a background service:
   ```
   ollama serve
   ```

2. **Run Postgres with pgvector.** Easiest via Docker:
   ```
   docker run -d --name scopesense-db -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
   ```

3. **Apply the migration.** `psql` ships inside the container itself, so
   run it via `docker exec` rather than needing it installed on your host:
   ```
   Get-Content migrations\001_create_projects.sql | docker exec -i scopesense-db psql -U postgres -d postgres
   ```

4. **Set the one required environment variable** (no API key needed —
   only the database connection):
   ```
   # PowerShell
   $env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"

   # bash
   export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/postgres"
   ```

5. **Install Go dependencies and run the seed script:**
   ```
   go mod tidy
   go run ./cmd/seed
   ```

6. **Run the API server:**
   ```
   go run ./cmd/api
   ```
   Startup will take up to a minute or two the first time, since it warms
   both models before accepting requests — watch for `clarifier model
   warm` / `embedding model warm` in the log. Confirm it's up:
   ```
   curl http://localhost:8080/health
   ```

7. **Keep models warm between requests during a dev/demo session.**
   Ollama's 5-minute idle unload still applies after startup, so if the
   server sits idle for a while, set this **persistently** (System
   Properties → Environment Variables on Windows — not just `$env:` in
   one PowerShell session, which resets when you close it):
   ```
   OLLAMA_KEEP_ALIVE=60m
   ```

## Notes / open decisions

- `bge-m3` is used for embeddings. No need to switch to a different
  embedding model at this stage — doing so would mean re-seeding the
  database (a different model changes the embedding dimension, which
  the migration's vector column is fixed to) for a benefit that's
  mostly theoretical.
- `Think: false` is set on clarifier requests; it's a no-op for
  `qwen2.5:7b-instruct` (not a reasoning model) but is left in place in
  case the model is swapped for one that emits a reasoning block, which
  would otherwise break JSON parsing.
- No automated tests yet. The pricing/verdict math in `internal/pricing`
  is the highest-value, lowest-effort place to start, since it's pure
  functions with no LLM or network dependency.
