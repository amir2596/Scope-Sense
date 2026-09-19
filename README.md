# ScopeSense

An AI layer for freelance marketplaces (Upwork-style): it turns a vague client description into a structured technical spec, and estimates whether the client's stated budget is realistic based on similar past projects.
1. A clarification agent that turns a vague client description into a
   structured spec (scope, tech stack, deliverables) plus clarifying
   questions.
2. A RAG-grounded "budget sanity check" that retrieves similar past
   projects and compares the client's stated budget against real
   closing prices.

Runs entirely on local, open-source models via Ollama — no cloud API
key needed anywhere (chosen specifically to work from regions where
OpenAI's API is not accessible).

## Project layout

- `cmd/seed` — one-off script that reads `seed_projects.json`, embeds
  each entry, and loads it into Postgres. Run this once before
  anything else.
- `internal/project` — domain types and the `VectorStore`/`Embedder`
  interfaces. No framework or database dependency lives here.
- `internal/vectorstore` — Postgres + pgvector implementation of
  `VectorStore`.
- `internal/embedding` — `OllamaEmbedder` (local, active) and
  `OpenAIEmbedder` (kept for reference, unused while OpenAI access is
  unavailable).
- `internal/agent` — `ClarificationAgent` interface and its Ollama
  implementation.
- `migrations` — SQL to create the `projects` table.

## Setup

1. **Install Ollama** — https://ollama.com, then pull the two local
   models this project uses:
   ```
   ollama pull bge-m3       # embeddings, multilingual (Persian included)
   ollama pull qwen2.5:7b-instruct   # reasoning / clarification agent (text-only, lighter than gemma4)
   ```
   Start the server (if not already running as a background service):
   ```
   ollama serve
   ```

2. **Run Postgres with pgvector.** Easiest via Docker:
   ```
   docker run -d --name scopesense-db -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
   ```

3. **Apply the migration:**
   ```
   psql "postgresql://postgres:postgres@localhost:5432/postgres" -f migrations/001_create_projects.sql
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
   This embeds and stores all 15 entries from `seed_projects.json`.

6. **The server warms both models up automatically on startup** (see `warmUpModels` in `cmd/api/main.go`), so the first real request doesn't pay the cold-load cost. Startup itself will take up to a minute or two the first time — that's expected, watch the log for `clarifier model warm` / `embedding model warm`.

   During a session, Ollama still unloads an idle model after 5 minutes by default, so if you leave the server idle for a while mid-demo, the *next* request will be slow again even though startup warmed things up. To avoid that, set `OLLAMA_KEEP_ALIVE` **persistently** (System Properties → Environment Variables on Windows, not just `$env:` in one PowerShell session which only lasts until you close it) so it's in effect every time `ollama serve` starts:
   ```
   OLLAMA_KEEP_ALIVE=60m
   ```

## Notes / open decisions

- `bge-m3` was chosen over `nomic-embed-text` specifically because it
  supports Persian; the embedding dimension (1024) in the migration
  must match whatever embedding model is actually in use.
- Switched from `gemma4` to `qwen2.5:7b-instruct` for the clarifier and
  explainer: `gemma4` pulled a multimodal build (vision + audio encoders)
  that added real load time and memory on CPU for capabilities this
  project never uses. `qwen2.5:7b-instruct` is text-only, loads faster,
  and is still strong at following instructions for structured JSON
  output.
- `Think: false` is set on clarification-agent requests; it's a no-op for
  `qwen2.5:7b-instruct` (not a reasoning model) but is left in place in
  case the model is swapped for one that emits a reasoning block, which
  would otherwise break JSON parsing.
- Not yet built: the pricing/retrieval step that combines the
  clarification agent's output with a `VectorStore.Search` call, and
  the Flutter client.
