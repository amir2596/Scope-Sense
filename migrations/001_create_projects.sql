CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE projects (
    id         TEXT PRIMARY KEY,
    embedding  VECTOR(1024) NOT NULL, -- 1024 = bge-m3's output size (chosen for Persian support)
    metadata   JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- speeds up similarity search once you have more than a few hundred rows
CREATE INDEX ON projects USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
