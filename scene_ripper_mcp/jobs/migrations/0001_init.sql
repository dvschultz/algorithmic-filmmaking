-- Initial schema for the jobs framework.
--
-- Single table, one row per job. Status transitions and column meanings are
-- documented in scene_ripper_mcp/jobs/store.py.
--
-- Pragmas (set on connection, not in migration): journal_mode=WAL,
-- synchronous=NORMAL, busy_timeout=5000, foreign_keys=ON.

CREATE TABLE IF NOT EXISTS jobs (
    id                       TEXT PRIMARY KEY,
    kind                     TEXT NOT NULL,
    status                   TEXT NOT NULL,
    idempotency_key          TEXT,
    args_json                TEXT NOT NULL,
    project_path             TEXT,
    project_mtime_at_start   REAL,
    progress                 REAL DEFAULT 0.0,
    status_message           TEXT,
    result_json              TEXT,
    error                    TEXT,
    queue_position           INTEGER,
    blocking_job_id          TEXT,
    created_at               REAL NOT NULL,
    updated_at               REAL NOT NULL,
    finished_at              REAL
);

-- Composite uniqueness on (kind, project_path, idempotency_key) — but only
-- for rows that supplied an idempotency key. SQLite's WHERE clause on a
-- partial index ignores NULL idempotency_key rows.
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency
    ON jobs (kind, project_path, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_finished_at ON jobs (finished_at);
