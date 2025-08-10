-- Claude Artifact Manager Database Schema
-- Clean SQL schema for PyCharm compatibility

-- Artifacts table stores artifact metadata and references to content files
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    language TEXT,
    tags TEXT,
    created TEXT NOT NULL,
    modified TEXT NOT NULL,
    size INTEGER DEFAULT 0,
    checksum TEXT,
    chat_context TEXT,
    project TEXT DEFAULT 'default',
    favorite INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    parent_id TEXT,
    filepath TEXT
);

-- Projects table organizes artifacts into logical groups
CREATE TABLE projects (
    name TEXT PRIMARY KEY,
    description TEXT,
    created TEXT NOT NULL,
    color TEXT DEFAULT '#007acc'
);

-- Usage statistics table tracks access patterns
CREATE TABLE usage_stats (
    id INTEGER PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    accessed TEXT NOT NULL,
    action TEXT NOT NULL
);

-- Indexes for better query performance
CREATE INDEX idx_artifacts_tags ON artifacts(tags);
CREATE INDEX idx_artifacts_project ON artifacts(project);
CREATE INDEX idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX idx_artifacts_language ON artifacts(language);
CREATE INDEX idx_artifacts_created ON artifacts(created);
CREATE INDEX idx_usage_stats_artifact ON usage_stats(artifact_id);

-- Default project
INSERT INTO projects (name, description, created)
VALUES ('default', 'Default project for artifacts', '2025-01-01T00:00:00');

-- Artifact summary view
CREATE VIEW artifact_summary AS
SELECT
    id,
    title,
    artifact_type,
    language,
    project,
    created,
    size,
    favorite
FROM artifacts
ORDER BY modified DESC;

-- Project statistics view
CREATE VIEW project_stats AS
SELECT
    project,
    COUNT(*) as artifact_count,
    SUM(size) as total_size,
    MAX(created) as last_created
FROM artifacts
GROUP BY project;