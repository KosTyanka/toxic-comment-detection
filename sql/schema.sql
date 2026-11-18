PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS comments (
 comment_id TEXT PRIMARY KEY,
 comment_text TEXT NOT NULL,
 dataset_split TEXT NOT NULL CHECK(dataset_split IN ('train','test')),
 character_count INTEGER NOT NULL CHECK(character_count >= 0),
 word_count INTEGER NOT NULL CHECK(word_count >= 0)
);
CREATE TABLE IF NOT EXISTS label_types (
 label_name TEXT PRIMARY KEY
);
INSERT OR IGNORE INTO label_types VALUES
 ('toxic'),('severe_toxic'),('obscene'),('threat'),('insult'),('identity_hate'),('any_toxic');
CREATE TABLE IF NOT EXISTS labels (
 comment_id TEXT NOT NULL REFERENCES comments(comment_id) ON DELETE CASCADE,
 label_name TEXT NOT NULL REFERENCES label_types(label_name),
 label_value INTEGER NOT NULL CHECK(label_value IN (0,1)),
 PRIMARY KEY(comment_id,label_name)
);
CREATE TABLE IF NOT EXISTS model_runs (
 run_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, configuration TEXT NOT NULL,
 training_time REAL NOT NULL CHECK(training_time >= 0),
 peak_memory REAL NOT NULL CHECK(peak_memory >= 0), created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
 run_id TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
 comment_id TEXT NOT NULL REFERENCES comments(comment_id) ON DELETE CASCADE,
 label_name TEXT NOT NULL REFERENCES label_types(label_name),
 probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
 predicted_label INTEGER NOT NULL CHECK(predicted_label IN (0,1)),
 PRIMARY KEY(run_id,comment_id,label_name)
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_comments_split_length ON comments(dataset_split,character_count);
CREATE INDEX IF NOT EXISTS idx_labels_name_value ON labels(label_name,label_value);
CREATE INDEX IF NOT EXISTS idx_predictions_comment_label ON predictions(comment_id,label_name);
