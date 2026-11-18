# Relational model

```mermaid
erDiagram
  comments ||--o{ labels : has_ground_truth
  label_types ||--o{ labels : defines
  comments ||--o{ predictions : receives
  label_types ||--o{ predictions : defines
  model_runs ||--o{ predictions : produces
  comments {
    TEXT comment_id PK
    TEXT comment_text
    TEXT dataset_split
    INTEGER character_count
    INTEGER word_count
  }
  labels {
    TEXT comment_id PK,FK
    TEXT label_name PK,FK
    INTEGER label_value
  }
  label_types {
    TEXT label_name PK
  }
  model_runs {
    TEXT run_id PK
    TEXT model_name
    TEXT configuration
    REAL training_time
    REAL peak_memory
    TEXT created_at
  }
  predictions {
    TEXT run_id PK,FK
    TEXT comment_id PK,FK
    TEXT label_name PK,FK
    REAL probability
    INTEGER predicted_label
  }
  metadata {
    TEXT key PK
    TEXT value
  }
```

Composite primary keys enforce uniqueness. Ground-truth values are 0/1 only.
Unscored official rows have comments but no ground-truth label records.
`dataset_split=train` identifies the original training source, including validation
rows. Exact train/validation membership is saved per experiment in
`models/<experiment_id>/split_assignments.parquet`; persisted train-source
predictions cover validation only. No training-set predictions enter error analysis.
`metadata` stores the input manifest and is intentionally independent.
Rendered view: `er_diagram.svg`. Schema source: `sql/schema.sql`.
