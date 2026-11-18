# Submission checklist

Student: Aubakirov Zhaslan. Group: ADA-2602M. These identity details were supplied for the final submission; the course and program match the earlier project documents.

| # | University technical requirement | Evidence | Status |
|---|---|---|---|
| 1 | At least five custom functions | Preparation, target derivation, splitting, training, evaluation, SQL and prediction functions in src/ | Verified |
| 2 | Lists and dictionaries | Target/model lists, configuration and per-label metric maps | Verified |
| 3 | Conditional branches | Model fitting, validation cases and dashboard routing | Verified |
| 4 | Loops | Chunks, targets, prediction batches and query exports | Verified |
| 5 | Persistent files | CSV, Parquet, JSON, SQLite and serialized model bundles | Verified locally; restricted runtime files excluded from Git |
| 6 | Error handling | Validation exceptions and malformed/missing-input tests | Verified |
| 7 | Interactive interface | Six Streamlit pages, live/batch inference, filters and SQL controls | Verified in the working project |
| 8 | Modules and standard library | Modular src/scripts; pathlib, sqlite3, json, logging and multiprocessing | Verified |
| 9 | Baseline plus two methods with quality/time/memory | Four original variants; five contextual refits; aggregated metrics and figures | Verified; protocols reported separately |
| 10 | Database, ER diagram, eight SQL queries and Pandas | sql/schema.sql, sql/queries.sql, docs/er_diagram.svg and aggregated exports | All eight executed locally; individual query-8 text export excluded |
| 11 | Data volume and memory awareness | Typed/chunked preparation, Parquet, sparse matrices and batched operations | Chunked preparation and sparse/batched operations verified |

The data contains 312,735 comment records and 140.41 decimal MB across four official CSVs. Template/label records do not create new comments. Fitting is not fully out of core.

## Software and artifact verification

On 18 November 2026, the existing real-data project passed 98 tests. The clean source repository passed 74 tests; 24 integration tests skipped because restricted data and saved models are absent. Neither run retrained the real models. A test pass count is not evidence of semantic model accuracy.

Final written report: eight pages. Final presentation: 12 slides. Both were opened through their native Office applications for PDF export. Every exported page/slide was rendered and reviewed. Presentation package, geometry, native chart/table and import checks passed. Original model, data, probability, threshold and experimental-result artifacts remain unchanged. Editorial updates to the report and presentation preserve their 8-page and 12-slide structure.

## Publication and next action

This source repository contains no raw/processed comment files, SQLite database, saved models, private runtime ZIP, original prompts, individual error exports or local absolute paths. The dashboard screenshot shows controls and scores only. Aggregated tables and charts are retained. Dataset access remains subject to the current Kaggle competition terms; the full rules require direct review through Kaggle, and no redistribution permission is inferred.

The owner-authorized publication target is the existing PUBLIC repository https://github.com/KosTyanka/toxic-comment-detection. Only the clean source and safe aggregated results belong there; the private laptop runtime remains separate. Verify the remote commit and visibility after each push. Before the exam, rehearse the demonstration and confirm the supplied name/group spelling. No personal contribution statement has been invented.
