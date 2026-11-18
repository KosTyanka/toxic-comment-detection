# Toxic Comment Detection

Machine Learning Classification and SQL-Based Data Analysis

## 1. Project Overview

Course: Programming for Data Analysis and Database. Program: Big Data Analysis, Master's degree.
Student: Aubakirov Zhaslan. Group: ADA-2602M.

Python processing, multi-label classification, SQLite analysis and a six-page Streamlit dashboard form one reproducible project. The original experiment and later contextual study use different splits and are reported separately.

## 2. Dataset

The [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) supplies 159,571 training and 153,164 test comments. Six labels are `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. A seventh target, `any_toxic`, is the maximum of the six ground-truth labels on fully scored rows; it includes obscene speech.

Use your own Kaggle account, accept the [competition rules](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules), and download the files into `data/raw/`. Unknown test labels of -1 are excluded from evaluation. Exactly 63,978 official test rows are scored. No raw/processed comment text, SQLite database, trained bundles, or individual error exports are included here. Review current competition terms before redistribution; access to data does not itself authorize redistribution.

## 3. Features

The dashboard provides overview, live detection, dataset exploration, model comparison, SQL analytics and batch prediction. Live detection supports side-by-side models and global, saved per-label, or custom thresholds. Thresholds affect decisions only. The application loads saved models and never trains on startup.

## 4. Models and Results

Original experiment: 127,656 fit / 31,915 validation rows, seed 42. Model selection used mean ROC-AUC across the six original labels.

| Model | Validation mean AUC | Binary precision | Binary recall | Binary F1 |
|---|---:|---:|---:|---:|
| Prior baseline | 0.500000 | 0.0000 | 0.0000 | 0.0000 |
| Bag of words / Naive Bayes | 0.923175 | 0.5923 | 0.7849 | 0.6751 |
| TF-IDF / LR | 0.980558 | 0.9382 | 0.6037 | 0.7347 |
| Balanced TF-IDF / LR | 0.981074 | 0.6659 | 0.8610 | 0.7510 |

Only the selected balanced LR received official-test evaluation: mean six-label AUC 0.974843; binary precision 44.95%, recall 89.44%, F1 0.598339. Its binary confusion matrix is [[50897, 6838], [659, 5584]], actual rows and predicted columns in order 0/1.

Contextual study: five fresh refits on 111,699 fit / 23,936 calibration / 23,936 assessment rows. Calibration selected a word + character TF-IDF balanced LR candidate and per-label thresholds. Against the refitted balanced-word model at 0.5, assessment FPR fell from 4.53% to 2.18%, precision increased from 68.17% to 80.78%, and recall fell from 85.78% to 80.81%. Mean six-label AUC was 0.985939. The candidate has no official-test evaluation. The three reported contextual failures remain unresolved, so the original default is retained. See aggregated results in `results/` and the final report.

## 5. Project Structure

```text
app.py          Streamlit entry point
src/            Processing, database, models, metrics and prediction
scripts/        Explicit pipeline commands
sql/            Schema and eight named queries
tests/          Synthetic software tests and optional runtime integration checks
results/        Aggregated experimental results and figures
docs/           Architecture, ER diagram and submission checklist
reports/        Final written report
presentation/   Final slides and PDF
```

## 6. Installation and Usage

Tested with Python 3.12.14. From the repository folder on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\Activate.ps1
```

After activating the environment, run these commands in order. The downloader requires your own Kaggle authentication; manual extraction into `data/raw/` is also supported.

```bash
python -m scripts.download_data
python -m scripts.validate_data
python -m scripts.prepare_data --chunksize 10000
python -m scripts.build_database
python -m scripts.train_models --seed 42 --validation-size 0.2 --max-features 50000 --min-df 2
python -m scripts.evaluate_models
python -m scripts.verify_artifacts
python -m streamlit run app.py
```

Open http://localhost:8501. An empty checkout displays setup status until local data and models are available. The private runtime package is separate and works without retraining after dependency installation.

Optional contextual reproduction uses only the original training source:

```bash
python -m scripts.contextual_experiment
python -m pytest -q
```

The optional diagnostics script checks the historical preservation manifest and supplied-case audit, which are retained only in the private working project. In a clean checkout, run the experiment and inspect its versioned outputs directly; do not use official-test labels for tuning. Each contextual experiment creates new folders. Restart the Streamlit process after changing prediction code.

## 7. SQL Database

`sql/schema.sql` defines comments, label types, labels, model runs, predictions and metadata, with primary/foreign keys and indexes. `docs/er_diagram.svg` shows the relationships. Eight queries cover split counts, binary distribution, category prevalence, co-occurrence, length groups, label ranking, prediction distribution and high-confidence errors. Aggregated exports 1-7 are included; query 8 is executable locally but its individual comment/error export is excluded. Original local database counts: 312,735 comments, 1,564,843 labels, four runs and 1,341,466 predictions. Rebuilding the database replaces prior run records; use a separate copy for reproduction.

## 8. Limitations

The dataset has 312,735 comments and 140.41 MB across four CSVs. Data preparation is chunked and features are sparse, but model fitting is not fully out of core. Scores are uncalibrated; profanity, criticism and abusive intent are not reliably distinguished. Rare-label estimates are uncertain. The contextual study improves an aggregate operating point and increases inference/memory costs; it does not establish general contextual understanding. No Transformer or fresh independently adjudicated intent benchmark was evaluated.

See `docs/submission_checklist.md` for all 11 technical requirements. Source code uses the existing MIT license; dataset rights remain separate.
