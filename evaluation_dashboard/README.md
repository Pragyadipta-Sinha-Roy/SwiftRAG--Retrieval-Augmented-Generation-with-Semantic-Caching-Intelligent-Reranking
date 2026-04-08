# Semantic Cache Analyzer — Evaluation Dashboard

> An interactive 5-page Streamlit application for benchmarking, tuning, and comparing semantic cache strategies against labelled test data.

---

## Overview

The **Semantic Cache Analyzer** is the evaluation and research companion to the CacheLab RAG system. It lets you upload your own FAQ and query datasets, run experiments across different caching strategies and distance thresholds, and visualise the precision-recall trade-offs that inform production configuration.

---

## Pages

### Landing Page — `app.py`

The entry point to the dashboard. Displays a high-level overview of what the Semantic Cache Analyzer does, feature cards for each section, and navigation links.

---

### Page 1: About — `pages/1_about.py`

A comprehensive educational guide covering:

- What semantic caching is and why it matters
- How sentence embeddings represent meaning as vectors
- Distance metrics: cosine distance, Euclidean, dot product
- Three caching strategies: Exact Match, Fuzzy Match, Semantic Match
- Four reranking strategies: None, Keyword, Cross-Encoder, LLM
- Key evaluation metrics: Precision, Recall, F1, Accuracy, Specificity, Hit Rate
- Real-world use cases across industries
- Choosing a threshold: conceptual trade-offs explained

---

### Page 2: Data — `pages/2_data.py`

Upload and inspect your datasets before running experiments.

**Required files:**

| File | Required Columns | Description |
|---|---|---|
| `ground_truth.csv` | `id`, `question`, `answer` | The FAQ entries seeded into the cache |
| `test_dataset.csv` | `question`, `answer`, `src_question_id`, `cache_hit` | Test queries with expected hit/miss labels |

**`cache_hit` column values:**
- `TRUE` — this query should match a cached entry (paraphrase of a known question)
- `FALSE` — this query is genuinely novel and should miss

The page displays row counts, column previews, data distributions, and validates the expected schema.

---

### Page 3: Testing — `pages/3_testing.py`

The main experimentation page. Configure parameters, run the cache against your test set, and inspect results in detail.

**Configuration options:**

| Parameter | Range | Default | Description |
|---|---|---|---|
| Cache Strategy | Exact / Fuzzy / Semantic | Semantic | Matching algorithm |
| Fuzzy Threshold | 0.0 – 1.0 | 0.40 | Minimum similarity for fuzzy match |
| Embedding Model | 3 options | all-MiniLM-L6-v2 | Sentence transformer model |
| Semantic Threshold | 0.05 – 0.80 | 0.42 | Cosine distance cut-off for hit |
| Top-K Matches | 1 – 10 | 5 | Candidates retrieved per query |
| Query Filter | All / Hits Only / Misses Only | All | Subset of test set to run |

**Outputs:**

- **Summary metrics** — Total tests, expected hits, actual hits, miss rate, average distance
- **Detailed results table** — Per-query: input, matched question, distance, hit/miss, correct/incorrect
- **Distance distributions** — Histogram and box plot for hit and miss groups
- **Query-level inspection** — Select any query to see its top-K matches with distances
- **Confusion matrix** — TP / TN / FP / FN counts with Accuracy, Precision, Recall, F1, Specificity
- **Strategy comparison** — Side-by-side metrics for all three cache strategies
- **Performance benchmark** — Speed (queries/sec) vs hit rate scatter for each strategy
- **Export** — Download results as CSV and configuration as JSON

---

### Page 4: Optimization — `pages/4_optimization.py`

Systematically sweep across threshold values and embedding models to find the configuration that best fits your data.

**What it does:**

- Runs the cache at multiple threshold values (e.g. 0.05 to 0.80 in steps)
- Plots Precision, Recall, and F1 as functions of threshold
- Overlays Hit Rate and Accuracy curves
- Highlights the threshold that maximises F1
- Compares multiple embedding models side-by-side
- Outputs a recommended configuration based on your data

**Use this page when:**

- You have a new FAQ dataset and need to pick a threshold from scratch
- You are evaluating a new embedding model
- You want to understand the precision-recall trade-off for your domain

---

### Page 5: Reranker — `pages/5_reranker.py`

Compare the four reranking strategies in a two-stage retrieval setup.

**Two-stage retrieval:**

```
Stage 1 (High Recall): Semantic cache returns top-K candidates at a loose threshold
Stage 2 (High Precision): Reranker filters and scores candidates → final decision
```

**Strategies compared:**

| Strategy | Description | Latency | When to Use |
|---|---|---|---|
| **Baseline** | No reranker, raw semantic score decides | ~1 ms | Baseline comparison |
| **Simple Keyword** | Token overlap between query and candidate | ~1 ms | High throughput, moderate precision |
| **Cross-Encoder** | Neural pairwise relevance scoring | ~50 ms | Good precision, acceptable latency |
| **LLM Reranker** | GPT with structured output and reasoning | ~500 ms | Maximum precision, cost not critical |

The page shows per-strategy confusion matrices, false positive / false negative rates, and an end-to-end latency breakdown.

---

## The `cachelab` Library

The dashboard is backed by a reusable Python library located at `src/cachelab/`. Install it in editable mode:

```bash
pip install -e src/
```

### Package Layout

```
src/cachelab/
├── cache/
│   ├── exact_match_cache.py      # Case-insensitive dictionary lookup
│   ├── fuzzy_match_cache.py      # Levenshtein / SequenceMatcher
│   └── semantic_match_cache.py   # SentenceTransformer + cosine distance
├── reranker/
│   ├── simple_keyword_reranker.py
│   ├── cross_encoder.py
│   ├── llm_reranker.py
│   ├── reranked_cache.py         # Two-stage: semantic cache + reranker
│   └── adaptors.py               # Uniform reranker interface adapters
├── evaluate/
│   ├── cache_evaluator.py        # Confusion matrix + metric calculation
│   ├── evaluatable_cache.py      # Wraps any cache to track TP/TN/FP/FN
│   └── evaluation_result.py      # Result dataclasses
└── utils/
    ├── cache_utils.py             # CacheResult, CacheResults dataclasses
    └── embedding_utils.py         # Batch cosine distance utilities
```

### Quick Example

```python
from cachelab.cache.semantic_match_cache import SemanticMatchCache
from cachelab.evaluate.evaluatable_cache import EvaluatableCache
from cachelab.evaluate.cache_evaluator import CacheEvaluator
import pandas as pd

# Load data
ground_truth = pd.read_csv("data/ground_truth.csv")
test_data    = pd.read_csv("data/test_dataset.csv")

# Build cache
cache = SemanticMatchCache(threshold=0.35, model="all-MiniLM-L6-v2")
cache.hydrate_from_df(ground_truth)

# Wrap for evaluation
eval_cache = EvaluatableCache(cache)
evaluator  = CacheEvaluator(eval_cache)

# Run
results = evaluator.evaluate(test_data)
print(results.summary())
```

---

## Data Format Reference

### `ground_truth.csv`

```csv
id,question,answer
1,How do I reset my password?,Go to Settings → Security → Reset Password
2,What file formats are supported?,We support PDF, DOCX, TXT, and CSV
...
```

### `test_dataset.csv`

```csv
question,answer,src_question_id,cache_hit
How can I change my password?,Go to Settings → Security → Reset Password,1,TRUE
What is the refund policy?,Our refund policy allows returns within 30 days,,FALSE
```

---

## Installation

```bash
cd evaluation_dashboard
pip install -r requirements.txt
pip install -e src/
```

**requirements.txt includes:**
- `streamlit==1.39.0`
- `pandas==2.2.0`
- `plotly==5.24.0`
- `pyprojroot`
- `sentence-transformers` (via cachelab)

---

## Running the Dashboard

```bash
cd evaluation_dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Environment Variables

The LLM reranker (Page 5) requires an OpenAI key:

```env
OPENAI_API_KEY=sk-...
```

Place this in a `.env` file at the project root.

---

## Visualisations

Pre-generated charts are saved to `visualizations/`:

| File | Description |
|---|---|
| `threshold_analysis.png` | Precision / Recall / F1 / Hit Rate vs threshold sweep |
| `evaluation_report.txt` | Text summary of evaluation run |

---

*Part of the [CacheLab RAG](../README.md) project.*
