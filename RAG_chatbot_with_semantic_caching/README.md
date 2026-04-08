# TaskFlow Assistant — RAG Chatbot with Semantic Caching

> A domain-specific conversational AI for the TaskFlow project management platform, built on LangGraph + ChromaDB with a live semantic cache that short-circuits repetitive queries before they reach the LLM.

---

## Overview

**TaskFlow Assistant** is a production-grade RAG chatbot that answers questions about the TaskFlow platform using a combination of:

1. A **semantic cache** — returns instant answers for questions that have already been asked (or semantically similar ones)
2. A **LangGraph RAG pipeline** — retrieves and reasons over TaskFlow documentation for novel questions
3. A **user feedback loop** — lets users approve LLM answers to be added to the cache over time

### Screenshot

<!-- Replace the path below with your actual screenshot -->
![TaskFlow Assistant](../docs/screenshots/rag_chatbot.png)

---

## Architecture

### LangGraph Workflow

![LangGraph Flow](outputs/langgraph_flow.png)

### Pipeline Detail

```
User Message
      │
      ▼
┌─────────────────────────┐
│     Semantic Cache       │  cosine distance < threshold → HIT
│  (all-MiniLM-L6-v2)     │  cosine distance ≥ threshold → MISS
└──────────┬──────────────┘
           │
       ┌───┴────┐
       │        │
      HIT      MISS
       │        │
       ▼        ▼
  Return     ┌──────────────────────────┐
  Cached     │  LangGraph RAG Pipeline  │
  Answer     └──────────┬───────────────┘
  (< 50ms)              │
                        ▼
              ┌─────────────────────┐
              │  generate_query     │  Reformulate user message
              │  _or_respond        │  as a retrieval query
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  ChromaDB Retrieval │  k=4, similarity / MMR
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  grade_documents    │  LLM grades relevance
              └──────┬──────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
     Relevant             Not Relevant
          │                     │
          ▼                     ▼
   generate_answer        rewrite_question
          │               (max 2 retries)
          │                     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Final Response     │  optionally added to cache
          └─────────────────────┘
```

---

## Components

### `cached_rag_chatbot_chroma.py` — `CachedRAGChatbot`

The orchestrator class that ties all components together.

| Method | Description |
|---|---|
| `query(question)` | Main entry point — checks cache first, runs RAG on miss |
| `_setup_graph()` | Builds the LangGraph StateGraph |
| `load_existing_vectorstore()` | Loads persisted ChromaDB collection |
| `load_cache_from_file(path)` | Hydrates semantic cache from CSV |
| `save_cache_to_file(path)` | Persists cache to CSV |

**LangGraph nodes:**

| Node | Role |
|---|---|
| `generate_query_or_respond` | Reformulates the user question for retrieval |
| `retrieve` | Fetches top-4 documents from ChromaDB |
| `grade_documents` | LLM rates document relevance (binary) |
| `generate_answer` | Produces final answer from context |
| `rewrite_question` | Rewrites query when docs are not relevant |
| `generate_fallback` | Returns graceful fallback when all retries exhausted |

**Configuration:**
- LLM: `ChatGroq` · `llama-3.3-70b-versatile` · temperature 0
- Max retries: 2
- Recursion limit: 10

---

### `semantic_cache.py` — `SemanticCache`

An in-memory semantic cache backed by HuggingFace sentence embeddings.

```python
cache = SemanticCache(threshold=0.1, model="all-MiniLM-L6-v2")
cache.hydrate_from_pairs(questions, answers)

result = cache.check("How do I reset my password?")
if result.hit:
    print(result.answer)      # instant, no LLM call
    print(result.distance)    # cosine distance to matched question
```

| Method | Description |
|---|---|
| `check(query)` | Returns `CacheResult(hit, answer, matched_question, distance)` |
| `hydrate_from_pairs(qs, as)` | Seed the cache from lists of Q&A pairs |
| `add_pair(q, a)` | Add a single new entry at runtime |
| `save_to_file(path)` | Export cache to CSV |
| `load_from_file(path)` | Import cache from CSV |

**Defaults:** threshold `0.1`, model `all-MiniLM-L6-v2` (384-d embeddings)

---

### `document_store_chroma.py` — `DocumentVectorStore`

A thin wrapper around ChromaDB for persistent document storage and retrieval.

| Method | Description |
|---|---|
| `load_existing()` | Load a pre-built collection from disk |
| `create_from_documents(docs)` | Chunk and embed documents into a new collection |
| `add_documents(docs)` | Append documents to an existing collection |
| `similarity_search(query, k)` | Return top-k documents by cosine similarity |
| `get_retriever(k, search_type)` | Return LangChain retriever (similarity or MMR) |

**Chunking config:** 500 chars · 50 char overlap · tiktoken tokeniser

---

### `rag_app.py` — Streamlit UI

The conversational front-end for TaskFlow Assistant.

#### Sidebar Features

| Control | Description |
|---|---|
| **Initialise** button | Loads ChromaDB and seeds the semantic cache |
| **Similarity Threshold** slider | Adjust cache sensitivity live (0.0 – 0.6) |
| **Session Stats** | Total queries · cache hits · hit rate · approved additions |
| **Cache Size** | Current number of cached Q&A pairs |
| **Download Cache** | Export current cache as CSV |

#### Chat Interface

- Renders conversation history with role avatars
- Displays **cache status badge** on every response:
  - Green — Cache HIT (similarity score shown)
  - Yellow/Orange — Cache MISS
- Shows **response time** in milliseconds

#### Feedback System

After each response, users can:
- **Approve** — adds the Q&A pair to the cache for future queries
- **Reject** — discards the answer (no cache write)

---

## Data

| File | Description |
|---|---|
| `data/taskflow_faq.csv` | 8 seed FAQ entries (id, question, answer) |
| `data/taskflow_docs.txt` | Full TaskFlow platform documentation |
| `data/taskflow_cache_seed.csv` | Initial cache hydration file |
| `taskflow_cache_approved.csv` | User-approved additions accumulated over sessions |

---

## Setup & Running

### 1. Prerequisites

```bash
pip install langchain langchain-huggingface langchain-chroma langchain-groq \
            langgraph python-dotenv sentence-transformers streamlit
```

### 2. Environment Variables

Create `.env` in the project root (or this directory):

```env
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...   # only needed if using LLM reranker
```

### 3. Build the Vector Store

Run once to chunk the documents and create the ChromaDB collection:

```bash
python prepare_chromadb.py
```

### 4. Launch the Chatbot

```bash
streamlit run rag_app.py
```

The app opens at `http://localhost:8501`.

---

## Integration Tests

```bash
python test_complete_flow.py
```

Covers 7 test scenarios:
1. System initialisation
2. Confirmed cache hits (semantically equivalent queries)
3. Confirmed cache misses (novel queries)
4. Vector store retrieval
5. Streaming responses
6. Runtime cache updates
7. Production simulation (mixed workload)

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `cache_threshold` | `0.1` | Cosine distance cut-off for cache hit |
| `llm_temperature` | `0` | Deterministic LLM output |
| `max_retries` | `2` | Query rewrite attempts before fallback |
| `retrieval_k` | `4` | Documents retrieved per query |
| `chunk_size` | `500` | Characters per document chunk |
| `chunk_overlap` | `50` | Overlap between consecutive chunks |
| `embedding_model` | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| `llm_model` | `llama-3.3-70b-versatile` | Groq model identifier |

---

*Part of the [CacheLab RAG](../README.md) project.*
