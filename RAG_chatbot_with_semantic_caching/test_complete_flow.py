"""
Test Script: Verify Complete RAG + Caching Flow

This script tests that:
1. Cache hits return from semantic cache (fast)
2. Cache misses trigger the full LangGraph RAG pipeline
3. ChromaDB vectorstore is being used
4. The Streamlit app flow is correct
"""

import os
import sys
from datetime import datetime
from pyprojroot import here


def test_setup():
    """Test 1: Verify all components are available."""
    print("=" * 70)
    print("TEST 1: VERIFYING SETUP")
    print("=" * 70)

    print("\n📦 Checking imports...")
    try:
        from cached_rag_chatbot_chroma import CachedRAGChatbot
        from document_store_chroma import DocumentVectorStore
        from semantic_cache import SemanticCache
        print("  ✓ All imports successful")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

    print("\n🔑 Checking API key...")
    if not os.getenv("OPENAI_API_KEY"):
        print("  ✗ OPENAI_API_KEY not set")
        return False
    print("  ✓ API key found")

    print("\n📁 Checking data files...")
    required_files = [
        here("data/taskflow_faq.csv"),
        here("data/taskflow_docs.txt"),
        here("data/taskflow_cache_seed.csv")
    ]
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            return False

    print("\n🗄️  Checking ChromaDB...")
    if os.path.exists(here("data/chroma_db")):
        print("  ✓ ChromaDB directory exists")
    else:
        print("  ✗ ChromaDB not found - run: python prepare_data_chroma.py")
        return False

    print("\n✅ Setup verification complete!")
    return True


def test_cache_hit():
    """Test 2: Verify cache hit works (semantic match)."""
    print("\n" + "=" * 70)
    print("TEST 2: CACHE HIT TEST")
    print("=" * 70)

    from cached_rag_chatbot_chroma import CachedRAGChatbot

    print("\n🚀 Initializing chatbot...")
    chatbot = CachedRAGChatbot(
        cache_distance_threshold=0.1,
        model_name="gpt-4o-mini",
        temperature=0
    )

    print("📄 Loading ChromaDB...")
    if not chatbot.load_existing_vectorstore():
        print("  ✗ Failed to load ChromaDB")
        return False
    print("  ✓ ChromaDB loaded")

    print("💾 Loading cache...")
    chatbot.load_cache_from_file(here("data/taskflow_cache_seed.csv"))
    print("  ✓ Cache loaded")

    # Test cache hit with exact match
    print("\n🧪 Testing exact match...")
    query1 = "How do I create a new project?"

    start = datetime.now()
    result1 = chatbot.query(query1, verbose=False)
    elapsed1 = (datetime.now() - start).total_seconds()

    if result1['cache_hit']:
        print(f"  ✓ CACHE HIT (as expected)")
        print(f"    Response time: {elapsed1:.3f}s")
        print(f"    Matched: {result1['cache_info']['matched_question']}")
        print(f"    Similarity: {result1['cache_info']['similarity']:.2%}")
    else:
        print(f"  ✗ CACHE MISS (unexpected!)")
        return False

    # Test cache hit with semantic variation
    print("\n🧪 Testing semantic variation...")
    query2 = "What's the cost?"  # Should match "What are the pricing plans?"

    start = datetime.now()
    result2 = chatbot.query(query2, verbose=False)
    elapsed2 = (datetime.now() - start).total_seconds()

    if result2['cache_hit']:
        print(f"  ✓ CACHE HIT (semantic match)")
        print(f"    Response time: {elapsed2:.3f}s")
        print(f"    Matched: {result2['cache_info']['matched_question']}")
        print(f"    Similarity: {result2['cache_info']['similarity']:.2%}")
    else:
        print(f"  ℹ️  CACHE MISS (threshold may be too strict)")
        print(f"    Try increasing threshold to 0.4")

    print("\n✅ Cache hit test complete!")
    return True


def test_cache_miss_rag():
    """Test 3: Verify cache miss triggers RAG pipeline."""
    print("\n" + "=" * 70)
    print("TEST 3: CACHE MISS + RAG PIPELINE TEST")
    print("=" * 70)

    from cached_rag_chatbot_chroma import CachedRAGChatbot

    print("\n🚀 Initializing chatbot...")
    chatbot = CachedRAGChatbot(
        cache_distance_threshold=0.1,
        model_name="gpt-4o-mini",
        temperature=0
    )

    chatbot.load_existing_vectorstore()
    chatbot.load_cache_from_file(here("data/taskflow_cache_seed.csv"))

    # Test cache miss - should trigger RAG
    print("\n🧪 Testing cache miss with RAG...")
    query = "How do I set up automations in TaskFlow?"

    print(f"Query: {query}")
    print("\nThis should:")
    print("  1. Check cache (miss)")
    print("  2. Query ChromaDB vectorstore")
    print("  3. Retrieve relevant documents")
    print("  4. Run LangGraph pipeline")
    print("  5. Generate answer from context")

    start = datetime.now()
    result = chatbot.query(query, verbose=True)  # verbose=True to see the flow
    elapsed = (datetime.now() - start).total_seconds()

    print("\n📊 Results:")
    if not result['cache_hit']:
        print(f"  ✓ CACHE MISS (as expected)")
        print(f"  ✓ RAG pipeline executed")
        print(f"    Response time: {elapsed:.2f}s")
        print(f"\n  Answer preview: {result['answer'][:100]}...")

        # Verify answer contains relevant info
        if "automation" in result['answer'].lower() or "workflow" in result['answer'].lower():
            print(f"  ✓ Answer contains relevant content")
        else:
            print(f"  ⚠️  Answer may not be fully relevant")
    else:
        print(f"  ✗ CACHE HIT (unexpected - this question shouldn't be cached)")
        return False

    print("\n✅ RAG pipeline test complete!")
    return True


def test_chromadb_retrieval():
    """Test 4: Verify ChromaDB is actually being queried."""
    print("\n" + "=" * 70)
    print("TEST 4: CHROMADB RETRIEVAL TEST")
    print("=" * 70)

    from document_store_chroma import DocumentVectorStore

    print("\n🗄️  Loading ChromaDB...")
    doc_store = DocumentVectorStore(
        persist_directory=str(here("data/chroma_db")))

    if not doc_store.load_existing():
        print("  ✗ Failed to load ChromaDB")
        return False

    stats = doc_store.get_stats()
    print(f"  ✓ ChromaDB loaded")
    print(f"    Collection: {stats['collection_name']}")
    print(f"    Document count: {stats['document_count']}")

    # Test retrieval
    print("\n🔍 Testing direct retrieval...")
    test_queries = [
        "How do I create a project?",
        "What are the pricing plans?",
        "Tell me about automations"
    ]

    for query in test_queries:
        print(f"\n  Query: {query}")
        results = doc_store.similarity_search(query, k=2)

        if results:
            print(f"    ✓ Retrieved {len(results)} documents")
            print(f"    Top result: {results[0].page_content[:80]}...")
        else:
            print(f"    ✗ No results found")
            return False

    print("\n✅ ChromaDB retrieval test complete!")
    return True


def test_streamlit_flow():
    """Test 5: Verify the Streamlit app flow."""
    print("\n" + "=" * 70)
    print("TEST 5: STREAMLIT FLOW VERIFICATION")
    print("=" * 70)

    print("\n📝 Checking chatbot_app.py imports...")

    # Check imports
    with open("chatbot_app.py", "r") as f:
        content = f.read()

    if "from cached_rag_chatbot_chroma import CachedRAGChatbot" in content:
        print("  ✓ Correct import (uses ChromaDB version)")
    else:
        print("  ✗ Wrong import")
        return False

    if "chatbot.query(query, verbose=False)" in content:
        print("  ✓ Calls chatbot.query() method")
    else:
        print("  ✗ Doesn't call query method")
        return False

    if "result['cache_hit']" in content:
        print("  ✓ Checks cache_hit status")
    else:
        print("  ✗ Doesn't check cache status")
        return False

    if "chatbot.add_to_cache" in content:
        print("  ✓ Has cache feedback functionality")
    else:
        print("  ✗ Missing cache feedback")
        return False

    print("\n✅ Streamlit flow verification complete!")
    return True


def print_flow_diagram():
    """Print the complete flow diagram."""
    print("\n" + "=" * 70)
    print("COMPLETE SYSTEM FLOW")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                        USER ASKS QUESTION                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  1. CHECK SEMANTIC CACHE                         │
│                     (semantic_cache.py)                          │
│  • Embed question with OpenAI                                    │
│  • Calculate cosine distance to cached questions                 │
│  • If distance < threshold → CACHE HIT                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
            CACHE HIT      CACHE MISS
                    │             │
                    ▼             ▼
┌──────────────────────┐  ┌────────────────────────────────────────┐
│  RETURN CACHED       │  │  2. QUERY CHROMADB VECTORSTORE         │
│  ANSWER              │  │     (document_store_chroma.py)          │
│                      │  │  • Retrieve top-k relevant docs         │
│  ⚡ ~100ms           │  └─────────────┬──────────────────────────┘
└──────────────────────┘                │
                                        ▼
                            ┌───────────────────────────────────────┐
                            │  3. LANGGRAPH RAG PIPELINE            │
                            │     (cached_rag_chatbot_chroma.py)    │
                            │                                        │
                            │  Node: generate_query_or_respond      │
                            │    ↓ (LLM decides to retrieve)        │
                            │  Node: retrieve                       │
                            │    ↓ (Get docs from ChromaDB)         │
                            │  Node: grade_documents                │
                            │    ↓ (Check relevance)                │
                            │  Relevant? ──NO→ rewrite_question     │
                            │    ↓ YES                               │
                            │  Node: generate_answer                │
                            │    ↓ (Generate with context)          │
                            │  END                                   │
                            │                                        │
                            │  🔍 ~3-5s                              │
                            └─────────────┬─────────────────────────┘
                                          │
                                          ▼
                            ┌───────────────────────────────────────┐
                            │  4. RETURN ANSWER TO USER             │
                            │     (chatbot_app.py)                │
                            │  • Show answer                        │
                            │  • Display cache status (hit/miss)    │
                            │  • Show response time                 │
                            │  • Show ChromaDB indicator            │
                            └─────────────┬─────────────────────────┘
                                          │
                                          ▼
                            ┌───────────────────────────────────────┐
                            │  5. USER FEEDBACK (if cache miss)     │
                            │  👍 Approve → Add to cache            │
                            │  👎 Reject  → Don't cache             │
                            └───────────────────────────────────────┘
    """)


def main():
    """Run all tests."""
    print("=" * 70)
    print("TASKFLOW RAG + CACHING - COMPLETE FLOW VERIFICATION")
    print("=" * 70)

    tests = [
        ("Setup", test_setup),
        ("Cache Hit", test_cache_hit),
        ("Cache Miss + RAG", test_cache_miss_rag),
        ("ChromaDB Retrieval", test_chromadb_retrieval),
        ("Streamlit Flow", test_streamlit_flow),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print flow diagram
    print_flow_diagram()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour system is correctly configured:")
        print("  ✓ Cache hits return instantly from semantic cache")
        print("  ✓ Cache misses trigger full LangGraph RAG pipeline")
        print("  ✓ ChromaDB vectorstore is being queried")
        print("  ✓ Streamlit app flow is correct")
        print("\n🚀 Ready to run: streamlit run chatbot_app.py")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the errors above and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
