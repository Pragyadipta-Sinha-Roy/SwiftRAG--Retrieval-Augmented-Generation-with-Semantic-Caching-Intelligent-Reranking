"""
RAG Chatbot with Semantic Caching using ChromaDB - FIXED VERSION
Uses LangGraph to orchestrate a RAG workflow with semantic caching layer.

FIXES:
- Added recursion limit configuration
- Added max retry logic to prevent infinite loops
- Fallback answer when documents aren't relevant
"""

from typing import Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_groq import ChatGroq
import operator
from pyprojroot import here

from semantic_cache import SemanticCache
from document_store_chroma import DocumentVectorStore


# Extended state to track retries
class ExtendedState(MessagesState):
    """Extended state with retry counter."""
    retry_count: Annotated[int, operator.add] = 0


class CachedRAGChatbot:
    """RAG Chatbot with semantic caching capability using ChromaDB."""

    def __init__(
        self,
        cache_distance_threshold: float = 0.1,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0,
        chroma_persist_dir: str = str(here("data/chroma_db")),
        chroma_collection: str = "taskflow_docs",
        max_retries: int = 2  # Maximum question rewrites
    ):
        """
        Initialize the cached RAG chatbot.

        Args:
            cache_distance_threshold: Threshold for semantic cache matching
            model_name: LLM model to use
            temperature: Temperature for LLM
            chroma_persist_dir: ChromaDB persistence directory
            chroma_collection: ChromaDB collection name
            max_retries: Maximum number of question rewrite attempts
        """
        self.cache = SemanticCache(distance_threshold=cache_distance_threshold)
        self.doc_store = DocumentVectorStore(
            persist_directory=chroma_persist_dir,
            collection_name=chroma_collection
        )
        self.response_model = ChatGroq(model=model_name, temperature=temperature)
        self.grader_model = ChatGroq(model=model_name, temperature=0)
        self.max_retries = max_retries

        # Will be set when graph is compiled
        self.graph = None
        self.retriever_tool = None

    def load_existing_vectorstore(self) -> bool:
        """
        Load existing ChromaDB vectorstore.

        Returns:
            True if loaded successfully, False otherwise
        """
        success = self.doc_store.load_existing()
        if success:
            self._setup_graph()
        return success

    def load_cache_pairs(self, pairs: list[tuple[str, str]]):
        """Load Q&A pairs into the semantic cache."""
        self.cache.hydrate_from_pairs(pairs)

    def load_cache_from_file(self, filepath: str):
        """Load cache from a CSV file."""
        import csv
        pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row['question'], row['answer']))
        self.cache.hydrate_from_pairs(pairs)

    def save_cache_to_file(self, filepath: str):
        """Save cache to a CSV file."""
        self.cache.save_to_file(filepath)

    def get_vectorstore_stats(self) -> dict:
        """Get statistics about the vectorstore."""
        return self.doc_store.get_stats()

    def _setup_graph(self):
        """Setup the LangGraph workflow with retry logic."""
        # Create retriever tool
        retriever = self.doc_store.get_retriever(k=4)

        @tool
        def retrieve_documents(query: str) -> str:
            """Search and return relevant information from the TaskFlow documentation."""
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        self.retriever_tool = retrieve_documents

        # Define node functions
        def generate_query_or_respond(state: ExtendedState):
            """Generate a response or decide to retrieve documents."""
            response = (
                self.response_model
                .bind_tools([self.retriever_tool])
                .invoke(state["messages"])
            )
            return {"messages": [response]}

        # Grade documents prompt and model
        GRADE_PROMPT = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )

        class GradeDocuments(BaseModel):
            """Grade documents using a binary score for relevance check."""
            binary_score: str = Field(
                description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
            )

        def grade_documents(
            state: ExtendedState,
        ) -> Literal["generate_answer", "rewrite_question", "generate_fallback"]:
            """Determine whether the retrieved documents are relevant to the question."""
            question = state["messages"][0].content
            context = state["messages"][-1].content
            retry_count = state.get("retry_count", 0)

            # Check if we've hit max retries
            if retry_count >= self.max_retries:
                print(
                    f"⚠️  Max retries ({self.max_retries}) reached. Generating fallback answer...")
                return "generate_fallback"

            prompt = GRADE_PROMPT.format(question=question, context=context)
            response = (
                self.grader_model
                .with_structured_output(GradeDocuments)
                .invoke([{"role": "user", "content": prompt}])
            )
            score = response.binary_score if hasattr(response, "binary_score") else response.get("binary_score", "no")

            if score == "yes":
                return "generate_answer"
            else:
                print(
                    f"❌ Documents not relevant. Will rewrite question (attempt {retry_count + 1}/{self.max_retries})...")
                return "rewrite_question"

        # Rewrite question
        REWRITE_PROMPT = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question:"
        )

        def rewrite_question(state: ExtendedState):
            """Rewrite the original user question and increment retry counter."""
            messages = state["messages"]
            question = messages[0].content
            retry_count = state.get("retry_count", 0)

            print(
                f"🔄 Rewriting question (attempt {retry_count + 1}/{self.max_retries})...")

            prompt = REWRITE_PROMPT.format(question=question)
            response = self.response_model.invoke(
                [{"role": "user", "content": prompt}])

            print(f"   New question: {response.content}")

            return {
                "messages": [HumanMessage(content=response.content)],
                "retry_count": 1  # Increment by 1
            }

        # Generate answer
        GENERATE_PROMPT = (
            "You are a helpful TaskFlow assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \n"
            "Context: {context}"
        )

        def generate_answer(state: ExtendedState):
            """Generate an answer from retrieved context."""
            question = state["messages"][0].content
            context = state["messages"][-1].content
            prompt = GENERATE_PROMPT.format(question=question, context=context)
            response = self.response_model.invoke(
                [{"role": "user", "content": prompt}])
            return {"messages": [response]}

        # Generate fallback answer when docs aren't relevant
        FALLBACK_PROMPT = (
            "You are a helpful TaskFlow assistant. "
            "A user asked the following question, but we couldn't find relevant information in our documentation.\n"
            "Question: {question}\n\n"
            "Provide a helpful response that:\n"
            "1. Politely acknowledges we don't have specific information about this in our TaskFlow documentation\n"
            "2. Suggests what they could try (contact support, check our website, rephrase question)\n"
            "3. If you can infer what they're asking about, provide general helpful context\n"
            "Keep it brief and friendly."
        )

        def generate_fallback(state: ExtendedState):
            """Generate a fallback answer when docs aren't relevant after retries."""
            question = state["messages"][0].content
            prompt = FALLBACK_PROMPT.format(question=question)
            response = self.response_model.invoke(
                [{"role": "user", "content": prompt}])
            return {"messages": [response]}

        # Build the graph
        workflow = StateGraph(ExtendedState)

        workflow.add_node("generate_query_or_respond",
                          generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("generate_answer", generate_answer)
        # New fallback node
        workflow.add_node("generate_fallback", generate_fallback)

        workflow.add_edge(START, "generate_query_or_respond")

        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            grade_documents,
            {
                "generate_answer": "generate_answer",
                "rewrite_question": "rewrite_question",
                "generate_fallback": "generate_fallback"  # New fallback path
            }
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("generate_fallback", END)  # Fallback ends workflow
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile with recursion limit
        self.graph = workflow.compile(
            checkpointer=None,
            debug=False
        )

    def query(self, question: str, verbose: bool = False) -> dict:
        """
        Query the chatbot with semantic caching.

        Args:
            question: User's question
            verbose: Whether to print detailed logs

        Returns:
            Dictionary with 'answer', 'cache_hit', and 'cache_info'
        """
        if verbose:
            print(f"\n🔍 Query: {question}")
            print("=" * 60)

        # Step 1: Check semantic cache
        cache_results = self.cache.check(question, num_results=1)

        if cache_results.hit:
            best_match = cache_results.best_match
            if verbose:
                print(f"✅ CACHE HIT!")
                print(f"   Matched question: {best_match.prompt}")
                print(f"   Distance: {best_match.vector_distance:.4f}")
                print(f"   Similarity: {best_match.cosine_similarity:.4f}")
                print(f"\n💬 Answer (from cache): {best_match.response}\n")

            return {
                "answer": best_match.response,
                "cache_hit": True,
                "cache_info": {
                    "matched_question": best_match.prompt,
                    "distance": best_match.vector_distance,
                    "similarity": best_match.cosine_similarity
                }
            }

        # Step 2: Cache miss - go through full RAG pipeline
        if verbose:
            print("❌ CACHE MISS - Running full RAG pipeline...")
            print("   Querying ChromaDB vectorstore...")

        if self.graph is None:
            raise ValueError("Graph not initialized. Load vectorstore first.")

        # Run the RAG workflow with recursion limit
        try:
            result = None
            config = {"recursion_limit": 50}  # Set reasonable recursion limit

            for chunk in self.graph.stream(
                {"messages": [{"role": "user", "content": question}],
                    "retry_count": 0},
                config=config
            ):
                for node, update in chunk.items():
                    if verbose:
                        print(f"\n📍 Node: {node}")
                    result = update

            # Extract the final answer
            answer = result["messages"][-1].content

            if verbose:
                print(f"\n💬 Answer (from RAG): {answer}\n")

            return {
                "answer": answer,
                "cache_hit": False,
                "cache_info": None
            }

        except Exception as e:
            # Graceful error handling
            error_message = str(e)
            if "recursion" in error_message.lower():
                return {
                    "answer": "I encountered an error while searching for an answer. The question might be too complex or outside our documentation scope. Please try:\n• Rephrasing your question\n• Asking about specific TaskFlow features\n• Breaking down complex questions",
                    "cache_hit": False,
                    "cache_info": None
                }
            else:
                raise e

    def add_to_cache(self, question: str, answer: str):
        """
        Manually add a Q&A pair to the cache.

        Args:
            question: The question
            answer: The answer
        """
        self.cache.add_pair(question, answer)

    def query_stream(self, question: str):
        """
        Query with streaming (for interactive use).
        Checks cache first, then streams RAG results if cache miss.

        Yields:
            Dictionary with status updates and results
        """
        # Check cache
        cache_results = self.cache.check(question, num_results=1)

        if cache_results.hit:
            best_match = cache_results.best_match
            yield {
                "type": "cache_hit",
                "answer": best_match.response,
                "cache_info": {
                    "matched_question": best_match.prompt,
                    "distance": best_match.vector_distance,
                    "similarity": best_match.cosine_similarity
                }
            }
            return

        yield {"type": "cache_miss"}

        # Run RAG pipeline
        if self.graph is None:
            raise ValueError("Graph not initialized. Load vectorstore first.")

        try:
            final_answer = None
            config = {"recursion_limit": 50}

            for chunk in self.graph.stream(
                {"messages": [{"role": "user", "content": question}],
                    "retry_count": 0},
                config=config
            ):
                for node, update in chunk.items():
                    yield {
                        "type": "node_update",
                        "node": node,
                        "messages": update["messages"]
                    }
                    final_answer = update["messages"][-1].content

            yield {
                "type": "complete",
                "answer": final_answer
            }

        except Exception as e:
            error_message = str(e)
            if "recursion" in error_message.lower():
                yield {
                    "type": "error",
                    "answer": "I encountered an error while searching. Please try rephrasing your question or asking about specific TaskFlow features."
                }
            else:
                raise e
