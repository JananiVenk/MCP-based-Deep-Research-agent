import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import chromadb
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# --- Setup ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

SAMPLE_DOCUMENTS = [
    {"text": "OpenAI released a new industrial policy blueprint with AI development proposals.", "source": "NewsAPI"},
    {"text": "Sam Altman is making political moves to appeal to Democrats at OpenAI.", "source": "NewsAPI"},
    {"text": "OpenAI is positioning itself politically amid controversy over its corporate structure.", "source": "NewsAPI"},
    {"text": "Transformer architecture research shows self-attention mechanisms improve NLP tasks.", "source": "arXiv"},
    {"text": "Latest transformer model research advances in natural language processing and text generation.", "source": "arXiv"},
    {"text": "Transformers primarily learn correlations not causations according to new research.", "source": "arXiv"},
    {"text": "Large language models based on transformer architecture show emergent capabilities.", "source": "arXiv"},
    {"text": "Netflix viewership patterns vary significantly across different countries.", "source": "arXiv"},
    {"text": "Netflix autoplay feature affects watching behavior according to new study.", "source": "arXiv"},
]

def setup_collection(collection_name: str):
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        embedding = embedding_model.encode(doc["text"]).tolist()
        collection.add(
            documents=[doc["text"]],
            embeddings=[embedding],
            metadatas=[{"source": doc["source"]}],
            ids=[f"doc_{i}"]
        )
    return collection

def version_a(distances: list[float], threshold: float = 0.7) -> bool:
    """Version A: average distance threshold (original broken approach)"""
    avg_distance = sum(distances) / len(distances)
    return avg_distance > threshold

def version_b(distances: list[float], threshold: float = 1.1) -> bool:
    """Version B: min distance threshold (dynamic approach)"""
    return min(distances) > threshold

def get_distances(collection, query: str, n_results: int = 5) -> list[float]:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["distances"]
    )
    return results["distances"][0]

# --- Test queries ---
# should_fallback=True means DuckDuckGo should be triggered
TEST_QUERIES = [
    {"query": "What is happening with OpenAI?",         "should_fallback": False},
    {"query": "What are the best Netflix shows?",       "should_fallback": True},
    {"query": "Latest transformer research papers",     "should_fallback": False},
    {"query": "Best restaurants in San Francisco",      "should_fallback": True},
]

# --- Tests ---

class TestABThreshold:

    def setup_method(self):
        self.collection = setup_collection("ab_test_collection")

    def test_openai_query_should_not_fallback(self):
        """OpenAI query has relevant NewsAPI docs — neither version should trigger fallback"""
        query = "What is happening with OpenAI?"
        distances = get_distances(self.collection, query)

        result_a = version_a(distances)
        result_b = version_b(distances)

        print(f"\n[OpenAI query]")
        print(f"  distances: {[round(d, 3) for d in distances]}")
        print(f"  avg={round(sum(distances)/len(distances), 3)}, min={round(min(distances), 3)}")
        print(f"  Version A fallback: {result_a} (expected: False)")
        print(f"  Version B fallback: {result_b} (expected: False)")

        assert result_b == False, f"Version B incorrectly triggered fallback for OpenAI query (min={min(distances):.3f})"

    def test_netflix_query_should_fallback(self):
        """Netflix show recommendations have no relevant docs — should trigger fallback"""
        query = "What are the best Netflix shows to watch?"
        distances = get_distances(self.collection, query)

        result_a = version_a(distances)
        result_b = version_b(distances)

        print(f"\n[Netflix query]")
        print(f"  distances: {[round(d, 3) for d in distances]}")
        print(f"  avg={round(sum(distances)/len(distances), 3)}, min={round(min(distances), 3)}")
        print(f"  Version A fallback: {result_a} (expected: True)")
        print(f"  Version B fallback: {result_b} (expected: True)")

        assert result_b == True, f"Version B failed to trigger fallback for Netflix query (min={min(distances):.3f})"

    def test_transformer_research_should_not_fallback(self):
        """Transformer research query has relevant arXiv docs — should not trigger fallback"""
        query = "Latest transformer research"
        distances = get_distances(self.collection, query)

        result_a = version_a(distances)
        result_b = version_b(distances)

        print(f"\n[Transformer research query]")
        print(f"  distances: {[round(d, 3) for d in distances]}")
        print(f"  avg={round(sum(distances)/len(distances), 3)}, min={round(min(distances), 3)}")
        print(f"  Version A fallback: {result_a} (expected: False)")
        print(f"  Version B fallback: {result_b} (expected: False)")

        assert result_b == False, f"Version B incorrectly triggered fallback for transformer query (min={min(distances):.3f})"

    def test_restaurants_query_should_fallback(self):
        """Restaurant query has no relevant docs — should trigger fallback"""
        query = "Best restaurants in San Francisco"
        distances = get_distances(self.collection, query)

        result_a = version_a(distances)
        result_b = version_b(distances)

        print(f"\n[Restaurants query]")
        print(f"  distances: {[round(d, 3) for d in distances]}")
        print(f"  avg={round(sum(distances)/len(distances), 3)}, min={round(min(distances), 3)}")
        print(f"  Version A fallback: {result_a} (expected: True)")
        print(f"  Version B fallback: {result_b} (expected: True)")

        assert result_b == True, f"Version B failed to trigger fallback for restaurants query (min={min(distances):.3f})"

    def test_version_b_outperforms_version_a(self):
        """Version B should make correct fallback decisions more often than Version A"""
        collection = self.collection
        version_a_correct = 0
        version_b_correct = 0

        print("\n[AB Comparison Summary]")
        for test in TEST_QUERIES:
            distances = get_distances(collection, test["query"])
            avg = sum(distances) / len(distances)
            minimum = min(distances)

            a_result = version_a(distances)
            b_result = version_b(distances)

            a_correct = a_result == test["should_fallback"]
            b_correct = b_result == test["should_fallback"]

            if a_correct:
                version_a_correct += 1
            if b_correct:
                version_b_correct += 1

            print(f"  Query: '{test['query'][:40]}...'")
            print(f"    avg={avg:.3f} min={minimum:.3f} | A={'✓' if a_correct else '✗'} B={'✓' if b_correct else '✗'}")

        print(f"\n  Version A accuracy: {version_a_correct}/{len(TEST_QUERIES)}")
        print(f"  Version B accuracy: {version_b_correct}/{len(TEST_QUERIES)}")

        assert version_b_correct >= version_a_correct, \
            f"Version B ({version_b_correct}) did not outperform Version A ({version_a_correct})"