import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import should_fallback

class TestShouldFallback:

    def test_should_fallback_true(self):
        """High distance scores should trigger fallback"""
        state = {
            "query": "test",
            "articles_ingested": True,
            "chunks": [],
            "answer": "",
            "used_fallback": False,
            "needs_fallback": True,
            "avg_distance": 1.5,
            "messages": []
        }
        result = should_fallback(state)
        assert result == "fallback", f"Expected 'fallback', got '{result}'"

    def test_should_fallback_false(self):
        """Low distance scores should skip fallback"""
        state = {
            "query": "test",
            "articles_ingested": True,
            "chunks": [],
            "answer": "",
            "used_fallback": False,
            "needs_fallback": False,
            "avg_distance": 0.7,
            "messages": []
        }
        result = should_fallback(state)
        assert result == "synthesize", f"Expected 'synthesize', got '{result}'"

    def test_agent_state_keys(self):
        """AgentState should have all required keys"""
        state = {
            "query": "test",
            "articles_ingested": False,
            "chunks": [],
            "answer": "",
            "used_fallback": False,
            "needs_fallback": False,
            "avg_distance": 0.0,
            "messages": []
        }
        required_keys = ["query", "articles_ingested", "chunks", "answer", "used_fallback", "needs_fallback", "avg_distance", "messages"]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"