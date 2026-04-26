import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import chunk_text

class TestChunkText:

    def test_chunk_text_basic(self):
        """Normal text should split into multiple chunks"""
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) > 1, "Expected multiple chunks for long text"

    def test_chunk_text_empty(self):
        """Empty string should return empty list"""
        chunks = chunk_text("", chunk_size=200)
        assert chunks == [], f"Expected empty list, got {chunks}"

    def test_chunk_text_short(self):
        """Text shorter than chunk size should return single chunk"""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"