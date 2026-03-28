"""Integration tests for MiniMax LLM provider.

These tests verify the MiniMax provider works end-to-end with the actual API.
Requires MINIMAX_API_KEY environment variable to be set.

Run with: MINIMAX_API_KEY=your_key python -m pytest tests/test_minimax_integration.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def test_minimax_chat_completion(self):
        """MiniMax M2.7 should return a valid chat completion."""
        from langchain.chat_models import ChatOpenAI

        llm = ChatOpenAI(
            model="MiniMax-M2.7",
            temperature=0.01,
            max_tokens=50,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base="https://api.minimax.io/v1",
        )
        response = llm.predict("Say 'hello' and nothing else.")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_minimax_highspeed_model(self):
        """MiniMax-M2.7-highspeed should also work."""
        from langchain.chat_models import ChatOpenAI

        llm = ChatOpenAI(
            model="MiniMax-M2.7-highspeed",
            temperature=0.5,
            max_tokens=50,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base="https://api.minimax.io/v1",
        )
        response = llm.predict("Reply with 'ok'.")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_minimax_via_get_llm(self):
        """MiniMax via get_llm-equivalent ChatOpenAI should work end-to-end."""
        from langchain.chat_models import ChatOpenAI

        # Replicate the get_llm logic for MiniMax
        model_name = "MiniMax|MiniMax-M2.7|None"
        splits = model_name.split("|")
        temperature = 0
        minimax_temp = max(temperature, 0.01) if temperature <= 0 else min(temperature, 1.0)

        llm = ChatOpenAI(
            model=splits[1],
            temperature=minimax_temp,
            max_tokens=50,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base="https://api.minimax.io/v1",
        )
        response = llm.predict("Say 'integration test passed' and nothing else.")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()
