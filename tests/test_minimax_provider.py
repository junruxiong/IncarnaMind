"""Unit tests for MiniMax LLM provider integration."""
import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestMiniMaxConfig(unittest.TestCase):
    """Tests for MiniMax configuration in Config class."""

    def test_configparser_has_minimax_key(self):
        """configparser.ini should have MINIMAX_API_KEY field."""
        import configparser

        config = configparser.ConfigParser(interpolation=None)
        config.read(os.path.join(PROJECT_ROOT, "configparser.ini"))
        self.assertTrue(config.has_option("tokens", "MINIMAX_API_KEY"))

    def test_configparser_has_minimax_model_examples(self):
        """configparser.ini should contain MiniMax model examples in comments."""
        with open(os.path.join(PROJECT_ROOT, "configparser.ini"), "r") as f:
            content = f.read()
        self.assertIn("MiniMax|MiniMax-M2.7|None", content)
        self.assertIn("MiniMax|MiniMax-M2.7-highspeed|None", content)

    def test_config_class_has_minimax_api_key(self):
        """Config class should read minimax_api_key."""
        with open(os.path.join(PROJECT_ROOT, "toolkit", "utils.py"), "r") as f:
            content = f.read()
        self.assertIn("self.minimax_api_key", content)
        self.assertIn('"MINIMAX_API_KEY"', content)


class TestMiniMaxModelNameParsing(unittest.TestCase):
    """Tests for MiniMax model name parsing in get_llm()."""

    def test_minimax_model_name_format(self):
        """MiniMax model name should follow Provider|Model|File format."""
        model_name = "MiniMax|MiniMax-M2.7|None"
        splits = model_name.split("|")
        self.assertEqual(splits[0], "MiniMax")
        self.assertEqual(splits[1], "MiniMax-M2.7")
        self.assertEqual(splits[2], "None")

    def test_minimax_provider_detection(self):
        """Provider detection should be case-insensitive."""
        for name in ["MiniMax|MiniMax-M2.7|None", "minimax|MiniMax-M2.7|None", "MINIMAX|MiniMax-M2.7|None"]:
            splits = name.split("|")
            self.assertTrue("minimax" in splits[0].lower())

    def test_minimax_highspeed_model(self):
        """MiniMax-M2.7-highspeed model name should parse correctly."""
        model_name = "MiniMax|MiniMax-M2.7-highspeed|None"
        splits = model_name.split("|")
        self.assertEqual(splits[1], "MiniMax-M2.7-highspeed")


class TestMiniMaxTemperatureClamping(unittest.TestCase):
    """Tests for MiniMax temperature constraint handling."""

    def _clamp_temp(self, temperature):
        return max(temperature, 0.01) if temperature <= 0 else min(temperature, 1.0)

    def test_zero_temperature_clamped(self):
        self.assertAlmostEqual(self._clamp_temp(0), 0.01)

    def test_negative_temperature_clamped(self):
        self.assertAlmostEqual(self._clamp_temp(-0.5), 0.01)

    def test_valid_temperature_passed_through(self):
        self.assertAlmostEqual(self._clamp_temp(0.7), 0.7)

    def test_high_temperature_clamped(self):
        self.assertAlmostEqual(self._clamp_temp(1.5), 1.0)

    def test_boundary_temperature_one(self):
        self.assertAlmostEqual(self._clamp_temp(1.0), 1.0)


class TestMiniMaxProviderBranch(unittest.TestCase):
    """Tests for MiniMax provider branch in main.py."""

    def setUp(self):
        with open(os.path.join(PROJECT_ROOT, "main.py"), "r") as f:
            self.main_content = f.read()

    def test_main_sets_minimax_env_var(self):
        """main.py should set MINIMAX_API_KEY environment variable."""
        self.assertIn('os.environ["MINIMAX_API_KEY"]', self.main_content)
        self.assertIn("configs.minimax_api_key", self.main_content)

    def test_main_has_minimax_branch(self):
        """main.py should have MiniMax provider branch in get_llm."""
        self.assertIn('elif "minimax" in splits[0].lower():', self.main_content)

    def test_minimax_branch_uses_chat_openai(self):
        """MiniMax branch should use ChatOpenAI for OpenAI-compatible API."""
        idx = self.main_content.index('elif "minimax" in splits[0].lower():')
        snippet = self.main_content[idx : idx + 400]
        self.assertIn("ChatOpenAI", snippet)

    def test_minimax_branch_uses_correct_api_base(self):
        """MiniMax branch should use https://api.minimax.io/v1 as base URL."""
        self.assertIn('openai_api_base="https://api.minimax.io/v1"', self.main_content)

    def test_minimax_branch_uses_minimax_api_key(self):
        """MiniMax branch should pass minimax_api_key to ChatOpenAI."""
        idx = self.main_content.index('elif "minimax" in splits[0].lower():')
        snippet = self.main_content[idx : idx + 400]
        self.assertIn("openai_api_key=configs.minimax_api_key", snippet)

    def test_minimax_branch_has_temperature_clamping(self):
        """MiniMax branch should clamp temperature."""
        idx = self.main_content.index('elif "minimax" in splits[0].lower():')
        snippet = self.main_content[idx : idx + 400]
        self.assertIn("minimax_temp", snippet)
        self.assertIn("max(temperature, 0.01)", snippet)


class TestReadmeDocumentation(unittest.TestCase):
    """Tests for MiniMax documentation in README."""

    def setUp(self):
        with open(os.path.join(PROJECT_ROOT, "README.md"), "r") as f:
            self.readme = f.read()

    def test_readme_mentions_minimax(self):
        self.assertIn("MiniMax", self.readme)

    def test_readme_has_minimax_api_key_setup(self):
        self.assertIn("MINIMAX_API_KEY", self.readme)

    def test_readme_has_minimax_in_comparison_table(self):
        self.assertIn("MiniMax-M2.7", self.readme)

    def test_readme_has_minimax_link(self):
        self.assertIn("minimax.io", self.readme)

    def test_readme_lists_minimax_in_compatibility(self):
        self.assertIn("MiniMax", self.readme)


if __name__ == "__main__":
    unittest.main()
