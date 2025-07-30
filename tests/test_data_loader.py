import unittest
import os
import tempfile
import json
from typing import List, Dict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))#Added path so that src/ import works 
from src.data_loader import (
    load_json_file,
    convert_to_input_target,
    filter_invalid_inputs
)

class TestDataLoader(unittest.TestCase):
    """
    Unit tests for the data_loader module.
    """

    def setUp(self):
        """
        Set up a temporary JSONL test file with mixed valid and noisy data.
        This file simulates the structure expected by load_json_file().
        """
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        lines = [
            json.dumps({"q1": {"disfluent": "what is your name?", "original": "What's your name?"}}),
            json.dumps({"q2": {"disfluent": "#VALUE!", "original": "Invalid question"}}),
            json.dumps({"q3": {"disfluent": "Question n/a", "original": "No valid question"}}),
            json.dumps({"q4": {"disfluent": "QUESTION N/A", "original": "missing"}})
        ]
        self.temp_file.write("\n".join(lines))
        self.temp_file.close()

    def tearDown(self):
        """
        Clean up the temporary test file after each test method.
        """
        os.remove(self.temp_file.name)

    def test_load_json_file(self):
        """
        Test that load_json_file() correctly loads all lines from a JSONL file.
        """
        result = load_json_file(self.temp_file.name)
        self.assertEqual(len(result), 4)
        self.assertIn("q1", result[0])
        self.assertEqual(result[0]["q1"]["disfluent"], "what is your name?")

    def test_convert_to_input_target(self):
        """
        Test that convert_to_input_target() flattens the structure
        and renames keys to 'input_text' and 'target_text'.
        """
        nested_dicts = load_json_file(self.temp_file.name)
        result = convert_to_input_target(nested_dicts)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]['input_text'], "what is your name?")
        self.assertEqual(result[0]['target_text'], "What's your name?")

    def test_filter_invalid_inputs(self):
        """
        Test that filter_invalid_inputs() correctly removes noisy input_text
        containing '#VALUE!' or 'Question n/a' (case-insensitive).
        """
        nested_dicts = load_json_file(self.temp_file.name)
        flat_data = convert_to_input_target(nested_dicts)
        filtered = filter_invalid_inputs(flat_data)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['input_text'], "what is your name?")

if __name__ == "__main__":
    unittest.main()