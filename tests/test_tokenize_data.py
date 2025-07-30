import unittest
import os
import shutil
import sys 
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) #Added path so that src/ import works 
from src.tokenize_data import tokenize_function, tokenize_dataset, save_tokenized_dataset
from src.config import model_used

class TestTokenization(unittest.TestCase):
    """
    Unit tests for tokenization-related utility functions in tokenize_data.py.
    """

    def setUp(self):
        """
        Set up sample input data and tokenizer.
        """
        self.sample_data = [
            {"input_text": "Where is the Eiffel Tower?", "target_text": "It is in Paris."},
            {"input_text": "What's your name?", "target_text": "I am ChatGPT."}
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(model_used)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Fix missing pad token

        # Expected save location for save_tokenized_dataset()
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))#Change this so that it be run from root folder 
        self.train_path = os.path.join(self.data_dir, "data", "train_tokenize")
        self.dev_path = os.path.join(self.data_dir, "data", "dev_tokenize")

        # # Ensure a clean state
        # if os.path.exists(self.train_path):
        #     shutil.rmtree(self.train_path)

    def tearDown(self):
        """
        Clean up saved dataset after test.
        """
        pass
        # if os.path.exists(self.train_path):
        #     shutil.rmtree(self.train_path)

    def test_tokenize_function(self):
        """
        Test tokenize_function returns tokenized input with masked labels.
        """
        batch_input = {
        "input_text": [self.sample_data[0]["input_text"]],
        "target_text": [self.sample_data[0]["target_text"]],
    }
        # Use small max lengths to force padding tokens
        tokenized = tokenize_function(self.sample_data[0], self.tokenizer, max_input_len=10, max_target_len=5)

        print("tokenized labels:", tokenized["labels"])  # DEBUG

        self.assertIn("input_ids", tokenized)
        self.assertIn("attention_mask", tokenized)
        self.assertIn("labels", tokenized)

        # Check at least one -100 mask in labels
        self.assertTrue(any(token == -100 for label in tokenized["labels"] for token in label))

    def test_tokenize_dataset(self):
        """
        Test tokenize_dataset returns a tokenized Dataset with correct structure.
        """
        tokenized_dataset = tokenize_dataset(self.sample_data, self.tokenizer)

        self.assertIsInstance(tokenized_dataset, Dataset)
        self.assertEqual(len(tokenized_dataset), 2)
        self.assertIn("input_ids", tokenized_dataset[0])
        self.assertIn("labels", tokenized_dataset[0])
        self.assertEqual(len(tokenized_dataset[0]["input_ids"]), 64)

    def test_save_tokenized_train_dataset(self):
        """
        Test save_tokenized_dataset saves the train dataset to the expected disk location.
        """
        tokenized_dataset = tokenize_dataset(self.sample_data, self.tokenizer)

        # Save
        save_tokenized_dataset(tokenized_dataset, string="train")

        self.assertTrue(os.path.exists(self.train_path))
        loaded = load_from_disk(self.train_path)
        self.assertEqual(len(loaded), 2)
        self.assertIn("labels", loaded[0])

    def test_save_tokenized_dev_dataset(self):
        """
        Test save_tokenized_dataset saves the dev dataset to the expected disk location.
        """
        tokenized_dataset = tokenize_dataset(self.sample_data, self.tokenizer)

        # Save
        save_tokenized_dataset(tokenized_dataset, string="dev")

        self.assertTrue(os.path.exists(self.dev_path))
        loaded = load_from_disk(self.dev_path)
        self.assertEqual(len(loaded), 2)
        self.assertIn("labels", loaded[0])


if __name__ == "__main__":
    unittest.main()