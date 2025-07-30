from datasets import Dataset
import os
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Union

# Tokenization function
def tokenize_function(examples : Dict[str, List[str]], tokenizer: PreTrainedTokenizerBase,max_input_len :int=64, max_target_len:int=64)-> Dict[str, Union[List[int], Dict[str, List[int]]]]:
    """
    Tokenizes input and target text pairs using the provided tokenizer.

    Args:
        examples (dict): A dictionary with keys "input_text" and "target_text" containing lists of strings.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_input_len (int): Maximum length for input sequences.
        max_target_len (int): Maximum length for target sequences.

    Returns:
        dict: A dictionary containing tokenized model inputs and masked target labels.
    """
    # Handling single strings
    # Wrap single strings into lists for batch processing
    inputs = examples["input_text"]
    if isinstance(inputs, str):
        inputs = [inputs]
    targets = examples["target_text"]
    if isinstance(targets, str):
        targets = [targets]

    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_len,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=max_target_len,
            padding="max_length",
            truncation=True
        )

        # Fix for single example: ensure labels["input_ids"] is a list of lists
        if len(labels["input_ids"]) > 0 and isinstance(labels["input_ids"][0], int):
            labels["input_ids"] = [labels["input_ids"]]

    # Mask out padding tokens in labels
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_dataset(data:List[Dict[str, str]],tokenizer)-> Dataset:
    """
    Converts a list of examples into a Hugging Face Dataset and tokenizes it.

    Args:
        data (List[Dict[str, str]]): A list of dictionaries with "input_text" and "target_text" keys.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to apply to the dataset.

    Returns:
        Dataset: A tokenized Hugging Face Dataset object.
    """
    # Convert list of dicts to Hugging Face Dataset
    dataset = Dataset.from_list(data)
    
    # Tokenize using map
    tokenized_data = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    return tokenized_data

def save_tokenized_dataset(dataset : Dataset,string :str)-> None:
    """
    Saves a tokenized Hugging Face Dataset to disk in a "data" directory one level up.

    Args:
        dataset (Dataset): The tokenized dataset to save.
        split_name (str): Either 'train' or 'dev' to determine save path.
        data_root (str): Optional root directory to use instead of default.
    """
    #Location of the data files
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    if 'train' in string:
        data_path_tokenized = os.path.join(DATA_DIR, "train_tokenize")
    else :
        data_path_tokenized = os.path.join(DATA_DIR, "dev_tokenize")
    #save tokenize data
    dataset.save_to_disk(data_path_tokenized)