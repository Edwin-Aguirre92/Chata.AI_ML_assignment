import json
import os
from typing import List,Dict,Tuple

#Load json files
def load_json_file(file_path : str)->List[Dict]:
    """
    Load a JSON file where each line is a separate JSON object.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List[Dict]: A list of parsed JSON objects from the file.
    """
    
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f.readlines()]


# Flatten into Hugging Face compatible format
def convert_to_input_target(example_dicts : List[Dict])-> List[Dict[str,str]]:
    """
    Convert nested dictionary format to a flat list of dictionaries 
    with 'input_text' and 'target_text' keys for Hugging Face compatibility.

    Args:
        example_dicts (List[Dict]): List of nested dictionaries with question pairs.

    Returns:
        List[Dict[str, str]]: List of dictionaries each containing 
        a disfluent 'input_text' and its corresponding fluent 'target_text'.
    """
    result = []
    for item in example_dicts:
        for _, v in item.items():
            result.append({
                "input_text": v["disfluent"],
                "target_text": v["original"]
            })
    return result

def filter_invalid_inputs(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filter out examples where the 'input_text' contains invalid placeholders
    such as '#VALUE!' or 'question n/a', case-insensitively.

    Args:
        data (List[Dict[str, str]]): List of dictionaries containing
        'input_text' and 'target_text'.

    Returns:
        List[Dict[str, str]]: Filtered list excluding invalid entries.
    """
    filtered = []
    for example in data:
        input_text = example["input_text"].strip().lower()
        if "#value!" not in input_text and "question n/a" not in input_text:
            filtered.append(example)
    return filtered

def load_data(EXCLUDE_NOISE_QUESTIONS: bool)-> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load and preprocess training and evaluation datasets from JSON files.Optionally filters out disfluent
    input questions containing noise like '#VALUE!' or 'question n/a'.

    Args:
        EXCLUDE_NOISE_QUESTIONS (bool): Whether to exclude noisy inputs from the datasets.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: A tuple containing
        training and evaluation datasets, formatted with 'input_text' and 'target_text'.
    """
    #Location of the data files
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    train_path = os.path.join(DATA_DIR, "train.json")
    val_path = os.path.join(DATA_DIR, "dev.json")
    # Load raw dict format
    raw_train = load_json_file(train_path)
    raw_eval = load_json_file(val_path)
    #Flatten files to Hugging face
    train_examples = convert_to_input_target(raw_train)
    eval_examples = convert_to_input_target(raw_eval)
    # Optionally filter out noisy inputs
    if EXCLUDE_NOISE_QUESTIONS:
        train_examples = filter_invalid_inputs(train_examples)
        eval_examples = filter_invalid_inputs(eval_examples)

    return train_examples,eval_examples