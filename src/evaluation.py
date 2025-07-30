from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from typing import List
import os
import json


def compute_bleu(preds: List[str], targets: List[str]) -> float:
    """
    Compute the BLEU score for a set of predictions against target references.

    Args:
        preds (List[str]): List of predicted strings.
        targets (List[str]): List of reference target strings.

    Returns:
        float: The corpus-level BLEU score (0 to 1).
    """
    references = [[t.split()] for t in targets]
    candidates = [p.split() for p in preds]
    return corpus_bleu(references, candidates)


def compute_rouge(preds: List[str], targets: List[str]) -> dict:
    """
    Compute ROUGE-1 and ROUGE-L F1 scores for predictions vs targets.

    Args:
        preds (List[str]): List of predicted strings.
        targets (List[str]): List of reference target strings.

    Returns:
        dict: Dictionary with keys "rouge1" and "rougeL", each mapping to the average F1 score.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(t, p) for t, p in zip(targets, preds)]
    return {
        "rouge1": sum(s['rouge1'].fmeasure for s in scores) / len(scores),
        "rougeL": sum(s['rougeL'].fmeasure for s in scores) / len(scores)
    }

def generate_and_save_predictions(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset_path: str,
    output_path: str
) -> None:
    """
    Generate predictions from a trained model and save them alongside inputs and targets.

    Args:
        model_path (str): Path to saved model directory.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used during training.
        eval_dataset_path (str): Path to tokenized evaluation dataset.
        output_path (str): Directory where the predictions JSON will be saved.

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "Model rephrasing predictions.json")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print(f'We have loaded the pretrained model...')
    model.eval()

    dataset = load_from_disk(eval_dataset_path)
    inputs = dataset["input_text"]
    targets = dataset["target_text"]

    predictions = []
    for input_text in inputs:
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        output_ids = model.generate(**input_ids, max_length=64)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(decoded)

    output_data = [
        {"input": i, "target": t, "prediction": p}
        for i, t, p in zip(inputs, targets, predictions)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Predictions saved to {output_file}")

def evaluate_and_save_metrics(model_path: str, tokenizer: PreTrainedTokenizerBase, eval_dataset_path: str,output_path: str) -> None:
    """
    Evaluate a trained Seq2Seq model using BLEU and ROUGE scores.

    This function:
        - Loads the fine-tuned model and tokenizer
        - Loads the tokenized evaluation dataset from disk
        - Generates predictions using greedy decoding
        - Computes BLEU and ROUGE scores
        - Prints evaluation metrics to stdout

    Args:
        model_path (str): Path to the directory containing the saved model.
        tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer used during training.
        eval_dataset_path (str): Path to the tokenized evaluation dataset on disk.
        output_path (str): Directory to save the evaluation metrics.

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)
    metrics_file = os.path.join(output_path, "Validation evaluation metrics.txt")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print(f'We have loaded the pretrained model...')
    model.eval()
    eval_dataset = load_from_disk(eval_dataset_path)
    input_texts = eval_dataset['input_text']
    target_texts = eval_dataset['target_text']

    predictions = []
    for input_text in input_texts:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        output_ids = model.generate(**inputs, max_length=64)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(prediction)

    bleu_score = compute_bleu(predictions, target_texts)
    rouge_scores = compute_rouge(predictions, target_texts)

    #Printing results
    print(f"\nEvaluation Results:")
    print(f"BLEU Score   : {bleu_score:.4f}")
    print(f"ROUGE-1 F1   : {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-L F1   : {rouge_scores['rougeL']:.4f}")


    # Write metrics to a .txt file
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("Evaluation Metrics\n")
        f.write("==================\n")
        f.write(f"BLEU Score   : {bleu_score:.4f}\n")
        f.write(f"ROUGE-1 F1   : {rouge_scores['rouge1']:.4f}\n")
        f.write(f"ROUGE-L F1   : {rouge_scores['rougeL']:.4f}\n")

    print(f"Evaluation metrics saved to {metrics_file}")