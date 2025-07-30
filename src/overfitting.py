from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import os
from typing import List

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

def compute_scores(model, tokenizer, dataset_path: str) -> dict:
    """
    Compute BLEU and ROUGE scores for model predictions on a given dataset.
    
    Args:
        model: A Hugging Face Seq2Seq model with a `.generate()` method.
        tokenizer: A tokenizer compatible with the model, used for encoding
                   inputs and decoding outputs.
        dataset_path (str): Path to the dataset stored on disk (in Hugging Face
                            datasets format), which must contain "input_text" and
                            "target_text" fields.
    
    Returns:
        dict: A dictionary containing the evaluation scores with keys:
            - "bleu": corpus-level BLEU score (float)
            - "rouge1_f1": average ROUGE-1 F1 score (float)
            - "rougeL_f1": average ROUGE-L F1 score (float)
    """
    dataset = load_from_disk(dataset_path)
    inputs = dataset["input_text"]
    targets = dataset["target_text"]

    predictions = []
    for input_text in inputs:
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        output_ids = model.generate(**input_ids, max_length=64)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(prediction)

    bleu = compute_bleu(predictions, targets)
    rouge = compute_rouge(predictions, targets)

    return {
        "bleu": bleu,
        "rouge1_f1": rouge["rouge1"],
        "rougeL_f1": rouge["rougeL"]
    }

def compare_and_save_metrics(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset_path: str,
    eval_dataset_path: str,
    output_path: str
) -> None:
    """
    Compare BLEU and ROUGE metrics on training and evaluation datasets to assess overfitting.

    Args:
        model_path (str): Path to the directory containing the trained model.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used during training.
        train_dataset_path (str): Path to tokenized training dataset.
        eval_dataset_path (str): Path to tokenized evaluation dataset.
        output_path (str): Directory to save the comparison metrics.

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)
    metrics_file = os.path.join(output_path, "overfitting_diagnostics.txt")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print(f'We have loaded the pretrained model...')
    model.eval()

    print("Computing metrics on training dataset...")
    train_scores = compute_scores(model, tokenizer, train_dataset_path)

    print("Computing metrics on validation dataset...")
    eval_scores = compute_scores(model, tokenizer, eval_dataset_path)

    print("\nOverfitting Diagnostics Results:")
    print(f"Train BLEU     : {train_scores['bleu']:.4f}")
    print(f"Eval  BLEU     : {eval_scores['bleu']:.4f}")
    print(f"Train ROUGE-1  : {train_scores['rouge1_f1']:.4f}")
    print(f"Eval  ROUGE-1  : {eval_scores['rouge1_f1']:.4f}")
    print(f"Train ROUGE-L  : {train_scores['rougeL_f1']:.4f}")
    print(f"Eval  ROUGE-L  : {eval_scores['rougeL_f1']:.4f}")

    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("Overfitting Diagnostics\n")
        f.write("========================\n\n")
        f.write("BLEU Score Comparison:\n")
        f.write(f"  Training BLEU   : {train_scores['bleu']:.4f}\n")
        f.write(f"  Validation BLEU : {eval_scores['bleu']:.4f}\n\n")

        f.write("ROUGE-1 F1 Comparison:\n")
        f.write(f"  Training ROUGE-1 F1   : {train_scores['rouge1_f1']:.4f}\n")
        f.write(f"  Validation ROUGE-1 F1 : {eval_scores['rouge1_f1']:.4f}\n\n")

        f.write("ROUGE-L F1 Comparison:\n")
        f.write(f"  Training ROUGE-L F1   : {train_scores['rougeL_f1']:.4f}\n")
        f.write(f"  Validation ROUGE-L F1 : {eval_scores['rougeL_f1']:.4f}\n\n")

        f.write("Interpretation:\n")
        if train_scores['bleu'] - eval_scores['bleu'] > 0.1:
            f.write("There may be signs of overfitting (BLEU gap > 0.1).\n")
        else:
            f.write("No significant overfitting detected based on BLEU.\n")

    print(f"\nOverfitting diagnostic metrics file saved to {metrics_file}")