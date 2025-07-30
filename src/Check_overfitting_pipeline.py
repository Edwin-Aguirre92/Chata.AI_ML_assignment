from transformers import AutoTokenizer
from src.overfitting import  compare_and_save_metrics
import os
from src.config import model_used


def main() -> None:
    """
    Pipeline responsible for running overfitting diagnostics on the trained model.

    This pipeline performs the following steps:
        1. Loads the tokenizer used during training.
        2. Loads the trained model from disk.
        3. Loads the tokenized train and evaluation dataset from disk.
        4. Generates predictions and saves them in disk.
        5. Computes BLEU and ROUGE metrics and saves them in disk.

    Returns:
        None
    """
    
    print("Starting the check for overfitting pipeline...")

    # Define paths to the trained model
    OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    MODEL_DIR = os.path.join(OUTPUTS_DIR, "model",model_used)
    #MODEL_DIR = os.path.join(OUTPUTS_DIR, "model",f"{model_used}_2025-07-28_02_48PM")
    #Location of the training data that is tokenized already
    TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train_tokenize")
    #Location of the evaluation data that is tokenized already
    EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dev_tokenize")
    #Directory to save the overfitting comparison
    OUTPUT_PRED_DIR = os.path.join(OUTPUTS_DIR, "overfitting_diagnostics")

    print(f'The model dir is :{MODEL_DIR}')

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f'We have loaded the pretrained tokenizer...')

    # 2. Run prediction and evaluation
    compare_and_save_metrics(MODEL_DIR, tokenizer,TRAIN_DIR ,EVAL_DIR, OUTPUT_PRED_DIR)

    print("Check for overfitting pipeline completed!")


if __name__ == "__main__":
    main()