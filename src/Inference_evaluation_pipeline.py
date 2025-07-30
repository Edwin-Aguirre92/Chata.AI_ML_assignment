from transformers import AutoTokenizer
from src.evaluation import generate_and_save_predictions, evaluate_and_save_metrics
import os
from src.config import model_used


def main() -> None:
    """
    Pipeline responsible for running inference and evaluation on a trained question rephrasing model.

    This pipeline performs the following steps:
        1. Loads the tokenizer used during training.
        2. Loads the trained model from disk.
        3. Loads the tokenized evaluation dataset from disk.
        4. Generates predictions and saves them in disk.
        5. Computes BLEU and ROUGE metrics and saves them in disk.

    Returns:
        None
    """

    print("Starting the Inference and evaluation pipeline...")

    # Define paths to the train model
    OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    #MODEL_DIR = os.path.join(OUTPUTS_DIR, "model",f"{model_used}_2025-07-28_02_48PM")
    MODEL_DIR = os.path.join(OUTPUTS_DIR, "model",model_used)
    #Location of the evaluation data that is tokenized already
    EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dev_tokenize")
    #Directory to save the predictions
    OUTPUT_PRED_DIR = os.path.join(OUTPUTS_DIR, "predictions")

    print(f'The model dir is :{MODEL_DIR}')

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f'We have loaded the pretrained tokenizer...')

    # 2. Run prediction and evaluation
    generate_and_save_predictions(MODEL_DIR, tokenizer, EVAL_DIR, OUTPUT_PRED_DIR)
    evaluate_and_save_metrics(MODEL_DIR, tokenizer, EVAL_DIR, OUTPUT_PRED_DIR)

    print("Inference and evaluation pipeline completed!")


if __name__ == "__main__":
    main()