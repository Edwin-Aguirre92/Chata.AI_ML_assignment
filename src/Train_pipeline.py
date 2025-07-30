from src.data_loader import load_data
from src.tokenizer_setup import get_tokenizer
from src.tokenize_data import tokenize_dataset, save_tokenized_dataset
from src.train import train_model
from src.config import EXCLUDE_NOISE_QUESTIONS

def main() -> None:
    """
    Pipeline responsible for training a question rephrasing model.

    This pipeline performs the following steps:

        1. Loads training and evaluation data.
        2. Initializes the tokenizer.
        3. Tokenizes the datasets.
        4. Saves tokenized datasets to disk
        5. Trains the model using the prepared datasets.

    Returns:
        None
    """
    
    print("Starting the train pipeline...")

    # 1. Load raw data and possibly filter it...
    train_data, eval_data = load_data(EXCLUDE_NOISE_QUESTIONS)

    # 2. Load Hugging Face tokenizer
    tokenizer = get_tokenizer()

    # 3. Tokenize both training and evaluation datasets
    train_dataset = tokenize_dataset(train_data, tokenizer)
    eval_dataset  = tokenize_dataset(eval_data, tokenizer)

    # 4. Save to disk (this will speed up the process when the dataset is big)
    save_tokenized_dataset(train_dataset, "train")
    save_tokenized_dataset(eval_dataset, "eval")

    print("Preprocessing completed!")

    # 5. Train model using tokenized data
    train_model(train_dataset, eval_dataset, tokenizer)
    
    print("The training and saving of the model is completed!")

    print("The train pipeline has succeded!")


if __name__ == "__main__":
    main()