from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from src.config import model_used

def get_tokenizer()-> PreTrainedTokenizerBase:
    """
    Load and return a tokenizer from the Hugging Face Transformers library.

    This function initializes the tokenizer corresponding to the specified 
    model checkpoint (default: "t5-small") and returns it. The tokenizer is 
    used to preprocess input text for transformer models.

    Returns:
        PreTrainedTokenizerBase: The tokenizer for the specified model.
    """
     
    model_checkpoint = model_used  # or something else , we will start with a small t5
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    return tokenizer