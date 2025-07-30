import os
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq , EarlyStoppingCallback
#from datetime import datetime
import random
import numpy as np
import torch
from src.config import model_used

#Need this function to reproduce ouputs
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_model(model_checkpoint=model_used):
    return T5ForConditionalGeneration.from_pretrained(model_checkpoint)

def train_model(train_dataset, eval_dataset, tokenizer):


    model = get_model()
    #Get current timestamp as a string, formatted like: "2025-07-26_07_30AM", which is done for logging purposes.
    #timestamp = datetime.now().strftime("%Y-%m-%d_%I_%M%p")
    #Define where we will save the trained model
    OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    MODEL_DIR = os.path.join(OUTPUTS_DIR, "model")
    #OUTPUT_MODEL_DIR=os.path.join(MODEL_DIR, f"{model_used}_{timestamp}")
    OUTPUT_MODEL_DIR = os.path.join(MODEL_DIR, model_used)

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(OUTPUT_MODEL_DIR, "logs"),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True, ##--># Enables use of model.generate() during eval, which means that we're telling the model to use its generation model instead of just returning raw logits from decoder when... 
        #computing predictions. Meaning no forward method is used. Without predict_with_generate=True, your metrics (like BLEU, ROUGE, etc.) will be meaningless.
        logging_steps=10,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42   #--> this ensures the final model is the best one (not the last one).
    )

    # Collator for padding and label masking
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]  # stops if metric doesn't improve for 10 evals
    )

    trainer.train()
    #Saving model
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"Model and tokenizer saved in: {OUTPUT_MODEL_DIR}")

