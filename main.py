"""
main.py

This script orchestrates the full disfluency rephrasing pipeline in a single command.
It runs the training, evaluation/inference, and overfitting diagnostics steps sequentially.

Usage:
    From the root directory of the project, run:
    $ python main.py

Dependencies:
    - src.Train_pipeline.main
    - src.Inference_evaluation_pipeline.main
    - src.Check_overfitting_pipeline.main
"""

from src.Train_pipeline import main as train_main
from src.Inference_evaluation_pipeline import main as inference_main
from src.Check_overfitting_pipeline import main as overfit_main
import time

def main():
    """
    Main function that executes the ML system pipeline in three stages:
    1. Train the rephrasing model.
    2. Evaluate the trained model and run inference.
    3. Check for overfitting between training and validation results.
    """
    start_time = time.time()
    train_main()
    inference_main()
    overfit_main()
    end_time = time.time()
    elapsed = (end_time - start_time)/60
    print(f"The ML system executed successfully in {elapsed:.2f} minutes.")



if __name__ == "__main__":
    main()
