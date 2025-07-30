# Disfluence rephrasing model (Disfluent → Fluent) 

This ML system fine-tunes T5 based models to rephrase disfluent natural language questions into fluent ones, using the [Disfl-QA dataset](https://github.com/google-research-datasets/disfl-qa) by Google Research. It supports training, evaluation through metrics like BLEU, ROUGE, and  overfitting checking.

## ML system project structure

The project structure is the following, along with its high level file description is given below : 
```
Chata.AI_ML_assignment/
├── data/
│ ├── train.json ->The data file that is used to train the rephrasing model.
| ├── dev.json   ->The data file that is used to evaluate the rephrasing model.
│ └── train_tokenize/ ->Folder used to save the train tokenized data.
│ └── dev_tokenize/ ->Folder used to save the evaluation tokenized data.
|
├── notebooks/  -> Jupyter notebooks used initialy to understand the data's behaviour. The project will not touch this
|
|── outputs/  
│ ├── model/ -> Folder where the trained model and its associated files are stored.
| ├── overfitting_diagnostics/ ->Folder where the output of the Check_overfitting_pipeline.py is stored. 
│ └── predictions/ ->Folder where the output of the Inference_evaluation_pipeline.py is stored.
|
├── src/ ->Folder where all the codebase is located.
│ ├── init.py
| ├── Check_overfitting_pipeline.py ->Runs the overfitting.py file to check for overfitting( difference in scores between training and eval. data).
| ├── config.py -> File that contains the environment variables or configuration of the ML system.
│ ├── data_loader.py -> File that loads the data into a structure recognizable by Huggingface library. Also, if user desires(config.py), it filters noisy data.
│ ├── evaluation.py -> File used by the Inference_evaluation_pipeline.py to calculate the ROUGE nad BLEU metrics of the trained model.
│ ├── Inference_evaluation_pipeline.py -> File that uses the evaluation.py file to calculate the evaluation metrics of the fine tuned model.
│ ├── overfitting.py -> File used by Check_overfitting_pipeline.py to create a file that calculates overfitting.
│ ├── tokenize_data.py -> File used to tokenize the train and eval dataset.
│ ├── tokenizer_setup.py ->Set up the tokenizer from a checkpoint.
│ ├── train.py ->File used by the Train_pipeline.py file to fine tune a rephrasing model.
│ ├── Train_pipeline.py ->File used to extract,save, and train the rephrasing model.
|
├── tests/ ->Folder where test where developed 
| ├── test_data_loader.py ->Python file used to test the data_loader function for anomalous entries.
| ├── test_tokenize_data.py ->Python file used to test the tokenize_data function for anomalous entries.
|
├── environment.yml ->File created so that the user can reproduce the exact environment that the repository was developed on.
└── main.py -> File use to run the ML system in sequence(The orchestrator).
└── README.md ->File that tells the user how to use the ML system in detail.
```

## Getting Started

### 1. Clone the repository
The user must git clone the repository in their local machine. To do this , the user needs the following command :
```bash
git clone https://github.com/your-username/your-repo-name.git
```
Please replace the "your-username" and "your-repo-name" values with the actual repository values. 
### 2.Reproduce the environment

The ML system uses `conda` for dependency management. To reproduce the environment from the `environment.yml` file in the repository, run the following command :

```bash
conda env create -f environment.yml
```
⚠️ **Note**: The ML system was developed and tested using **Python 3.10**.  
Using a different Python version may lead to dependency or runtime issues.

Once the environment has been created, the user needs to activate it to run the ML system. The command is the following :
```bash
conda activate hf_env
```

### 3.Define the run configuration 

The ML system uses the config.py file found inside the src folder to run the system with specific configurations variables. The allowed configurations are the following :

1. **EXCLUDE_NOISE_QUESTIONS** is a boolean variable that has either a true or false value. If true, the ML system filters out questions in which the disfluent text value is '#Value' or 'question n/a'. If false, the ML system uses all questions.
2. **model_used** is a string variable that tells which model to train on . The system has been tested with the following models :
"t5-small", and "t5-base".

## Run the ML system 

### 1.ML system workflow
To project was designed with the following workflow in mind :
```
1. Train_pipeline.py --> Saves the trained model in the /outputs/model folder.
2. Inference_evaluation_pipeline.py --> Saves the evaluation metrics and predictions of the trained model in /outputs/predictions folder.
3. Check_overfitting_pipeline.py --> Saves the overfitting diagnostics of the trained model in  /outputs/overfitting_diagnostics folder.
```
As such, the "main.py" in the project root (Chata.AI_ML_assignment) was design to orchestrate the pipelines in the order above.

### 2.Running the ML system
To run the ML system using the the pipeline orchestrator(main.py), the user must use the following command :
```bash
python main.py
```
⚠️ **Note**: You must run this command from the root of the project directory (Chata.AI_ML_assignment) for the system to execute correctly.

## Run the tests
This project uses Python's built-in `unittest` framework.  
To run all unit tests located in the `test/` folder, execute the following command from the project root(Chata.AI_ML_assignment):

```bash
python -m unittest discover -s tests
```
## Additional information
The notebooks directory contains two Jupyter notebooks used for exploratory data analysis (EDA) on the dataset. These notebooks developed an understanding of the dataset's structure and supported the development of utility functions, some of which are integrated into the machine learning system.

## Author

**Edwin Aguirre**  
- [LinkedIn](https://www.linkedin.com/in/edwin-aguirre-140687b1/)  
