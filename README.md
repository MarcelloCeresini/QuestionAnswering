# QuestionAnswering

This repository contains the code for the Final Project of the Natural Language Processing course in the Artificial Intelligence master degree at UniBo.

The objective of the project is to create an NLP system that solves the problem of Question Answering on the SQuAD dataset. 

## Quick start

Clone the repository, create a virtual environment and install the requirements provided in [`requirements.txt`](./requirements.txt).

```bash
python3 -m venv .env # or conda create -n NLP python3
```
Then, once the environment is active:
```bash
python3 -m pip install -r requirements.txt
```

Our normal model's weights can be downloaded from [here](https://drive.google.com/file/d/1wtVeJk5Szqc8nLt6wjsPh-oniIA1splv/view?usp=sharing). They must be placed in `src/checkpoints`.

Another important step is to download SpaCy's english language model:

```bash
python3 -m spacy download en_core_web_sm
```

Then, the model can be evaluated on a test dataset using `python3 compute_answers.py *PATH_TO_TEST_JSON_FILE*`.


## Organization of the repository

- [`TaskExplanation.pdf`](./TaskExplanation.pdf) contains the explanation of the task
- [`data`](data/) contains the JSON files of the training (`training_set.json`), validation (`validation_set.json`), test (`dev_set.json`)
- [`src`](src/) contains the code of our tests and experiments.
    - [config.py](src/config.py) and [utils.py](src/utils.py) contain utility code that is used thoughout all other files
    - [checkpoints](src/checkpoints/) should contain the weights of the model
    - [baselines.ipynb](src/baselines.ipynb) is a notebook containing the implementation of the baselines described in the report
    - [data_analysis.ipynb](src/data_analysis.ipynb) contains an analysis of the dataset
    - [error_analysis.ipynb](src/error_analysis.ipynb) contains an analysis of the mistakes that the model makes with respect to the ground truth
    - [train.ipynb](src/train.ipynb) is a notebook containing all of the training experiments we conducted
    - [evaluation_tests.ipynb](src/evaluation_tests.ipynb) contains the evaluations whose results we presented in the report
- [report_final_project.pdf](report_final_project.pdf) contains the report for the final project (Question Answering with DistilBert).
- [report_project_work.pdf](report_project_work.pdf) contains the report for the project work (Open Domain QA with Sparse and Dense Representations).
