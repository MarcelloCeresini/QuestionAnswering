# QuestionAnswering

This repository contains the code for the Final Project of the Natural Language Processing course in the Artificial Intelligence master degree at UniBo.

The objective of the project is to create an NLP system that solves the problem of Question Answering on the SQuAD dataset. The project has been extended for Open Domain Question Answering, with an additional module ([DPR](https://arxiv.org/abs/2004.04906)) for a 3 CFU Project Work for the same course.

## Quick start

Clone the repository, create a virtual environment and install the requirements provided in [`requirements.txt`](./requirements.txt).

```bash
python3 -m venv .env # or conda create -n NLP python3
```
Then, once the environment is active:
```bash
python3 -m pip install -r requirements.txt
```

Our normal model's weights can be downloaded from [here](https://drive.google.com/file/d/1wtVeJk5Szqc8nLt6wjsPh-oniIA1splv/view?usp=sharing), while the BERT model's weights can be downloaded from [here](https://drive.google.com/file/d/1gMv8aer4v9lF_ipPhLMgIHTZiziSSPBt/view?usp=sharing). They must be placed in `src/checkpoints`. The DPR module's weights can be downloaded [here](https://drive.google.com/file/d/1-6dnYYZuIOulHz0-EKGmOWvmsH7ciKGh/view?usp=sharing) and must be placed in `src/checkpoints/training_dpr`.

Another important step is to download SpaCy's english language model:

```bash
python3 -m spacy download en_core_web_sm
```

Then, the model can be evaluated on a test dataset using `python3 compute_answers.py *PATH_TO_TEST_JSON_FILE*`.

## Organization of the repository

- [`TaskExplanation.pdf`](./TaskExplanation.pdf) contains the explanation of the task
- [`data`](data/) contains the JSON files of the training (`training_set.json`), validation (`validation_set.json`), test (`dev_set.json`), as well as some intermediate files for analysis.
- [`src`](src/) contains the code of our tests and experiments.
    - Final Project (QA)
        - [config.py](src/config.py) and [utils.py](src/utils.py) contain utility code that is used thoughout all other files
        - [checkpoints](src/checkpoints/) should contain the weights of the model
        - [baselines.ipynb](src/baselines.ipynb) is a notebook containing the implementation of the baselines described in the report
        - [data_analysis.ipynb](src/data_analysis.ipynb) contains an analysis of the dataset
        - [error_analysis.ipynb](src/error_analysis.ipynb) contains an analysis of the mistakes that the model makes with respect to the ground truth
        - [train.ipynb](src/train.ipynb) is a notebook containing all of the training experiments we conducted
        - [evaluation_tests.ipynb](src/evaluation_tests.ipynb) contains the evaluations whose results we presented in the report
    - 3 CFU Project Work (Open Domain QA)
        - [tf_idf_retrieval_baseline.ipynb](src/tf_idf_retrieval_baseline.ipynb) contains the implementation and analysis of a simple baseline for the OpenQA task, as explained in the report
        - [dense_passage_retriever.ipynb](src/dense_passage_retriever.ipynb) contains the definition and training of the DPR
        - [mixing_evaluations.ipynb](src/mixing_evaluations.ipynb) contains the evaluations for the full OpenQA task uisng our different mixing methods
        - [DPR_and_other_methods_analysis_over_k_and_epochs.ipynb](src/DPR_and_other_methods_analysis_over_k_and_epochs.ipynb) is an analysis evaluating the performances of the DPR and its mixing methods during the training epochs.
        - [create_exact_over_retrieval_accuracy_graph.ipynb](src/create_exact_over_retrieval_accuracy_graph.ipynb) was used to obtain a graph included in the report.
- [report_final_project.pdf](report_final_project.pdf) contains the report for the final project (Question Answering with DistilBert).
- [report_project_work.pdf](report_project_work.pdf) contains the report for the project work (Open Domain QA with Sparse and Dense Representations).
