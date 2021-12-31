import sys
import os
import json
from typing import Dict

def read_question_set(path_to_json:str) -> Dict:
    with open(path_to_json, 'r') as f:
        questions = json.load(f)
    return questions

if __name__ == '__main__':
    # Check that there is exactly one argument (the path to the
    # file containing the questions)
    assert len(sys.argv) == 2
    # READ JSON FILE
    read_question_set(sys.argv[1])
    # PROCESS QUESTIONS
    # LOAD MODEL
    # COMPUTE PREDICTIONS
    # FORMAT AND SAVE PREDICTION FILE
