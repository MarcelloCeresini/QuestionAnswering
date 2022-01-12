import sys
import os
import json
from typing import Dict

import utils

def read_question_set(path_to_json:str) -> Dict:
    with open(path_to_json, 'r') as f:
        questions = json.load(f)
    return questions

if __name__ == '__main__':
    # Check that there is exactly one argument (the path to the
    # file containing the questions)
    assert len(sys.argv) == 2
    # READ JSON FILE
    data = read_question_set(sys.argv[1])
    # PROCESS QUESTIONS
    # LOAD MODEL
    # COMPUTE PREDICTIONS
    # FORMAT AND SAVE PREDICTION FILE
    dataset = utils.create_data_for_dataset_predictions(data)
    model = utils.config.create_standard_model([3, 4, 5, 6])
    BEST_WEIGHTS_PATH = "../data/training/training_normal/cp-0005.ckpt" ######################################### TODO: actually add the path
    model.load_weights(BEST_WEIGHTS_PATH)
    predictions = utils.compute_predictions(dataset, model)

    PATH_TO_PREDICTIONS_JSON = "predictions.json"
    with open(PATH_TO_PREDICTIONS_JSON, 'w') as f:
        json.dump(predictions, f)
