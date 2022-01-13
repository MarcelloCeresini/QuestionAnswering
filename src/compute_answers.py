import os
import sys
import json
import utils
from config import Config

if __name__ == '__main__':
    config = Config()
    # Check that there is exactly one argument (the path to the
    #   file containing the questions)
    assert len(sys.argv) == 2, "Please, provide the position of the JSON dataset as argument"
    # Read dataset (JSON file)
    data = utils.read_question_set(sys.argv[1])
    # Process questions
    dataset = utils.create_dataset_and_ids(data, config, for_training=False)
    dataset = dataset.batch(config.BATCH_SIZE)
    print("Number of samples: ", len(dataset)*config.BATCH_SIZE)
    # Load model
    model = config.create_standard_model(hidden_state_list=[3,4,5,6])
    # Load best model weights
    #   TODO: We have no weights yet
    #   BEST_WEIGHTS_PATH = "../data/training/training_normal/cp-0005.ckpt"
    #   model.load_weights(BEST_WEIGHTS_PATH)
    # Predict the answers to the questions in the dataset
    predictions = utils.compute_predictions(dataset, config, model,
        mode='predict') # mode='baseline_random' for random baseline
    # Create a prediction file formatted like the one that is expected
    PATH_TO_PREDICTIONS_JSON = os.path.join('src', 'eval', 'predictions.json')
    with open(PATH_TO_PREDICTIONS_JSON, 'w') as f:
        json.dump(predictions, f)
