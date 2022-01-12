import sys
import json
import utils
from config import Config

if __name__ == '__main__':
    config = Config()
    # Check that there is exactly one argument (the path to the
    #   file containing the questions)
    assert len(sys.argv) == 2
    # Read dataset (JSON file)
    data = utils.read_question_set(sys.argv[1])
    # Process questions
    dataset, ids = utils.create_dataset_and_ids(data, return_labels=False)
    # Load model
    model = config.create_standard_model()
    # Load best model weights
    #   TODO: We have no weights yet
    #   BEST_WEIGHTS_PATH = "some_path"
    #   model.load_weights(BEST_WEIGHTS_PATH)
    # Predict the answers to the questions in the dataset
    predictions = utils.compute_predictions(dataset, ids, config, model)
    # Create a prediction file formatted like the one that is expected
    PATH_TO_PREDICTIONS_JSON = "predictions.json"
    with open(PATH_TO_PREDICTIONS_JSON, 'w') as f:
        json.dump(predictions, f)





