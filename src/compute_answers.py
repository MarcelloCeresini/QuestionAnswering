import os
import json
import sys
import tensorflow as tf
from config import Config
import utils

# For Colab:
# BEST_WEIGHTS_PATH = "/content/drive/MyDrive/Uni/Magistrale/NLP/Shared/ProjectWeights/training/data/training/"
BEST_WEIGHTS_PATH = os.path.join('checkpoints', 'normal', 'normal.h5')
PATH_TO_PREDICTIONS_JSON = os.path.join('eval', 'output_predictions.json')

if __name__ == '__main__':
    # Check that there is exactly one argument (the path to the file containing the questions)
    assert len(sys.argv) == 2, "Please, provide the position of the JSON dataset as argument"
    # Read dataset (JSON file)
    DATASET_PATH = sys.argv[1]
    # Create Config object
    config = Config()
    # Read dataset (JSON file)
    data = utils.read_question_set(DATASET_PATH)
    # Process questions for passing through the model
    dataset = utils.create_dataset_from_generator(data, config, for_training=False)
    print("Number of samples: ", len(dataset))
    dataset = dataset.batch(config.BATCH_SIZE)
    # Create the original dataset with the same order as the processed one
    original_dataset = utils.create_original_dataset(data, config)
    original_dataset = original_dataset.batch(config.BATCH_SIZE)
    # Load model
    model:tf.keras.Model = config.create_standard_model(hidden_state_list=[3,4,5,6])
    # Load best model weights
    model.load_weights(BEST_WEIGHTS_PATH)
    # Predict the answers to the questions in the dataset
    predictions = utils.compute_predictions(dataset, original_dataset, model)
    # Create a prediction file formatted like the one that is expected
    with open(PATH_TO_PREDICTIONS_JSON, 'w') as f:
        json.dump(predictions, f)
