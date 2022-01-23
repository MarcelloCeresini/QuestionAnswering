import os
import json
import tensorflow as tf
from config import Config
import utils

# For Colab:
# DATASET_PATH = "/content/QuestionAnswering/data/training_set.json"
# BEST_WEIGHTS_PATH = "/content/drive/MyDrive/Uni/Magistrale/NLP/Shared/ProjectWeights/training/data/training/"
DATASET_PATH = os.path.join('..', 'data', 'dev_set.json')
BEST_WEIGHTS_PATH = None # TODO: Add the correct path to the weights
PATH_TO_PREDICTIONS_JSON = os.path.join('eval', 'output_predictions.json')

if __name__ == '__main__':
    config = Config()
    # Read dataset (JSON file)
    data = utils.read_question_set(DATASET_PATH)
    # Process questions
    dataset = utils.create_dataset_and_ids(data, config, for_training=False)
    print("Number of samples: ", len(dataset))
    dataset = dataset.batch(config.BATCH_SIZE)
    # Load model
    model = config.create_standard_model(hidden_state_list=[3,4,5,6])
    # Load best model weights
    #   TODO: We have no weights yet
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(tf.train.latest_checkpoint(BEST_WEIGHTS_PATH))
    # Predict the answers to the questions in the dataset
    predictions = utils.compute_predictions(dataset, config, model)
    # Create a prediction file formatted like the one that is expected
    with open(PATH_TO_PREDICTIONS_JSON, 'w') as f:
        json.dump(predictions, f)
