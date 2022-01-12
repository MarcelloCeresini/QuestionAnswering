from tqdm import tqdm
from typing import List, Dict, Tuple
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from config import Config

def read_question_set(path_to_json:str) -> Dict:
    '''
    Reads the dataset's JSON file and returns it as a 
    Python dictionary
    '''
    with open(path_to_json, 'r') as f:
        questions = json.load(f)
    return questions

    
def find_start_end_token_one_hot_encoded(
    answers: Dict, 
    offsets: List[Tuple[int]]) -> int:
    '''
    This function returns the starting and ending token of the answer, 
    already one hot encoded and ready for binary crossentropy.

    Inputs:
        - answers: `List[Dict]` --> for each question, a list of answers.
            Each answer contains:
            - answer_start: the index of the starting character
            - text: the text of the answer, that we exploit through the 
                number of chars that it containts
        - offsets: `List[Tuple[int]]` --> the tokenizer from HuggingFace 
            transforms the sentence (question+context) into a sequence of tokens. 
            Offsets keeps track of the character start and end indexes for each token.
   
    Output:
        - result: `Dict` --> each key contains only one array, the one-hot 
            encoded version of, respectively, the start and end token of 
            the answer in the sentence (question+context)
    '''
    result = {
        "out_S": np.zeros(len(offsets)),
        "out_E": np.zeros(len(offsets))
    } 

    for answer in answers:
        starting_char = answer['answer_start']
        answer_len = len(answer['text'])

        # We skip the first token, [CLS], that has (0,0) as a tuple
        for i in range(1, len(offsets)):
            # We cycle through all the tokens of the question, until we find (0,0), 
            # which determines the [SEP] token, a special character which indicates 
            # the beginning of the context
            if offsets[i] == (0,0): 
                # We skip the first and the last tokens, both special tokens
                for j in range(1, len(offsets)-i-1): 
                    # If the starting char is in the interval, the index (j) 
                    # of its position inside the context, plus the length 
                    # of the question (i) is the right index
                    if (starting_char >= offsets[i+j][0]) and \
                        (starting_char <= offsets[i+j][1]):
                        result["out_S"][i+j] += 1
                    # If the ending char (starting + length -1) is in the interval, 
                    # same as above.
                    if (starting_char + answer_len - 1 >= offsets[i+j][0]) and \
                        (starting_char + answer_len - 1 < offsets[i+j][1]):
                        result["out_E"][i+j] += 1
                        break
                # After this cycle, we must check other answers
                break
    return result


def create_dataset_and_ids(
        data: Dict,
        config: Config, 
        return_labels:bool=False,
        return_NER_attention:bool=False
    ) -> Tuple[tf.data.Dataset, List[str]]:
    '''
    This function takes in input the whole data structure and iteratively 
    composes (question+context) pairs, plus their label

    Inputs:
        - data: `Dict` - The data structure containing the data
        - config: `Constants` - The configuration object containing the tokenizer
        - return_labels: `bool` - Whether labels are needed or not. For example,
            when testing the model we might not have labels in the dataset.

    Outputs:
        - dataset: `tf.data.Dataset` --> the data structure containing 
            (features, labels) that will be fed to the model during fitting
            more specifically:
            - features: `Dict` --> keys:
                - input_ids: array of token ids
                - attention_mask: array indicating if the corresponding 
                    token is padding or not
                - NER_attention (optional, flag `return_NER_attention`): 
                    array containing the NER attention weights (TODO)
             - labels: `Dict` --> keys:
                - gt_S: array representing the index of the initial token 
                    of the answer, one-hot encoded
                - gt_E: array representing the index of the final token 
                    of the answer, one-hot encoded
        - ids: List[str] --> The list of IDs of questions in the dataset.

    This function, for each artiscle in "data", extracts all paragraphs 
    (and their text, the "context"), and for each paragraph, all 
    questions_and_answers.
    At this point, it tokenizes (question+context) while truncating and 
    padding up to MAX_LEN_PAIRS.
    Moreover, it also returns the "attention_mask", an array that tells if the 
    token is padding or normal, that will be used by the model.

    In the end, it returns a dataset (`tf.data.Dataset`) with the structure 
    (features, labels), to be injected directly in the fit method of the model
    '''
    features = []
    labels = []
    ids = []

    for article in tqdm(data["data"]):
        for paragraph in article["paragraphs"]:
            for question_and_answer in paragraph["qas"]:
                ### QUESTION AND CONTEXT TOKENIZATION ###
                # For question answering with BERT we need to encode both 
                # question and context, and this is the way in which 
                # HuggingFace's BertTokenizer does it.
                # The tokenizer returns a dictionary containing all the information we need
                encoded_inputs = config.tokenizer(
                    question_and_answer["question"],    # First we pass the question
                    paragraph["context"],               # Then the context

                    max_length = config.INPUT_LEN,      # We want to pad and truncate to this length
                    truncation = True,
                    padding = 'max_length',             # Pads all sequences to 512.

                    return_token_type_ids = False,      # Return if the token is from sentence 
                                                        # 0 or sentence 1
                    return_attention_mask = True,       # Return if it's a pad token or not

                    return_offsets_mapping = True       # Returns each token's first and last char 
                                                        # positions in the original sentence
                                                        # (we will use it to match answers starting 
                                                        # and ending points to tokens)
                )

                if return_labels:
                    ### MAPPING OF THE START OF THE ANSWER BETWEEN CHARS AND TOKENS ###
                    # We want to pass from the starting position in chars to the starting position in tokens
                    label = find_start_end_token_one_hot_encoded(
                        # We pass the list of answers (usually there is still one per question,
                        #   but we mustn't assume anything)
                        answers = question_and_answer["answers"],
                        # And also the inputs offset mapping just recieved from the tokenizer
                        offsets = encoded_inputs["offset_mapping"]
                    )
                    labels.append(label)
                
                encoded_inputs.pop("offset_mapping", None) # Removes the offset mapping, not useful anymore 
                                                           # ("None" is used because otherwise KeyError 
                                                           # could be raised if the key wasn't present)

                if return_NER_attention:
                    # TODO: implement a realistic NER attention vector
                    encoded_inputs['NER_attention'] = np.ones(config.INPUT_LEN)

                features.append(encoded_inputs)
                ids.append(question_and_answer["id"])

    print("Creating dataset")
    if return_labels:
        dataset = tf.data.Dataset.from_tensor_slices((
            pd.DataFrame.from_dict(features).to_dict(orient="list"),  # dataframe for features 
            pd.DataFrame.from_dict(labels).to_dict(orient="list")     # dataframe for labels 
        ))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            pd.DataFrame.from_dict(features).to_dict(orient="list")
        )
    return (dataset, ids)


def start_end_token_from_probabilities(
    pstartv: np.array, 
    pendv: np.array, 
    dim:int=512) -> List[List[int]]:
    '''
    Returns a List of [StartToken, EndToken] elements computed from the batch outputs.
    '''
    idxs = []
    for i in range(pstartv.shape[0]):
        # For each element in the batch, transform the vectors into matrices
        # by repeating them dim times:
        # - Vectors of starting probabilities are stacked on the columns
        pstart = np.stack([pstartv[i,:]]*dim, axis=1)
        # - Vectors of ending probabilities are repeated on the rows
        pend = np.stack([pendv[i,:]]*dim, axis=0)
        # Once we have the two matrices, we sum them (element-wise operation)
        # to obtain the scores of each combination
        sums = pstart + pend
        # We only care about the scores in the upper triangular part of the matrix
        # (where the ending index is greater than the starting index)
        # therefore we zero out the diagonal and the lower triangular area
        sums = np.triu(sums, k=1)
        # The most probable set of tokens is the one with highest score in the
        # remaining matrix. Through argmax we obtain its position.
        val = np.argmax(sums)
        # Since the starting probabilities are repeated on the columns, each element
        # is identified by the row. Ending probabilities are instead repeated on rows,
        # so each element is identified by the column.
        row = val // dim
        col = val - dim*row
        idxs.append([row,col])
    return idxs


def random_baseline_predict():
    raise NotImplementedError


def compute_predictions(dataset: tf.data.Dataset, 
                        ids: List[str], 
                        config: Config,
                        model: keras.Model=None,
                        mode='predict'):
    '''
    Computes predictions given the dataset, the list of IDs, the 
    used configuration parameters and optionally a model.

    `mode` can be one of `predict`, `baseline_random`. When using
    a baseline mode, the `model` argument will not be used for the
    predictions (can be None) and can therefore be omitted.
    '''
    predictions = {}
    for sample, id in dataset.take(len(dataset)), ids:
        input_ids = sample["input_ids"]
        if mode == 'predict':
            assert model is not None, "Model is None, cannot use mode 'predict'"
            pstartv, pendv = model.predict(sample)
        elif mode == 'baseline_random':
            pstartv, pendv = random_baseline_predict()
        else:
            raise NotImplementedError
        # Obtain the limits from the probabilities
        predicted_limits = start_end_token_from_probabilities(
            pstartv, pendv
        )
        # Decode the answer's tokens
        predictions[id] = config.tokenizer.decode(
            input_ids[predicted_limits[0]:predicted_limits[1]+1], 
            skip_special_tokens=True
        )
    return predictions
