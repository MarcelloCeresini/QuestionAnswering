import os
import requests
import zipfile
from tqdm import tqdm
import time
import random
import datetime
from IPython.display import display
from functools import partial

from typing import List, Dict, Callable, Sequence, Tuple

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import ConfigFile
config = ConfigFile()

tokenizer = config.tokenizer

def create_data_for_dataset_predictions(data):
    '''
    This function takes in input the whole data structure and iteratively composes question+context pairs, plus their label
    Inputs:
        data: Dict --> the data structure containing the data
    Outputs:
        tf.data.Dataset --> the data structure containing (features, labels) that will be fed to the model during fitting
        more specifically:
        features: Dict --> keys:
            - input_ids: array of token ids
            - attention_mask: array indicating if the corresponding token is padding or not
        labels: Dict --> keys:
            - gt_S: array representing the index of the initial token of the answer, one-hot encoded
            - gt_E: array representing the index of the final token of the answer, one-hot encoded

    This function, for each article in "data", extracts all paragraphs (and their text, the "context"), for each paragraph, all questions_and_answers
    At this point, it tokenizes (question+context) while truncating and padding up to MAX_LEN_PAIRS
    Moreover, it also returns the "attention_mask", an array that tells if the token is padding or normal, that will be used by the model

    It also keeps track, through "find_start_end_token_one_hot_encoded", of the index of the initial and final token of the answer, the labels for the model

    In the end, it returns a tf.data.Dataset with the structure (features, labels), to be injected directly in the fit method of the model
    '''
    features = []
    ids = []

    for article in tqdm(data["data"]):
        for paragraph in article["paragraphs"]:
            for question_and_answer in paragraph["qas"]:
                ### QUESTION AND CONTEXT TOKENIZATION ###
                # For question answering with BERT we need to encode both 
                # question and context, and this is the way in which 
                # HuggingFace's BertTokenizer does it.
                # The tokenizer returns a dictionary containing all the information we need
                encoded_inputs = tokenizer(
                    question_and_answer["question"],    # First we pass the question
                    paragraph["context"],               # Then the context
                    max_length = config.INPUT_LEN,         # We want to pad and truncate to this length
                    truncation = True,
                    padding = 'max_length',             # Pads all sequences to 512.
                                                        # If "True" it would pad to the longest sentence in the batch 
                                                        # (in this case we only use 1 sentence, so no padding at all)
                    return_token_type_ids = False,      # Return if the token is from sentence 0 or sentence 1 
                    return_attention_mask = True,       # Return if it's a pad token or not
                    return_offsets_mapping = True       # Really important --> returns each token's first and last char position in the original sentence 
                )
                
                encoded_inputs.pop("offset_mapping", None) # Removes the offset mapping, not useful anymore 
                                                           # ("None" is used because otherwise KeyError could be raised if the key wasn't present)
                features.append(encoded_inputs)
                ids.append(question_and_answer["id"])

    return (
        tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(features).to_dict(orient="list")), 
        ids
    )


def start_end_token_from_probabilities(pstartv: np.array, 
                                    pendv: np.array, 
                                    dim:int=config.INPUT_LEN) -> List[List[int]]:
    '''
    Returns a List of [StartToken, EndToken] elements computed from the batch outputs.
    '''
    idxs = []
    for i in range(pstartv.shape[0]):
        pstart = np.stack([pstartv[i,:]]*dim, axis=1)
        pend = np.stack([pendv[i,:]]*dim, axis=0)
        sums = pstart + pend
        sums = np.triu(sums, k=1) # Zero out lower triangular matrix + diagonal
        val = np.argmax(sums)
        row = val // dim
        col = val - dim*row
        idxs.append([row,col])
    return idxs


def compute_predictions(dataset, ids, model):
    predictions = {}
    for sample, id in dataset.take(len(dataset)), ids:
        input_ids = sample["input_ids"]
        predicted_limits = start_end_token_from_probabilities(*model.predict(sample))
        predictions[id] = tokenizer.decode(input_ids[predicted_limits[0]:predicted_limits[1]+1], skip_special_tokens=True)
