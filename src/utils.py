### This file contains utility functions that are used in   ###
###      all notebooks and scripts of the project.          ###

from typing import List, Dict, Tuple
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from functools import partial

from config import Config


def read_question_set(path_to_json:str) -> Dict:
    '''
    Reads the dataset's JSON file and returns it as a Python dictionary
    '''
    with open(path_to_json, 'r') as f:
        questions = json.load(f)
    return questions

def find_start_end_token_one_hot_encoded(
    answers: Dict, 
    offsets: List[Tuple[int]]) -> Dict:
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


def create_NER_attention_vector(context: str, 
    offsets: List[Tuple[int]],
    spacy_instance,
    config:Config, non_ne_weight:float=0.8,
    ne_weight:float=1.2):
    '''
    Creates a NER_attention vector. 
    It uses SpaCy for finding named entities in the context. Then it matches the
    named entities to the tokens produced by BERT's tokenizer.
    The NER_attention vector is a vector as long as the tokens containing 
    `non_ner_weight` (default 0.8) for each token that is not a named entity and
    `ner_weight` (default 1.2) for each token that is a named entity.

    Inputs:
    - context: `str` - The context where we look for named entities
    - offsets: `List[Tuple[int]]` - A list keeping track of the character start 
        and end indexes for each token.
    - spacy_instance: `spacy.lang.en.English` - An element that converts a text
        into a document which contains high level information (eg. about named
        entities)
    - config: `Config` - The configuration object containing all constants
    - non_ne_weight: `float` - The weight for non-named entity tokens (default 0.8)  
    - ne_weight: `float` - The weight for named entity tokens (default 1.2)

    Outputs:
    - NER_attention: `np.array` - The array of weights for the tokens, where named
        entity tokens are weighted more than the rest
    '''
    # Processes the context 
    doc = spacy_instance(context)

    # Instantiates the NER attention vector (default weight: 0.8)
    NER_attention  = np.ones(config.INPUT_LEN) * non_ne_weight
    # Create lists of starting and ending character indices of named entities
    starting_chars = [ ent.start_char for ent in doc.ents ]
    ending_chars   = [ ent.end_char for ent in doc.ents ]
    
    # Iterate over the offsets to obtain the NER tokens
    NER_index = 0
    # We skip the first token, [CLS], that has (0,0) as a tuple
    for i in range(1, len(offsets)):
        # We cycle through all the tokens of the question, until we find (0,0), 
        # which determines the [SEP] token, a special character which indicates 
        # the beginning of the context
        # OSS: character counts reset at the beginning of the context, restarting from 0
        if offsets[i] == (0,0): 
            # We skip the first and the last tokens, both special tokens
            j = i+1
            while j < len(offsets):
                # If there are still named entities to find
                if NER_index < len(starting_chars):
                    # When we find a match with the starting index, go on to find the end index
                    if starting_chars[NER_index] >= offsets[j][0] and starting_chars[NER_index] < offsets[j][1]:
                        # Put a ne_weight at all indices containing a named entity
                        NER_attention[j] = ne_weight
                        while ending_chars[NER_index] > offsets[j][1] and j < len(offsets)-1:
                            j += 1
                            NER_attention[j] = ne_weight
                        # Update the counter for tagged named entities
                        NER_index += 1
                j += 1
    
    return NER_attention

def create_full_dataset(data: Dict, config: Config,
    return_labels:bool=False, return_NER_attention:bool=False,
    return_question_id:bool=False, NER_value:float=0):
    '''
    This function takes in input the whole data structure and constructs
    a full high-level dataset, which produces (question+context) pairs
    plus optionally their label and question IDs.

    Inputs:
        - data: `Dict` - The data structure containing the data
        - config: `Constants` - The configuration object containing the tokenizer
        - return_labels: `bool` - Whether labels are needed or not. For example,
            when testing the model we might not have labels in the dataset.
        - return_NER_attention: `bool` - Whether to also create and return a NER
            attention vector
        - return_question_IDs: `bool` - Whether to return or not the question ID
        - NER_value: `float` - hyperparameter value for NER attention enhancement

    Note: all optional return labels are set to False by default

    Outputs:
        - dataset: `tf.data.Dataset` --> the data structure containing either
            (features), (features, ids), (features, labels) or (features, ids, labels)
            that will be fed to the model during training or evalution.
            More specifically:
            - features: `Dict` --> keys:
                - input_ids: array of token ids
                - attention_mask: array indicating if the corresponding 
                    token is padding or not
                - NER_attention (optional, flag `return_NER_attention`): 
                    array containing the NER attention weights
                - token_type_ids (optional, if `config` was instanced using 
                    the `bert=True` flag)
            - ids: List[str] --> The list of IDs of questions in the dataset.
            - labels: (optional) `Dict` --> keys:
                - gt_S: array representing the index of the initial token 
                    of the answer, one-hot encoded
                - gt_E: array representing the index of the final token 
                    of the answer, one-hot encoded

    This function, for each article in "data", extracts all paragraphs 
    (and their text, the "context"), and for each paragraph, all 
    questions_and_answers.
    At this point, it tokenizes (question+context) while truncating and 
    padding up to MAX_LEN_PAIRS.
    Moreover, it also returns the "attention_mask", an array that tells if the 
    token is padding or normal, that will be used by the model.

    In the end, it returns a dataset (`tf.data.Dataset`) with the structure 
    (features, labels), to be injected directly in the fit method of the model

    Note: this method can be RAM intensive, because it has to store the entire
    dataset while producing it. Using `create_dataset_from_generator`, which uses a 
    Python generator that produces data on-the-fly, is an alternative for low-memory
    environments.
    '''
    features = []
    labels = []
    ids = []

    for article in tqdm(data["data"]):
        for paragraph in article["paragraphs"]:
            for question_and_answer in paragraph["qas"]:
                ### QUESTION AND CONTEXT TOKENIZATION ###
                # For question answering with DistilBERT we need to encode both 
                # question and context, and this is the way in which 
                # HuggingFace's DistilBertTokenizer does it.
                # The tokenizer returns a dictionary containing all the information we need
                encoded_inputs = config.tokenizer(
                    question_and_answer["question"],    # First we pass the question
                    paragraph["context"],               # Then the context

                    max_length = config.INPUT_LEN,      # We want to pad and truncate to this length
                    truncation = True,
                    padding = 'max_length',             # Pads all sequences to 512.

                    return_token_type_ids = config.bert,# Return if the token is from sentence 
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


                if return_NER_attention:
                    # We call the function that produces NER weights for tokens
                    encoded_inputs['NER_attention'] = create_NER_attention_vector(
                        context=paragraph["context"], 
                        offsets=encoded_inputs["offset_mapping"],
                        spacy_instance=config.ner_extractor,
                        config=config, 
                        non_ne_weight=1-NER_value,
                        ne_weight=1+NER_value
                    )

                encoded_inputs.pop("offset_mapping", None) # Removes the offset mapping, not useful anymore 
                                                           # ("None" is used because otherwise KeyError 
                                                           # could be raised if the key wasn't present)

                features.append(encoded_inputs)
                ids.append(question_and_answer['id'])

    # Depending on the flags, we return the kind of Dataset that is requested by the user.
    if return_question_id and return_labels:
        return tf.data.Dataset.from_tensor_slices((
            pd.DataFrame.from_dict(features).to_dict(orient="list"),
            pd.DataFrame.from_dict(labels).to_dict(orient="list"),
            ids
        ))
    elif return_labels:
        return tf.data.Dataset.from_tensor_slices((
            pd.DataFrame.from_dict(features).to_dict(orient="list"),
            pd.DataFrame.from_dict(labels).to_dict(orient="list")
        ))
    elif return_question_id:
        return tf.data.Dataset.from_tensor_slices((
            pd.DataFrame.from_dict(features).to_dict(orient="list"),
            ids
        ))
    else:
        return tf.data.Dataset.from_tensor_slices(
            pd.DataFrame.from_dict(features).to_dict(orient="list"))


def dataset_generator(data: Dict, config: Config,
    return_labels:bool=False, return_NER_attention:bool=False,
    return_question_id:bool=False, NER_value:float=0):
    '''
    This function takes in input the whole data structure and iteratively 
    yields (question+context) pairs, plus optionally their label and question
    IDs.

    Inputs:
        - data: `Dict` - The data structure containing the data
        - config: `Constants` - The configuration object containing the tokenizer
        - return_labels: `bool` - Whether labels are needed or not. For example,
            when testing the model we might not have labels in the dataset.
        - return_NER_attention: `bool` - Whether to also create and return a NER
            attention vector
        - return_question_IDs: `bool` - Whether to return or not the question ID
        - NER_value: `float` - hyperparameter value for NER attention enhancement

    Note: all optional return labels are set to False by default

    Outputs:
        - dataset: `tf.data.Dataset` --> the data structure containing either
            (features), (features, ids), (features, labels) or (features, ids, labels)
            that will be fed to the model during training or evalution.
            More specifically:
            - features: `Dict` --> keys:
                - input_ids: array of token ids
                - attention_mask: array indicating if the corresponding 
                    token is padding or not
                - NER_attention (optional, flag `return_NER_attention`): 
                    array containing the NER attention weights
                - token_type_ids (optional, if `config` was instanced using 
                    the `bert=True` flag)
            - ids: List[str] --> The list of IDs of questions in the dataset.
            - labels: (optional) `Dict` --> keys:
                - gt_S: array representing the index of the initial token 
                    of the answer, one-hot encoded
                - gt_E: array representing the index of the final token 
                    of the answer, one-hot encoded

    This function, for each article in "data", extracts all paragraphs 
    (and their text, the "context"), and for each paragraph, all 
    questions_and_answers.
    At this point, it tokenizes (question+context) while truncating and 
    padding up to MAX_LEN_PAIRS.
    Moreover, it also returns the "attention_mask", an array that tells if the 
    token is padding or normal, that will be used by the model.

    In the end, it returns a dataset (`tf.data.Dataset`) with the structure 
    (features, labels), to be injected directly in the fit method of the model
    '''
    for article in data["data"]:
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

                    return_token_type_ids = config.bert,# Return if the token is from sentence 
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


                if return_NER_attention:
                    encoded_inputs['NER_attention'] = create_NER_attention_vector(
                        context=paragraph["context"], 
                        offsets=encoded_inputs["offset_mapping"],
                        spacy_instance=config.ner_extractor,
                        config=config, 
                        non_ne_weight=1-NER_value,
                        ne_weight=1+NER_value
                    )

                encoded_inputs.pop("offset_mapping", None) # Removes the offset mapping, not useful anymore 
                                                           # ("None" is used because otherwise KeyError 
                                                           # could be raised if the key wasn't present)

                if return_question_id and return_labels:  
                    yield dict(encoded_inputs), question_and_answer['id'], {
                        'out_S': label['out_S'],
                        'out_E': label['out_E']
                    }
                elif return_labels:
                    yield dict(encoded_inputs), {
                        'out_S': label['out_S'],
                        'out_E': label['out_E']
                    }
                elif return_question_id:
                    yield dict(encoded_inputs), question_and_answer['id']
                else:
                    yield dict(encoded_inputs)

def create_dataset_from_generator(
        data: Dict,
        config: Config,
        for_training: bool=True,
        use_NER_attention:bool=False,
        NER_value:float=0
    ) -> tf.data.Dataset:
    '''
    This function is used to create a lightweight TensorFlow Dataset from
    a data generator which iterates over all questions in the passed dataset.

    Inputs:
        - data: `Dict` - The data structure containing the data
        - config: `Constants` - The configuration object containing the tokenizer
        - for_training: `bool` (default: `True`) - Whether the produced dataset 
            will be used for training or not. 
            This changes the output, since a training dataset needs labels and 
            doesn't need question IDs, while an evaluation dataset needs
            question IDs and doesn't need labels.
        - use_NER_attention: `bool` - Whether to return the additional NER_attention
            tensor feature    
        - NER_value: `float` - hyperparameter value for NER attention enhancement


    Outputs:
        - dataset: `tf.data.Dataset` --> the data structure containing either
            (features), (features, ids), (features, labels) or (features, ids, labels)
            that will be fed to the model during training or evalution.
            More specifically:
            - features: `Dict` --> keys:
                - input_ids: array of token ids
                - attention_mask: array indicating if the corresponding 
                    token is padding or not
                - NER_attention (optional, flag `return_NER_attention`): 
                    array containing the NER attention weights
                - token_type_ids (optional, if `config` was instanced using 
                    the `bert=True` flag)
            - ids: List[str] (optional) --> The list of IDs of questions in the dataset.
            - labels: (optional) `Dict` --> keys:
                - gt_S: array representing the index of the initial token 
                    of the answer, one-hot encoded
                - gt_E: array representing the index of the final token 
                    of the answer, one-hot encoded

    Note: to work with the model, the dataset must be batched.
    '''
    # Labels are only returned in training, while question IDs only when not training
    return_labels = for_training
    return_question_id = not for_training

    # Create expected signature for the generator output
    if config.bert:
        features = {
            'input_ids': tf.TensorSpec(shape=(512,), dtype=tf.int32), 
            'attention_mask': tf.TensorSpec(shape=(512,), dtype=tf.int32),
            'token_type_ids': tf.TensorSpec(shape=(512,), dtype=tf.int32)
        }
    else:
        features = {
            'input_ids': tf.TensorSpec(shape=(512,), dtype=tf.int32), 
            'attention_mask': tf.TensorSpec(shape=(512,), dtype=tf.int32)
        }
    if use_NER_attention:
        features['NER_attention'] = tf.TensorSpec(shape=(512,), dtype=tf.float64)
    if for_training:
        # The dataset contains the features and the labels
        signature = (features, {
            'out_S': tf.TensorSpec(shape=(512,), dtype=tf.float64), 
            'out_E': tf.TensorSpec(shape=(512,), dtype=tf.float64)
        })
    else:
        # The dataset contains the features and the question IDs (strings)
        signature = (features, tf.TensorSpec(shape=(), dtype=tf.string))

    # Instantiates a partial generator
    data_gen = partial(dataset_generator, data, config, 
        return_labels=return_labels, 
        return_question_id=return_question_id,
        return_NER_attention=use_NER_attention,
        NER_value=NER_value)
    # Creates the dataset with the computed signature
    dataset = tf.data.Dataset.from_generator(data_gen,
        output_signature=signature)
    # Compute dataset length, to be used by tensorflow internals
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len([
        question_and_answer
        for article in data["data"]
        for paragraph in article["paragraphs"]
        for question_and_answer in paragraph["qas"]
    ])))
    # Return the dataset
    return dataset

def create_original_dataset(data: Dict, config: Config):

    features = []

    for article in tqdm(data["data"]):
        for paragraph in article["paragraphs"]:
            for question_and_answer in paragraph["qas"]:

                inputs={}
                ### QUESTION AND CONTEXT TOKENIZATION ###
                # For question answering with DistilBERT we need to encode both 
                # question and context, and this is the way in which 
                # HuggingFace's DistilBertTokenizer does it.
                # The tokenizer returns a dictionary containing all the information we need
                encoded_inputs = config.tokenizer(
                    question_and_answer["question"],    # First we pass the question
                    paragraph["context"],               # Then the context

                    max_length = config.INPUT_LEN,      # We want to pad and truncate to this length
                    truncation = True,
                    padding = 'max_length',             # Pads all sequences to 512.

                    return_token_type_ids = False,      # Return if the token is from sentence 
                                                        # 0 or sentence 1
                    return_attention_mask = False,      # Return if it's a pad token or not

                    return_offsets_mapping = True       # Returns each token's first and last char 
                                                        # positions in the original sentence
                                                        # (we will use it to match answers starting 
                                                        # and ending points to tokens)
                )
                inputs["context"] = paragraph["context"]
                inputs["offset_mapping"] = encoded_inputs["offset_mapping"]

                features.append(inputs)

    return tf.data.Dataset.from_tensor_slices(
        pd.DataFrame.from_dict(features).to_dict(orient="list"))


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

def compute_predictions(dataset: tf.data.Dataset,
                        original_dataset: tf.data.Dataset,
                        model: keras.Model):
    '''
    Computes predictions given the dataset, the used configuration parameters and model

    Inputs:
    - dataset: a `tf.data.Dataset` on which we will compute predictions.
    - original_dataset: a `tf.data.Dataset` which contains the original context
        and initial/starting characters for each token.
    - model: a `keras.Model` that computes the predictions.
    '''
    predictions = {}
    # For each sample we can extract from the dataset (it can be a single element or 
    # a batch)
    for sample, original_sample in tqdm(zip(dataset, original_dataset), total=len(dataset)):
        # We let the model predict the probability tensors given the input features
        contexts = original_sample["context"].numpy()
        offsets = original_sample["offset_mapping"].numpy()

        features = sample[0]
        pstartv, pendv = model.predict(features, verbose=0)
        # We obtain the span from the probabilities
        predicted_limits = start_end_token_from_probabilities(
            pstartv, pendv
        )
        # Then we decode the answer's tokens 
        question_ids = [x.decode('utf-8') for x in sample[1].numpy()]

        # Finaally, we produce the output dictionary for the batch
        for i in range(len(features["input_ids"])):
            question_id = question_ids[i]
            predicted_limit = predicted_limits[i]
            context = contexts[i]
            offset = offsets[i]
            # take the index of the predicted start token token, 
            # take the corresponding offsets, and take the first number (the start)
            # same thing for the end token, but with the end of the offset
            predictions[question_id] = context.decode()[
                offset[predicted_limit[0], 0] 
                : 
                offset[predicted_limit[1], 1]
            ]

    return predictions

###################################################################################
###################################################################################

##### RETRIEVAL CLASSES

class ReducedDistilBertModel(keras.Model):
    def __init__(self, distilbert_model, dimensionality=768):
        super(ReducedDistilBertModel, self).__init__()
        self.distilbert_model = distilbert_model
        self.dimensionality = dimensionality
        self.reduction_layer = keras.layers.Dense(self.dimensionality, 
                                            activation='gelu')

    def call(self, inputs):
        hidden_state = self.distilbert_model(inputs).last_hidden_state
        # We introduce a dense layer that simply reduces the dimensionality of the model's output.
        # It's not used if the output dimensionality is the same of the Bert model
        return self.reduction_layer(hidden_state) if self.dimensionality != 768 else hidden_state


class DenseEncoder(layers.Layer):
    def __init__(self, model_q, model_p):
        super().__init__()
        self.model_q = model_q  # Dense encoder for questions
        self.model_p = model_p  # Dense encoder for paragraphs
    
    def call(self, inputs, training=False):
        # Encode the questions in the batch
        # Take the first token as representation of each question
        q_repr = self.model_q({
            'input_ids': inputs['questions']['input_ids'],
            'attention_mask': inputs['questions']['attention_mask']
        })[:,0,:]
        # If we are training, we also return the representation of the paragraphs
        # and of the hard paragraph
        if training:
            # Encode the batch of paragraphs
            p_repr = self.model_p({
                'input_ids': inputs['paragraphs']['input_ids'],
                'attention_mask': inputs['paragraphs']['attention_mask']
            })[:,0,:]
            # We also encode the batch of hard paragraphs separately. 
            p_hard_repr = self.model_p({
                'input_ids': inputs['hard_paragraphs']['input_ids'],
                'attention_mask': inputs['hard_paragraphs']['attention_mask']
            })[:,0,:]
            return q_repr, p_repr, p_hard_repr
        else:
            return q_repr


class DeepQPEncoder(keras.Model):
    def __init__(self, model_q, model_p):
        super().__init__()
        self.enc = DenseEncoder(model_q, model_p)

    def call(self, inputs, training=False):
        if training:
            # For training we return the similarity matrix
            repr_q, repr_p, repr_hard_p = self.enc(inputs, training=training)
            S = tf.tensordot(repr_q, tf.transpose(repr_p), axes=1)
            # We append the hard scores
            hard_scores = tf.gather(
                # Get the elements on the diagonal of the 8x8 matrix of 
                # scores between questions and hard paragraphs
                tf.tensordot(repr_q, tf.transpose(repr_hard_p), axes=1), 
                    tf.expand_dims(
                        tf.range(tf.shape(inputs['questions']['input_ids'])[0]), 
                        axis=1), 
                    batch_dims=1
            )
            S = tf.concat([S, hard_scores], axis=1)
            return S
        else:
            # In other cases, we return the representation of the question(s)
            repr_q = self.enc(inputs, training=training)            
            return repr_q

    def train_step(self, data):
        x = data
        # y = [0, ..., batch_size-1]
        y = tf.range(tf.shape(x['questions']['input_ids'])[0])
        with tf.GradientTape() as tape:
            # Obtain similarities
            S = self(x, training=True)
            # Obtain loss value
            loss = self.compiled_loss(y, S)
        # Construct gradients and apply them through the optimizer
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(y, S)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data
        # y = [0, ..., batch_size-1]
        y = tf.range(tf.shape(x['questions']['input_ids'])[0])
        S = self(x, training=True) # We are not really training, but we have to obtain S
        self.compiled_loss(y, S)
        self.compiled_metrics.update_state(y, S)
        return {m.name: m.result() for m in self.metrics}

##### RETRIEVAL FUNCTIONS

def decode_dataset_fn(record_bytes, MAX_SEQ_LEN=512):
    # Reads one element from the dataset (as bytes) and decodes it in a tf.data Dataset element.
    example = tf.io.parse_single_example(
      # Data
      record_bytes,
      # Schema
      {"question__input_ids": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "question__attention_mask": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "question__index": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
       "answer__out_s": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "answer__out_e": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "paragraph__input_ids": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "paragraph__attention_mask": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "hard_paragraph__input_ids": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "hard_paragraph__attention_mask": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "paragraph__tokens_s": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "paragraph__tokens_e": tf.io.FixedLenFeature(shape=(MAX_SEQ_LEN,), dtype=tf.int64),
       "context__index": tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
       "paragraph__index": tf.io.FixedLenFeature(shape=(), dtype=tf.int64)})
    return {
      "questions": {'input_ids': example['question__input_ids'],
                    'attention_mask': example['question__attention_mask'],
                    'index': example['question__index']},
      "answers":   {'out_s': example['answer__out_s'],
                    'out_e': example['answer__out_e']},
      "paragraphs":{'input_ids': example['paragraph__input_ids'],
                    'attention_mask': example['paragraph__attention_mask'],
                    'tokens_s': example['paragraph__tokens_s'],
                    'tokens_e': example['paragraph__tokens_e']},
      "hard_paragraphs": {'input_ids': example['hard_paragraph__input_ids'],
                          'attention_mask': example['hard_paragraph__attention_mask']},
      "context_ids": (example['context__index'], example['paragraph__index'])
    }

def get_questions_and_paragraphs(dataset):
    '''
    Returns a processed set of questions and paragraphs given a dataset.
    '''
    questions = [{
            'qas': qas,
            'context_id': (i,j)    # We also track the question's context index 
                                   # so we are able to match a question to its 
                                   # groundtruth paragraph.
        }
        for i in range(len(dataset))
        for j, para in enumerate(dataset[i]['paragraphs'])
        for qas in para['qas']
    ]

    paragraphs = [{
            'context': para['context'],
            'context_id': (i,j)
        }
        for i in range(len(dataset))
        for j, para in enumerate(dataset[i]['paragraphs'])
    ]

    return questions, paragraphs

def get_paragraph_from_question(qas, dataset):
    '''
    Obtain the text of a paragraph given the related question.
    '''
    i,j = qas['context_id']
    return dataset[i]['paragraphs'][j]

def get_context_ids_from_top_indices(paragraphs_set, topn_indices):
    '''
    Maps the retrieved paragraph indexes to their `context_id`s so that we can 
    check whether the question and retrieved paragraphs have the same ID.
    '''
    return [topn_parag['context_id'] for topn_parag in np.take(paragraphs_set, topn_indices)]

def top_n_for_question_neural(tok_q, paragraphs_encodings, model, n=100):
    '''
    Returns the top n paragraphs in a set of encoded paragraphs given a tokenized question
    and a model that can deal with it.
    '''
    sample_q_repr = model(tok_q)[:,0,:].numpy()
    scores = np.dot(sample_q_repr, paragraphs_encodings.T)
    topn_indices = np.argsort(scores, axis=1)[0, -n:]
    return topn_indices

def top_n_for_question_tfidf(vectorizer, query, docs, n=100):
    '''
    Returns the top n paragraphs in a set of paragraphs encoded through a vectorizer (docs) 
    given a question and a vectorizer that can deal with it.
    '''
    q = query['qas']['question']
    q = vectorizer.transform([q])
    scores = np.asarray(np.dot(docs, q.T).todense()).flatten()
    sorted_scores = np.argsort(scores) # Negated scores for descending order
    return sorted_scores[-n:]