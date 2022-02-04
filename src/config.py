import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import spacy
import os
from transformers import BertTokenizerFast, TFBertModel, \
                         DistilBertTokenizerFast, TFDistilBertModel

class Config():
    '''
    A single object to contain all constants and some utility methods.
    '''    
    def __init__(self, bert=False) -> None:

        self.bert = bert
        self.ROOT_PATH = self.find_root_path(os.getcwd())   # The root path of the project, useful for using relative paths

        self.RANDOM_SEED = 42       # The random seed for all random operations
        self.TRAIN_SPLIT = 0.75     # The percentage of training elements wrt. the whole dataset
                                    # The rest (1 - TRAIN_SPLIT) will be the percentage of elements
                                    # in the validation set
        self.SMALL_TRAIN_LEN = 20   # Number of articles to use to build the small training set
        self.SMALL_VAL_LEN = 5      # Number of articles to use to build the small validation set
        
        self.BATCH_SIZE = 64        # Number of (question+context) pairs fed to the network for training
        self.VAL_BATCH_SIZE = 64    # number of (question+context) pairs fed to the network for validation

        if not self.bert:
            self.HuggingFace_import = 'distilbert-base-uncased'         # Which model to instantiate from HuggingFace
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(   # Instance of the tokenizer
                self.HuggingFace_import
            )
        else:
            self.HuggingFace_import = 'bert-base-uncased'         # Which model to instantiate from HuggingFace
            self.tokenizer = BertTokenizerFast.from_pretrained(   # Instance of the tokenizer
                self.HuggingFace_import
            ) 
        self.INPUT_LEN = 512        # The maximum sequence length for the model

        self.SAVE_PATH_TRAIN_DS_TRAINING_NER = os.path.join(self.ROOT_PATH, 
            "data", "full_datasets", "train_ds_training_NER")       # Save paths for complete datasets: train set with NER
        self.SAVE_PATH_VAL_DS_TRAINING_NER = os.path.join(self.ROOT_PATH, 
            "data", "full_datasets", "val_ds_training_NER")         # Save paths for complete datasets: validation set with NER

        self.SAVE_PATH_TRAIN_DS_TRAINING = os.path.join(self.ROOT_PATH, 
            "data", "full_datasets", "train_ds_training")           # Save paths for complete datasets: train set without NER
        self.SAVE_PATH_VAL_DS_TRAINING = os.path.join(self.ROOT_PATH,
             "data", "full_datasets", "val_ds_training")            # Save paths for complete datasets: validation set without NER

        self.SAVE_PATH_TRAIN_DS_INFERENCE_NER = os.path.join(self.ROOT_PATH, 
            "data", "full_datasets", "train_ds_inference_NER")      # Save paths for complete datasets: train set with NER
        self.SAVE_PATH_VAL_DS_INFERENCE_NER = os.path.join(self.ROOT_PATH, 
            "data", "full_datasets", "val_ds_inference_NER")        # Save paths for complete datasets: validation set with NER

        self.SAVE_PATH_TRAIN_DS_INFERENCE = os.path.join(self.ROOT_PATH, 
            "data", "full_datasets", "train_ds_inference")          # Save paths for complete datasets: train set without NER
        self.SAVE_PATH_VAL_DS_INFERENCE = os.path.join(self.ROOT_PATH, 
            "data", "full_datasets", "val_ds_inference")            # Save paths for complete datasets: validation set without NER

        # If spacy doesn't work:
        # !python -m spacy download en_core_web_sm
        self.ner_extractor = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", 
            "parser", "attribute_ruler", "lemmatizer"])

    def get_new_distilbert_transformer(self) -> TFDistilBertModel:
        '''
        This function returns a fresh instance of the transformer (DistilBert) model.
        '''
        return TFDistilBertModel.from_pretrained( # The instantiation of the transformer model
            self.HuggingFace_import, output_hidden_states = True
        )   

    def get_new_bert_transformer(self) -> TFBertModel:
        '''
        This function returns a fresh instance of the transformer (Bert) model.
        '''
        return TFBertModel.from_pretrained( # The instantiation of the transformer model
            self.HuggingFace_import, output_hidden_states = True
        )  

    def get_transformer(self):
        '''
        This function returns the appropriate transformer based on the `bert` flag.
        '''
        if self.bert:
            return self.get_new_bert_transformer()
        else:
            return self.get_new_distilbert_transformer()

    def find_root_path(self, current_path):
        '''
        A simple utility function to find the root path, which is useful when we
        want to use relative paths.
        '''
        flag_dir = False
        for dir in os.path.normpath(current_path).split(os.sep):
            if dir == "QuestionAnswering":
                flag_dir = True
                break
        if not flag_dir:
            raise FileNotFoundError("Move inside root directory and then launch the file")

        while current_path != "":
            if os.path.basename(os.path.normpath(current_path)) == "QuestionAnswering":
                return current_path
            else:
                current_path = os.path.dirname(current_path)
    
        return FileNotFoundError("Move inside root directory and then launch the file")


    def create_standard_model(self, hidden_state_list=[3,4,5,6]) -> keras.Model:
        '''
        A utility method to create our "standard" model.

        Inputs:
            - The index or list of indexes of transformer hidden states to be 
                concatenated before the final classification (default: [3,4,5,6])
        
        Outputs:
            - The complete Keras model
        '''
        # We have 2 inputs: the token IDs and the attention mask
        input_ids = tf.keras.Input(shape=(self.INPUT_LEN, ), 
            name="input_ids", dtype='int32'
        )
        attention_mask = tf.keras.Input(shape=(self.INPUT_LEN, ), 
            name="attention_mask", dtype='int32'
        )
        # In case BERT is used rather than DistilBERT, we should also have this third input
        if self.bert:
            token_type_ids = tf.keras.Input(shape=(self.INPUT_LEN, ), 
                name='token_type_ids', dtype='int32'
            )

        if self.bert:
            # We pass them to the transformer (that is instanced ex-novo)
            transformer = self.get_new_bert_transformer()(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
            )
        else:
            # We pass them to the transformer (that is instanced ex-novo)
            transformer = self.get_new_distilbert_transformer()(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    # "token_type_ids": token_type_ids # uncomment if using BERT
                }
            )

        # We care about the hidden states of the transformer
        hidden_states = transformer.hidden_states
        
        # We take the hiddent states we specified in the parameter to this function and
        # concatenate them
        if isinstance(hidden_state_list, int):
            chosen_hidden_states = hidden_states[hidden_state_list]
        elif len(hidden_state_list) == 1:
            chosen_hidden_states = hidden_states[hidden_state_list[0]]
        else:
            chosen_hidden_states = layers.concatenate(
                tuple([hidden_states[i] for i in hidden_state_list])
            )
        
        # Finally, in order to produce probabilities we:
        # 1. Compute a dot product between the token representation and a learnt start vector
        #    This is actually implemented as a simple Dense layer with 1 output and without
        #    an activation.
        # 2. The representation needs to be flattened, meaning that from a sequence of 512
        #    individual outputs we need to obtain a 512-d vector of dot products
        # 3. We apply a softmax function over this vector to obtain the probabilities of
        #    each token to be a starting token
        out_S = layers.Dense(1)(chosen_hidden_states) 
        out_S = layers.Flatten()(out_S)
        out_S = layers.Softmax(name="out_S", dtype='float32')(out_S)

        # The same is done for the end tokens.
        out_E = layers.Dense(1)(chosen_hidden_states)
        out_E = layers.Flatten()(out_E)
        out_E = layers.Softmax(name="out_E", dtype='float32')(out_E)

        # Return the model
        if self.bert:
            model = keras.Model(
                inputs=[input_ids, attention_mask, token_type_ids],
                outputs = [out_S, out_E]
            )
        else:
            model = keras.Model(
                inputs=[input_ids, attention_mask],
                outputs = [out_S, out_E]
            )
        return model
        