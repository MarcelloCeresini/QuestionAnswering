from transformers import DistilBertTokenizerFast, TFDistilBertModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import spacy

class Config():
    '''
    A single object to contain all constants and some utility methods.
    '''    
    def __init__(self) -> None:
        self.RANDOM_SEED = 42       # The random seed for all random operations
        self.TRAIN_SPLIT = 0.75     # The percentage of training elements wrt the whole dataset
                                    # The rest (1 - TRAIN_SPLIT) will be the percentage of elements
                                    # in the validation set
        self.SMALL_TRAIN_LEN = 20   # Number of articles to use to build the small training set
        self.SMALL_VAL_LEN = 5      # Number of articles to use to build the small validation set
        self.BATCH_SIZE = 4        # Number of (question+context) pairs fed to the network for training
        self.VAL_BATCH_SIZE = 4    # number of (question+context) pairs fed to the network for validation

        self.HuggingFace_import = 'distilbert-base-uncased'         # Which model to instantiate from HuggingFace
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(   # Instance of the tokenizer
            self.HuggingFace_import
        )
        self.INPUT_LEN = 512        # The maximum sequence length for the model

        self.transformer_model = TFDistilBertModel.from_pretrained( # The instantiation of the transformer model
            self.HuggingFace_import, output_hidden_states = True
        )
        # if it doesn't work:
        # !pip3 install spacy
        # !python -m spacy download en_core_web_sm

        #### UNCOMMENT FOR NER
        # self.spacy_nlp = spacy.load("en_core_web_sm") # NOT NEEDED FOR NORMAL MODEL

    def create_standard_model(self, hidden_state_list=[3,4,5,6]) -> keras.Model:
        '''
        A utility method to create our "standard" model.

        Inputs:
            - The index or list of indexes of transformer hidden states to be 
                concatenated before the final classification (default: [3,4,5,6])
        
        Outputs:
            - The complete Keras model
        '''
        input_ids = tf.keras.Input(shape=(self.INPUT_LEN, ), 
            name="input_ids", dtype='int32'
        )
        attention_mask = tf.keras.Input(shape=(self.INPUT_LEN, ), 
            name="attention_mask", dtype='int32'
        )
        # token_type_ids = tf.keras.Input(shape=(SHAPE_ATTENTION_MASK, ), dtype='int32') # uncomment if using BERT

        transformer = self.transformer_model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "token_type_ids": token_type_ids # uncomment if using BERT
            }
        )

        hidden_states = transformer.hidden_states
        
        if isinstance(hidden_state_list, int):
            chosen_hidden_states = hidden_states[hidden_state_list]
        elif len(hidden_state_list) == 1:
            chosen_hidden_states = hidden_states[hidden_state_list[0]]
        else:
            chosen_hidden_states = layers.concatenate(
                tuple([hidden_states[i] for i in hidden_state_list])
            )
        
        out_S = layers.Dense(1)(chosen_hidden_states) # Dot product between token representation and start vector
        out_S = layers.Flatten()(out_S)
        out_S = layers.Softmax(name="out_S")(out_S)

        out_E = layers.Dense(1)(chosen_hidden_states) # Dot product between token representation and end vector
        out_E = layers.Flatten()(out_E)
        out_E = layers.Softmax(name="out_E")(out_E)

        return keras.Model(
            inputs=[input_ids, attention_mask],
            outputs = [out_S, out_E]
        )
        