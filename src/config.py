from transformers import DistilBertTokenizerFast, TFDistilBertModel
import tensorflow as tf
from tensorflow.keras import layers



class ConfigFile():
    
    def __init__(self) -> None:
        self.RANDOM_SEED = 42
        self.TRAIN_SPLIT = 0.75
        self.SMALL_TRAIN_LEN = 20   # number of articles to use to build the small training set
        self.SMALL_VAL_LEN = 5      # number of articles to use to build the small validation set
        self.BATCH_SIZE = 64        # number of question+context pairs fed to the network for training
        self.VAL_BATCH_SIZE = 64        # number of question+context pairs fed to the network for validation

        self.HuggingFace_import = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.HuggingFace_import)
        self.INPUT_LEN = 512

        self.transformer_model = TFDistilBertModel.from_pretrained(self.HuggingFace_import, output_hidden_states = True)


    def create_standard_model(self, hidden_state_list):

        input_ids = tf.keras.Input(shape=(self.INPUT_LEN, ), name="input_ids", dtype='int32')
        attention_mask = tf.keras.Input(shape=(self.INPUT_LEN, ), name="attention_mask", dtype='int32')
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
        
        out_S = layers.Dense(1)(chosen_hidden_states) # dot product between token representation and start vector
        out_S = layers.Flatten()(out_S)
        out_S = layers.Softmax(name="out_S")(out_S)

        out_E = layers.Dense(1)(chosen_hidden_states) # dot product between token representation and end vector
        out_E = layers.Flatten()(out_E)
        out_E = layers.Softmax(name="out_E")(out_E)

        return tf.keras.models.Model(
            inputs=[input_ids, attention_mask],
            outputs = [out_S, out_E]
        )
        