from dataclasses import dataclass

import keras
from keras import Layer, Input

from main.model import ModelGenerator


@dataclass
class ABAEConfig:
    max_seq_len: int = 80
    negative_sample_size: int = 20


class ABAE(ModelGenerator):
    def __init__(self, config: ABAEConfig):
        self.c: ABAEConfig = config
        # The shape of the negative input layer
        self.negative_shape = (self.c.negative_sample_size, self.c.max_seq_len)

    def __make_layers(self) -> tuple[list[Layer], list[Layer]]:
        sentence_input = Input(shape=(self.c.max_seq_len,), name='positive', dtype='int32')
        n_sentences_input = Input(shape=self.negative_shape, name='negative', dtype='int32')

        # todo: Continua
        # todo prova con:
        # MultiHeadAttention if key query and value same self attention
        return [sentence_input, n_sentences_input], [output, attention_w, predicted_aspect]

    @staticmethod
    def restore_training_model(custom_objects: dict, existing_model_path: str):
        try:
            template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)

            model = keras.Model(inputs=template_model.inputs, outputs=template_model.outputs[0])
            # Transfer properties of the template model.
            model.compile(optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics)
            return model

        except Exception as error:
            # We keep going and simply generate a new model if we fail in finding the one in the path provided
            print(error)
            return None  # No model could be restored.

    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None):
        # We need an inference model as the training model takes a negative sample we dont want to pass further
        if existing_model_path is None:
            raise FileNotFoundError("Cannot load inference model from {fs} as it is missing")

        # custom_objects = {'max_margin_loss': max_margin_loss}
        template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)
        outputs = [template_model.get_layer('attention').output, template_model.get_layer('sentence_aspect').output]
        # If the previously stored model was a training model I have to build the correct new output shape.

        model = keras.Model(inputs=template_model.inputs[0], outputs=outputs)
        model.compile(optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics)

        return model

    def generate_training_model(self, custom_objects: dict, existing_model_path: str = None):
        if existing_model_path is not None:
            existing_model = ABAE.restore_training_model(custom_objects, existing_model_path)
            if existing_model is not None:
                # The model could be loaded with success
                return existing_model

        inputs, outputs = self.__make_layers()
        # The model is not compiled. Compilation is delegated
        return keras.Model(inputs=inputs, outputs=outputs[0])
