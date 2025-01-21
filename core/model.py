from abc import abstractmethod

import keras

import core.embeddings
import core.layer as layer
from core.utils import max_margin_loss


class ModelGenerator:
    @abstractmethod
    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        pass

    @abstractmethod
    def generate_model(self, existing_model_path: str = None, is_train: bool = True) -> keras.Model:
        pass


class ABAEGenerator(ModelGenerator):
    def __init__(self, max_seq_length: int, negative_length: int, embeddings_model: core.embeddings.Embedding,
                 aspect_embeddings_model: core.embeddings.AspectEmbedding):
        self.max_seq_length = max_seq_length
        self.negative_length = negative_length

        self.emb_model = embeddings_model
        self.aspect_emb_model = aspect_embeddings_model

    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        positive_input_shape = (self.max_seq_length,)  # 512
        negative_input_shape = (self.negative_length, self.max_seq_length)

        pos_input_layer = keras.layers.Input(shape=positive_input_shape, name='positive', dtype='int32')
        neg_input_layer = keras.layers.Input(shape=negative_input_shape, name='negative', dtype='int32')

        emb_layer = self.emb_model.build_embedding_layer(layer_name="word_embedding")

        embeddings = emb_layer(pos_input_layer)
        average = layer.Average()(embeddings)  # (64, 1017, 128) -> (64, 128) Avg of the embeddings

        negative_embeddings = emb_layer(neg_input_layer)
        neg_average = layer.Average()(negative_embeddings)  # (64, 10, 1017, 128) -> (64, 128)

        att_weights = layer.Attention(name='att_weights')([embeddings, average])
        weighted_positive = layer.WeightedSum()([embeddings, att_weights])

        aspect_size = self.aspect_emb_model.aspect_size
        dense_layer = keras.layers.Dense(units=aspect_size, activation='softmax')(weighted_positive)
        aspect_embeddings = self.aspect_emb_model.build_embedding_layer("aspect_embedding")(dense_layer)

        output = layer.MaxMargin(name="max_margin")([weighted_positive, neg_average, aspect_embeddings])

        # Model outputs: [Loss, AttentionWeights, AspectProbability]
        return [pos_input_layer, neg_input_layer], [output, att_weights, dense_layer]

    def generate_training_model(self, existing_model_path: str = None):
        if existing_model_path is not None:
            try:
                custom_objects = {'max_margin_loss': max_margin_loss}
                template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)

                model = keras.Model(inputs=template_model.inputs, outputs=template_model.outputs[0])
                # Transfer properties of the template model.
                model.compile(
                    optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics
                )

                return model

            except Exception as error:
                # We keep going and simply generate a new model if we fail in finding the one in the path provided
                print(error)

        inputs, outputs = self.make_layers()
        return keras.Model(inputs=inputs, outputs=outputs[0])

    def generate_model(self, existing_model_path: str = None, is_train: bool = True) -> keras.Model:
        if is_train:  # Give the training model on demand.
            return self.generate_training_model(existing_model_path=existing_model_path)

        if existing_model_path is None:
            raise FileNotFoundError("Cannot load inference model from fs as it is missing")
        # todo delegare a train parte di questo?
        custom_objects = {'max_margin_loss': max_margin_loss}
        template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)

        outputs = template_model.outputs

        # If the previously stored model was a training model I have to build the correct new output shape.
        if len(template_model.outputs) == 1:
            outputs = [template_model.outputs[0], template_model.layers[3].output, template_model.layers[6].output]

        model = keras.Model(inputs=template_model.inputs, outputs=outputs)
        model.compile(
            optimizer=template_model.optimizer, loss=template_model.loss, metrics={'max_margin': max_margin_loss}
        )

        return model
