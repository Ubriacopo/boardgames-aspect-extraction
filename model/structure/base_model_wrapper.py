from __future__ import annotations

from abc import abstractmethod

import keras


class BaseModelWrapper:
    """
    A learning algorithm with one or more hyperparameters is not really an algorithm, but rather
    a family of algorithms, one for each possible assignment of values to the hyperparameters.
        ~ Hyperparameter tuning and risk estimates, Nicolò Cesa-Bianchi.

    The idea is that in the base model family we have ways to customize the learning algorithm
    to be an instance of the many possible of its family.
    """

    @abstractmethod
    def make_layers(self, input_shape: (int, int, int)) -> tuple[keras.Layer, keras.Layer]:
        """
        It defines the main structure of the model.
        :param input_shape:
        :return:
        """
        pass

    def make_model(self, input_shape: int) -> keras.Model:
        input_layer, output_layer = self.make_layers(input_shape=input_shape)
        return keras.Model(inputs=input_layer, outputs=output_layer)
