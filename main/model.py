from abc import abstractmethod

import keras


class ModelGenerator:
    @abstractmethod
    def generate_training_model(self, custom_objects: dict, existing_model_path: str = None):
        pass

    @abstractmethod
    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None):
        pass


class KerasModelGenerator(ModelGenerator):
    @abstractmethod
    def make_layers(self):
        pass

    @abstractmethod
    def get_input_output_layers(self, model, is_train: bool) -> tuple:
        pass

    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None):
        # We need an inference model as the training model takes a negative sample we dont want to pass further
        if existing_model_path is None:
            raise FileNotFoundError("Cannot load inference model from {fs} as it is missing")

        # custom_objects = {'max_margin_loss': max_margin_loss}
        template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)
        inputs, outputs = self.get_input_output_layers(template_model, is_train=False)
        # If the previously stored model was a training model I have to build the correct new output shape.¶
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics)
        return model

    def generate_training_model(self, custom_objects: dict, existing_model_path: str = None):
        if existing_model_path is not None:
            existing_model = self.restore_training_model(custom_objects, existing_model_path)

            if existing_model is not None:
                return existing_model  # The model could be loaded with success

        print("Making model from scratch...")
        inputs, outputs = self.make_layers()
        # The model is not compiled. Compilation is delegated
        return keras.Model(inputs=inputs, outputs=outputs[0])

    def restore_training_model(self, custom_objects: dict, existing_model_path: str):
        try:
            template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)
            inputs, outputs = self.get_input_output_layers(template_model, is_train=True)
            model = keras.Model(inputs=inputs, outputs=outputs)
            # Transfer properties of the template model.
            model.compile(optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics)
            return model

        except Exception as error:
            # We keep going and simply generate a new model if we fail in finding the one in the path provided
            print(error)
            return None  # No model could be restored.
