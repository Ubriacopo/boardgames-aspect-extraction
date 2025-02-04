from abc import abstractmethod


class ModelGenerator:
    @abstractmethod
    def generate_training_model(self, custom_objects: dict, existing_model_path: str = None):
        pass

    @abstractmethod
    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None):
        pass
