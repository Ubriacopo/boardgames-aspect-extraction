import pickle

import numpy as np
from sklearn.cluster import KMeans


class AspectEmbedding:
    def __init__(self, aspect_size: int, emb_size: int):
        self.aspect_size: int = aspect_size
        self.embedding_size: int = emb_size
        # We initialize the aspect weights on the centroids of the Kmeans learning algorithm
        self.model: KMeans | None = None

    def load_existing(self, file_path: str):
        print(f"Loading the existing found model as requested in path {file_path}")
        self.model = pickle.load(open(file_path, "rb"))
        return self.model

    def persist(self, file_path: str):
        pickle.dump(self.model, open(file_path, "wb"))

    def generate(self, embedding_weights):
        print("Creating new model")
        self.model = KMeans(n_clusters=self.aspect_size, verbose=False)
        self.model.fit(embedding_weights)

        return self.model

    def weights(self):
        if self.model is None:
            raise 'You must first generate the model to get its weights.'

        aspect_m = self.model.cluster_centers_ / np.linalg.norm(self.model.cluster_centers_, axis=-1, keepdims=True)
        return aspect_m.astype(np.float32)

    def vocabulary(self) -> dict:
        """
        Our aspects have label (yet associated). We could opt for the most representative word, yet for that we have to
        calculate it or infer by hand the meaning, for now it simply is a number (its index).
        @return: The built vocabulary
        """
        return {value: value for value in range(self.aspect_size)}
