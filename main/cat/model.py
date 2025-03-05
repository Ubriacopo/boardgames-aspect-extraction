from collections import defaultdict

from main.model import ModelGenerator


class CAt:
    def candidate(self, embeddings, adjectives: list[tuple], nouns, seed_words, nouns_num: int, min_count):
        unique_adjectives = list(set(adjectives))
        similarities = embeddings.similarity(adjectives, seed_words).max(1)

        adjective_scores = dict(zip(unique_adjectives, similarities))
        noun_scores = defaultdict(lambda: [0, 0])

        for adjective, noun in zip(adjectives, nouns):
            noun_scores[noun][0] += adjective_scores[adjective]
            noun_scores[noun][1] += 1

        noun_scores = {k: v[0] for k, v in noun_scores.items() if v[1] > min_count}
        return sorted(noun_scores.items(), key=lambda x: x[1])[-nouns_num:]


class CAtGenerator(ModelGenerator):

    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None):
        pass

    def generate_training_model(self, custom_objects: dict, existing_model_path: str = None):
        return self.generate_inference_model(custom_objects, existing_model_path)
