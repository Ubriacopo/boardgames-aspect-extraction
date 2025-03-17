from dataclasses import dataclass, fields
from pathlib import Path

from main.config import BaseConfig


@dataclass
class ABAEGeneratorConfig(BaseConfig):
    max_seq_len: int = 80
    negative_sample_size: int = 20
    embedding_size: int = 200
    aspect_size: int = 14


@dataclass
class ABAEManagerConfig(ABAEGeneratorConfig):
    min_word_count: int = 5  # How often a word has to occur in the corpus to not be invalidated
    max_vocab_size: int | None = None  # Max number of distinct terms in the vocab size
    batch_size: int = 128 # Training step batch size
    epochs: int = 15 # Passes on full dataset during training
    # My ABAE implementation like the paper proposed only uses Adam
    learning_rate: float = 1e-3 # adam learning-rate
    output_folder: str = "./output" # In what folder the model is stored

    def output_path(self):
        path = f"{self.output_folder}/{self.name}"
        # If the folder does not exist we take care of it immediately.
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
