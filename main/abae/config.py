from dataclasses import dataclass, fields
from pathlib import Path

from main.config import BaseConfig


@dataclass
class ABAEGeneratorConfig(BaseConfig):
    max_seq_len: int = 80
    negative_sample_size: int = 20
    embedding_size: int = 100
    aspect_size: int = 14


@dataclass
class ABAEManagerConfig(ABAEGeneratorConfig):
    min_word_count: int = 5
    max_vocab_size: int | None = None
    batch_size: int = 128
    epochs: int = 15
    # My ABAE implementation like the paper proposed only uses Adam
    learning_rate: float = 1e-3
    output_folder: str = "./output"

    def output_path(self):
        path = f"{self.output_folder}/{self.name}"
        # If the folder does not exist we take care of it immediately.
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
