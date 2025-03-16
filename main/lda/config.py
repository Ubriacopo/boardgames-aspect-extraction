from dataclasses import dataclass
from pathlib import Path

from main.config import BaseConfig


@dataclass
class LdaGeneratorConfig(BaseConfig):
    topics: int = 14
    random_state: int = 42
    passes: int = 10

    alpha: float | str = 'symmetric'
    eta: float = None

    output_folder: str = "./output"

    def output_path(self):
        path = f"{self.output_folder}/{self.name}"
        # If the folder does not exist we take care of it immediately.
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
