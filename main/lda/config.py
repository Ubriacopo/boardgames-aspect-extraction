from dataclasses import dataclass

from main.config import BaseConfig


@dataclass
class LdaGeneratorConfig(BaseConfig):
    topics: int = 14
    random_state: int = 42
    chunk_size: int = 1000
    passes: int = 10
    alpha: float | str = 'symmetric'
    eta: float = 0.01