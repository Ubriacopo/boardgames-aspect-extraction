from dataclasses import dataclass, fields


@dataclass
class BaseConfig:
    """
    explain why: todo
    """
    name: str

    @classmethod
    def from_configuration(cls, name: str, objects: dict):
        instance = cls(name)
        [instance.__setattr__(f.name, objects[f.name]) for f in fields(instance) if f.name in objects]
        return instance
