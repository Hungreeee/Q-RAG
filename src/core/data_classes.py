from dataclasses import dataclass, field
from typing import Any, Dict
from abc import ABC


@dataclass
class Node(ABC):
    id: str = field(default_factory=str)
    type_label: str = field(default_factory=str)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Question(Node):
    id: str
    type_label: str = "Question"

    def __post_init__(self):
        required_metadata_keys = [
            "text",
        ]

        for key in required_metadata_keys:
            if key not in self.metadata:
                raise ValueError(f"Missing required metadata key: '{key}'")


@dataclass
class Chunk(Node):
    id: str
    type_label: str = "Chunk"

    def __post_init__(self):
        required_metadata_keys = [
            "text",
        ]

        for key in required_metadata_keys:
            if key not in self.metadata:
                raise ValueError(f"Missing required metadata key: '{key}'")