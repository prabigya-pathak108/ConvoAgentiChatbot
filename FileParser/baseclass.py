from abc import ABC, abstractmethod
from typing import Any

class FileParserBaseClass(ABC):
    @abstractmethod
    def parse(cls) -> Any:
        raise NotImplementedError
    