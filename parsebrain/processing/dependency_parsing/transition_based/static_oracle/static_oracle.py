from typing import List

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    GoldConfiguration,
)


class StaticOracle:
    def compute_sequence(self, gold_config: List[GoldConfiguration]) -> List[List[int]]:
        raise NotImplementedError("subclass need to implement this functions")
