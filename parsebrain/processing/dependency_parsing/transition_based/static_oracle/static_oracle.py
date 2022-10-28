from typing import List

import torch

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    GoldConfiguration,
)


class StaticOracle:
    def __init__(self, padding_value: int):
        self.padding_value = padding_value

    def compute_sequence(
        self, batch_gold_config: List[GoldConfiguration]
    ) -> torch.Tensor:
        raise NotImplementedError("subclass need to implement this functions")
