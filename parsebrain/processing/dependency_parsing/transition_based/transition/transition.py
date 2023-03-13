import torch

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
    Word,
)
from parsebrain.processing.dependency_parsing.transition_based.transition.arc import Arc
from typing import List


class Transition:
    def __init__(self):
        pass

    def get_relation_from_decision(
        self, decision: int, config: Configuration
    ) -> (torch.Tensor, torch.tensor):
        raise NotImplementedError("Subclass need to implement this function")

    def is_decision_valid(self, decision: int, config: Configuration) -> bool:
        raise NotImplementedError("Subclass need to implement this function")

    def apply_decision(self, decision: int, config: Configuration) -> Configuration:
        raise NotImplementedError("Subclass need to implement this function")

    @staticmethod
    def is_terminal(config: Configuration) -> bool:
        raise NotImplementedError("Subclass need to implement this function")

    def update_tree(self, decision: int, config: Configuration, tree: dict) -> dict:
        raise NotImplementedError("Subclass need to implement this function")

    def require_label(self, decision: int) -> bool:
        raise NotImplementedError("Subclass need to implement this function")

    def get_transition_dict(self) -> dict:
        raise NotImplementedError("Subclass need to implement this function")

    @staticmethod
    def has_head(wi: Word, arc: List[Arc]) -> bool:
        """
        The condition HEAD(wi ) is true in a configuration (σ, β, A) if A contains an arc wk → wi
        Check if wi already has a HEAD. ie. check if wi is the dependent in a dependency.
        :param wi:
        :param arc:
        :return:
        """
        for a in arc:
            if a.dependent == wi:
                return True
        return False
