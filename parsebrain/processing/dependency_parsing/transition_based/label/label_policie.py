import torch
from typing import List
from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
)
from parsebrain.processing.dependency_parsing.transition_based.transition import (
    Transition,
)


class LabelPolicie:
    """
    This abstract class is used to setup different way of computing the label
    of the relation between two word.
    """

    def compute_representation(
        self, config: Configuration, decision: int, transition: Transition
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclass need to implement function compute_representation"
        )

    def compute_representation_batch(
        self, config: Configuration, decision: int, transition: Transition
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclass need to implement function compute_representation_batch"
        )


class LabelPolicieEmbedding(LabelPolicie):
    """
    This class take the embedding of the two word with a relation
    and concatenate their embedding. The first embedding is always the Head,
    and the second is the dependent.
    """

    def compute_representation(
        self, config: Configuration, decision: int, transition: Transition
    ) -> torch.Tensor:
        # slow part is torch cat.
        head, dependent = transition.get_relation_from_decision(decision, config)
        rep = torch.cat((head, dependent))
        return rep

    def compute_representation_batch(
        self, config: List[Configuration], decision: List[int], transition: Transition
    ) -> torch.Tensor:
        head, dep = [], []
        for c, d in zip(config, decision):
            h, d = transition.get_relation_from_decision(d, c)
            head.append(h)
            dep.append(d)
        head = torch.stack(head)
        dep = torch.stack(dep)
        rep = torch.cat((head, dep))
        return rep
