import torch

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
        raise NotImplementedError("Subclass need to implement function get_label")


class LabelPolicieEmbedding(LabelPolicie):
    """
    This class take the embedding of the two word with a relation
    and concatenate their embedding. The first embedding is always the Head,
    and the second is the dependent.
    """

    def compute_representation(
            self, config: Configuration, decision: int, transition: Transition
    ) -> torch.Tensor:
        head, dependent = transition.get_relation_from_decision(decision, config)
        rep = torch.cat((head, dependent))
        return rep
