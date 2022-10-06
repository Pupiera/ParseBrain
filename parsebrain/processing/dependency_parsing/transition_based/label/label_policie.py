import torch
class LabelPolicie:
    '''
    This abstract class is used to setup different way of computing the label
    of the relation between two word.
    '''

    def compute_representation(self, config, decision, transition):
        raise NotImplementedError("Subclass need to implement function get_label")


class LabelPolicieEmbedding(LabelPolicie):
    '''
    This class take the embedding of the two word with a relation
    and concatenate their embedding. The first embedding is always the Head,
    and the second is the dependent.
    '''

    def compute_representation(self, config, decision, transition):
        head, dependent = transition.get_relation_from_decision(decision, config)
        rep = torch.cat((head, dependent))
        return rep

