import torch
import itertools


class ConfigurationFeaturesComputer:
    def compute_feature(self, configuration):
        raise NotImplementedError


class ConfigurationFeaturesComputerConcat(ConfigurationFeaturesComputer):
    """
    This class implement the logics for combining the features from the stack and buffer to take the decision.
    Here the logic is to concatenate the n top element of the stack and the first (next) element of the buffer.

    >>> from parsebrain.processing.dependency_parsing.transition_based.transition.configuration import Configuration
    >>> x = ConfigurationFeaturesComputerConcat(2, 10)
    >>> conf = Configuration()
    >>> conf.buffer = [torch.ones(10)]
    >>> conf.stack = [torch.zeros(10), torch.ones(10)]
    >>> y = x.compute_feature(conf)
    >>> y
    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>>

    >>> from parsebrain.processing.dependency_parsing.transition_based.transition.configuration import Configuration
    >>> x = ConfigurationFeaturesComputerConcat(4, 10)
    >>> conf = Configuration()
    >>> conf.buffer = [torch.ones(10)]
    >>> conf.stack = [torch.zeros(10), torch.ones(10)]
    >>> y = x.compute_feature(conf)
    >>> y
    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>>
    """

    def __init__(self, stack_depth, dim):
        super().__init__()
        self.stack_depth = stack_depth
        self.dim = dim

    def compute_feature(self, configuration):
        '''
        May need to change a few things with batched parser
        :param configuration:
        :return:
        '''
        emb_buffer = configuration.buffer[0]
        emb_stack = configuration.stack[-self.stack_depth:]  # maybe reverse it ?
        if len(emb_stack) != self.stack_depth:  # Need to add blank tensor
            emb_stack = [x.unsqueeze(0) for x in emb_stack]
            emb_stack.append(torch.zeros((self.stack_depth - len(emb_stack), self.dim)))
        emb_stack = torch.cat(emb_stack).view(-1)
        result = torch.cat((emb_buffer, emb_stack))
        return result


if __name__ == "__main__":
    import doctest
    doctest.testmod()
