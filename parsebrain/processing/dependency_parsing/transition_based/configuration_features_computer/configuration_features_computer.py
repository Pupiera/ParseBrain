from typing import List

import torch


class ConfigurationFeaturesComputer:
    def compute_feature(self, stack: List, buffer: List, device: str):
        raise NotImplementedError


class ConfigurationFeaturesComputerConcat(ConfigurationFeaturesComputer):
    """
    This class implement the logics for combining the features from the stack and buffer to take the decision.
    Here the logic is to concatenate the n top element of the stack and the first (next) element of the buffer.

    >>> x = ConfigurationFeaturesComputerConcat(2, 10)
    >>> buffer = [[torch.ones(10)]]
    >>> stack = [[torch.zeros(10), torch.ones(10)]]
    >>> y = x.compute_feature(stack, buffer ,"cpu")
    >>> y
    tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    >>>

    >>> x = ConfigurationFeaturesComputerConcat(4, 10)
    >>> buffer = [[torch.ones(10)]]
    >>> stack = [[torch.zeros(10), torch.ones(10)]]
    >>> y = x.compute_feature(stack, buffer, "cpu")
    >>> y
    tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    >>>
    """

    def __init__(self, stack_depth: int, dim: int):
        super().__init__()
        self.stack_depth = stack_depth
        self.dim = dim

    # toDo: Rework this function (replace with compute_feature_batch_fixed_size when finished and optimized)
    def compute_feature(self, stack: List, buffer: List, device: str):
        """
        Return a tensor of shape [batch, (1+self.stack_depth)*dim]
        Adapted for fully connected.
        >>> x = ConfigurationFeaturesComputerConcat(2, 10)
        >>> stack = [[torch.ones(10)*3], []]
        >>> buffer = [[torch.ones(10)*5],[]]
        >>> computer = ConfigurationFeaturesComputerConcat(2, 10)
        >>> x = computer.compute_feature(stack, buffer, "cpu")
        >>> x
        tensor([[5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 3., 3., 3., 3., 3., 3., 3., 3.,
                 3., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        >>> x.shape
        torch.Size([2, 30])
        """
        # ToDo: replace torch.zeros with something allowing us to control padding value.
        # ToDo: Allow different buffer window
        batch_size = len(stack)
        emb_stack = torch.zeros((batch_size, self.stack_depth, self.dim)).to(device)
        emb_buffer = torch.zeros((batch_size, self.dim)).to(device)
        for i, x in enumerate(stack):
            try:
                tmp_stack = torch.stack(x[-self.stack_depth :])
                emb_stack[i, 0 : tmp_stack.shape[0], :] = tmp_stack
            except RuntimeError:
                # happen if the stack is empty, nothing to add
                pass
        for i, x in enumerate(buffer):
            x = x[0:1]
            try:
                tmp_buffer = torch.stack(x)
                emb_buffer[i, :] = tmp_buffer
            except RuntimeError:
                # happen if the buffer is empty
                pass
            except TypeError:
                try:
                    emb_buffer[i, :] = x
                except RuntimeError:
                    pass
                    # do nothing
        return torch.cat((emb_buffer, emb_stack.reshape(batch_size, -1)), dim=1)

        """
        tmp_stack = []
        for x in stack:
            try:
                tmp_stack.append(torch.stack(x[-stack_depth:]))
            except RuntimeError:
                # x[-stack_depth:] empty
                tmp_stack.append(torch.zeros(size=(1, self.dim)).to(device))

        e_stack = torch.nn.utils.rnn.pad_sequence(
            tmp_stack,
            batch_first=True,
            padding_value=0.0,
        )

        emb_stack[:, 0 : emb_stack.shape[1], :].reshape(batch_size, -1).to(device)
        return emb_stack
        """


class ConfigurationFeaturesComputerConcatRNN(ConfigurationFeaturesComputerConcat):
    def compute_feature(self, stack: List, buffer: List, device: str):
        # toDo: Newest is at the end of tensor... Should it be reversed ?
        """
        Return a tensor of shape [batch, 1+ len(padded(stack)), dim]
        Where the first element of the sequence is the buffer state and the rest is the stack.
        Adapted for RNN

        >>> import torch
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import Configuration
        >>> buffer = [[torch.tensor([19,20,21]),torch.tensor([21,22,23])],[torch.tensor([24,25,26])]]
        >>> stack = [[torch.tensor([1,2,3]),torch.tensor([4,5,6])],
        ... [torch.tensor([7,8,9]),torch.tensor([10,11,12]),torch.tensor([13,14,15]),torch.tensor([16,17,18])]]
        >>> computer = ConfigurationFeaturesComputerConcatRNN(stack_depth = 3, dim=3)
        >>> computer.compute_feature(stack ,buffer, "cpu")
        tensor([[[19, 20, 21],
                 [ 1,  2,  3],
                 [ 4,  5,  6],
                 [ 0,  0,  0]],
        <BLANKLINE>
                [[24, 25, 26],
                 [10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18]]])
        """
        stack_depth = self.stack_depth
        tmp_stack = []
        for x in stack:
            try:
                tmp_stack.append(torch.stack(x[-stack_depth:]))
            except RuntimeError:
                # x[-stack_depth:] empty
                tmp_stack.append(torch.zeros(size=(1, self.dim)).to(device))

        emb_stack = torch.nn.utils.rnn.pad_sequence(
            tmp_stack,
            batch_first=True,
            padding_value=0.0,
        ).to(device)
        # if need to be able to take multiple element of buffer, update this. (remove unsqueeze and edit x[0])
        tmp_buffer = []
        for x in buffer:
            try:
                tmp_buffer.append(x[0])
            except IndexError:
                tmp_buffer.append(torch.zeros(self.dim).to(device))

        emb_buffer = torch.stack(tmp_buffer).unsqueeze(1).to(device)
        result = torch.cat((emb_buffer, emb_stack), dim=1)
        return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
