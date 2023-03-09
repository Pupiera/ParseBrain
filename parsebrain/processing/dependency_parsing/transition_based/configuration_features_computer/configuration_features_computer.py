from typing import List

import torch


class ConfigurationFeaturesComputer:
    def compute_feature(self, stack: List, buffer: List, device: str):
        raise NotImplementedError


class ConfigurationFeaturesComputerConcat(ConfigurationFeaturesComputer):
    """
    This class implement the logics for combining the features from the stack and buffer to take the decision.
    Here the logic is to concatenate the n top element of the stack and the first (next) element of the buffer.
    """

    def __init__(self, stack_depth: int, dim: int):
        super().__init__()
        self.stack_depth = stack_depth
        self.dim = dim

    def compute_feature(
        self, stack: List[List[int]], buffer: List[torch.Tensor], device: str
    ):
        """
        Return a tensor of shape [batch, (1+self.stack_depth)*dim]
        Adapted for fully connected.
        This version is around 5% better than the previous one on GPU. (slower on cpu)
        >>> x = ConfigurationFeaturesComputerConcat(2, 10)
        >>> stack = [[torch.ones(10)*3], []]
        >>> buffer = [torch.ones((5,10)*5),[]]
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
        # todo: find way to remove for loop...
        n_buffer = 1
        batch_size = len(stack)
        emb_stack = torch.zeros((batch_size, self.stack_depth, self.dim)).to(device)
        emb_buffer = torch.zeros((batch_size, n_buffer, self.dim)).to(device)

        stack_list = [torch.stack(x[-self.stack_depth :]) if x else [] for x in stack]
        for i, s in enumerate(stack_list):
            if len(s) > 0:
                emb_stack[i, 0 : s.shape[0], :] = s
        buffer_list = [x[-n_buffer:] for x in buffer]
        for i, b in enumerate(buffer_list):
            if len(b) > 0:
                emb_buffer[i, 0 : b.shape[0], :] = b
        result = torch.cat((emb_buffer, emb_stack), dim=1)
        return result.reshape(batch_size, -1)


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
        batch_size = len(stack)
        stack_depth = self.stack_depth
        tmp_stack = []
        for x in stack:
            try:
                x = torch.stack(x[-stack_depth:])
                tmp_stack.append(torch.reshape(x, (stack_depth, self.dim)))
            except RuntimeError:
                # x[-stack_depth:] empty
                tmp_stack.append(torch.zeros(size=(stack_depth, self.dim)).to(device))
        emb_stack = torch.nn.utils.rnn.pad_sequence(
            tmp_stack,
            batch_first=True,
            padding_value=0.0,
        ).to(device)
        emb_stack = torch.reshape(emb_stack, (batch_size, stack_depth, self.dim))
        # emb_stack = torch.reshape(emb_stack, )
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
