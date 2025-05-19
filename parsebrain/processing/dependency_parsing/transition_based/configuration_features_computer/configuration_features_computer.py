from typing import List

import torch


class ConfigurationFeaturesComputer:
    def compute_feature(self, 
            stack: List, 
            buffer: List, 
            has_root: List, 
            root_embedding: torch.Tensor, 
            device: str):
        raise NotImplementedError


class ConfigurationFeaturesComputerSequence(ConfigurationFeaturesComputer):
    def __init__(self, stack_depth: int, dim: int, embedding, n_buffer: int = 1 ):
        #super().__init__(stack_depth, dim, embedding)
        super().__init__()
        self.stack_depth = stack_depth
        self.n_buffer = n_buffer
        self.dim = dim
        self.emb = embedding
        self.STACK_PADDING_INDEX = 1
        self.BUFFER_PADDING_INDEX = 2


    def compute_feature(self, stack: List, buffer: List[torch.Tensor], has_root: List, root_embedding: List[torch.Tensor] , device: str):
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
        >>> feat = computer.compute_feature(stack ,buffer, "cpu")
        >>> feat
        tensor([[[19, 20, 21],
                 [ 1,  2,  3],
                 [ 4,  5,  6],
                 [ 0,  0,  0]],
        <BLANKLINE>
                [[24, 25, 26],
                 [10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18]]])
        >>> feat.shape
        """
        batch_size = len(stack)
        stack_depth = self.stack_depth
        tmp_stack = []
        for x, root_emb in zip(stack, root_embedding):
            if len(x) != 0:
                # take everything it can until stack_depth
                x = torch.stack(x[-stack_depth:])
                if len(x) < stack_depth:
                    x = torch.cat((x, root_emb), dim=0)
                tmp_stack.append(x)
            else:
                # x[-stack_depth:] empty
                padding_stack = torch.zeros(size=(stack_depth-1, self.dim)).to(device)
                tmp_stack.append(torch.cat((padding_stack, root_emb), dim=0))
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

        emb_buffer = torch.nn.utils.rnn.pad_sequence(
            tmp_buffer,
            batch_first=True,
            padding_value=0.0,
        ).to(device)

        # case where we only take the first element of buffer, need to add the seq_len dimension.
        if len(emb_buffer.shape) == 2:
            emb_buffer = torch.stack(tmp_buffer).unsqueeze(1).to(device)
        result = torch.cat((emb_buffer, emb_stack), dim=1)

        return result

class ConfigurationFeaturesComputerConcatFlat(ConfigurationFeaturesComputerSequence):
    """
    This class implement the logics for combining the features from the stack and buffer to take the decision.
    Here the logic is to concatenate the n top element of the stack and the first (next) element of the buffer.
    """

    def __init__(self, stack_depth: int, dim: int, n_buffer: int = 1, embedding=None):
        super().__init__(stack_depth = stack_depth,
                dim = dim,
                n_buffer = n_buffer,
                embedding = embedding)
        '''
        super().__init__()
        self.stack_depth = stack_depth
        self.n_buffer = n_buffer
        self.dim = dim
        self.emb = embedding
        self.STACK_PADDING_INDEX = 1
        self.BUFFER_PADDING_INDEX = 2
        '''

    
    def compute_feature(
        self,
        stack: List[List[int]],
        buffer: List[torch.Tensor],
        has_root: List,
        root_embedding: torch.Tensor, 
        device: str,
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
                [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        >>> x.shape
        torch.Size([2, 30])
        """
        n_buffer = self.n_buffer
        batch_size = len(stack)
        if self.emb is None:
            emb_stack = torch.zeros(
                (batch_size, self.stack_depth, self.dim), device=device
            )
            emb_buffer = torch.zeros((batch_size, n_buffer, self.dim), device=device)
        else:
            emb_stack = self.emb(
                torch.ones(
                    (batch_size, self.stack_depth), dtype=torch.long, device=device
                )
                * self.STACK_PADDING_INDEX
            )
            emb_buffer = self.emb(
               torch.ones((batch_size, n_buffer), dtype=torch.long, device=device)
                * self.BUFFER_PADDING_INDEX
            )

        # Root embedding, if not enough element in stack, then last element of stack is root
        not_has_root = [not x for x in has_root]
        emb_stack[not_has_root, -1, :] = root_embedding[not_has_root, :]
        #import pudb; pudb.set_trace()

        stack_list = [torch.stack(x[-self.stack_depth :]) if x else [] for x in stack]
        # first element in emb_stack is the last element to be poped on the stack,
        # last is the current element on top of the stack
        #maybe should be reversed since the root is supposed to be the last of last element of the stack (only available when stack is empty)

        # to optimize in a batch way, replace the loop
        # best way probably to compute mask based on the length
        for i, s in enumerate(stack_list):
            if len(s) > 0:
                emb_stack[i, 0 : s.shape[0], :] = s
        buffer_list = [x[:n_buffer] for x in buffer]
        # first element in emb_buffer is the first element on the buffer
        for i, b in enumerate(buffer_list):
            if len(b) > 0:
                emb_buffer[i, 0 : b.shape[0], :] = b
        result = torch.cat((emb_buffer, emb_stack), dim=1)
        result = result.reshape((batch_size, -1))
        return result

    def compute_feature_todo(
        self,
        stack: List[List[int]],
        buffer: List[torch.Tensor],
        has_root: List,
        root_embedding: List[torch.Tensor], 
        device: str,
    ):
        concat_seq = super().compute_feature(stack, buffer, has_root, root_embedding, device)
        batch_size = concat_seq.shape[0]
        #flatten the sequence and embedding dim, should be [batch_size, (stack_depth+buffer_depth)*emb_dim]
        res = torch.reshape(concat_seq, (batch_size, -1))
        assert res.shape[-1] == (self.stack_depth + self.n_buffer) * self.dim, f"{res.shape}, {concat_seq.shape}"
        return res
    
    



if __name__ == "__main__":
    import doctest

    doctest.testmod()
