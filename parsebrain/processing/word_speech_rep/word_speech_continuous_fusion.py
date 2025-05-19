import torch


class WordSpeechContinuousFusion(torch.nn.Module):
    def __init__(self, score_model, combine_model, fusion_threshold=0.5):
        super().__init__()
        self.score_model = score_model
        self.combine_model = combine_model
        self.fusion_threshold = fusion_threshold

    def forward(self, frame_input: torch.tensor):
        """
        @param frame_input: [Batch, seq_len, dim]
        @return:
        """

        x = frame_input
        pair_score = self.score_pair(x)
        pair_to_fuse = self.pair_to_fuse(pair_score)
        x = self.compute_word_rep_step(x, pair_to_fuse)
        return x

    def score_pair(self, x):
        """
        Compute the score of each consecutive pair of repreentation
        For sake of efficiency, a frame can only fused into one other representations.
        Assuring that the number of frame can decrease faster than 1 per step.
        @param x: [batch, seq_len, dim]
        @return: pair_scores : [batch, seq_len-1]
        >>> x = torch.randn((2, 5, 2))
        >>> fusion = WordSpeechBinaryFusion(score_model=torch.nn.Linear(2*2,1), combine_model=torch.nn.Linear(2*2,2))
        >>> fusion.score_pair(x).shape
        torch.Size([2, 4, 1])
        """
        # create pairs of consecutive frame, shape is [batch, seq_len-1, dim, 2]
        x_pairs = x.unfold(dimension=1, size=2, step=1)
        # reshape to [batch, seq_len-1, 2, dim]
        x_pairs = x_pairs.transpose(2, 3)
        x_pairs = x_pairs.reshape((x_pairs.shape[0], x_pairs.shape[1], -1))
        pair_scores = self.score_model(x_pairs)
        return pair_scores

    def pair_to_fuse(self, pair_scores):
        """

        @param pair_scores:
        @return:
        >>> pair_scores = torch.tensor([[0.2,0.6, 0.7, 0.9], [0.8, 0.2, 0.3, 0.1]])
        >>> fusion = WordSpeechBinaryFusion(score_model=torch.nn.Linear(2*2,1), combine_model=torch.nn.Linear(2*2,2))
        >>> fusion.pair_to_fuse(pair_scores)
        tensor([[False,  True, True,  True],
                [ True, False, False, False]])
        """
        pair_to_fuse = pair_scores > self.fusion_threshold
        return pair_to_fuse

    def compute_word_rep_step(self, input_frame, pair_to_fuse):
        """
        Fuse all consecutive frame marked with true using the given NN.
        The input is of variable length, so NN take variable input lenght input.
        So the NN must be either a RNN/CNN or a specific pooling method to get it to a fixed shape.
        @param input_frame: [batch, seq_len, dim]
        @param pair_to_fuse: boolean tensor : [batch, seq_len-1]
        @return: fusion_representation : [batch, word, dim]
        >>> input_frame = torch.randn((2, 5, 2))
        >>> pair_score = torch.tensor([[False,True, True, True], [True, False, True, True]])
        >>> fusion = WordSpeechContinuousFusion(score_model=torch.nn.Linear(2*2,1), combine_model=torch.nn.Linear(2*2,2))
        >>> fusion.compute_word_rep_step(input_frame, pair_score)
        """


if __name__ == "__main__":
    import doctest

    doctest.testmod()
