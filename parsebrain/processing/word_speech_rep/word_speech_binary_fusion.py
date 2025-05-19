import torch


class WordSpeechBinaryFusion(torch.nn.Module):
    def __init__(self, score_model, combine_model, fusion_threshold=0.5):
        super().__init__()
        self.score_model = score_model
        self.combine_model = combine_model
        self.fusion_threshold = fusion_threshold

    def forward(self, frame_input: torch.tensor):
        """
        While there is a pir where the score is above the threshold, do the compute_word_rep_step(
        @param frame_input: [Batch, seq_len, dim]
        @return:
        """

        x = frame_input
        pair_score = self.score_pair(x)
        while (pair_score >= self.fusion_threshold).any():
            pair_to_fuse = self.pair_to_fuse(pair_score)
            x = self.compute_word_rep_step(x, pair_to_fuse)
            pair_score = self.score_pair(x)

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
        tensor([[False,  True, False,  True],
                [ True, False, False, False]])

        """
        pair_to_fuse = pair_scores > self.fusion_threshold
        for i in range(pair_to_fuse.size(0)):
            for y in range(pair_to_fuse.size(1) - 1):
                if pair_to_fuse[i][y] and pair_to_fuse[i][y + 1]:
                    pair_to_fuse[i][y + 1] = False
        return pair_to_fuse

    def compute_word_rep_step(self, input_frame, pair_to_fuse):
        """
        By using the score of each pair, fuse the representation when score above a certain threshold using the combine_model
        @param input_frame: [batch, seq_len, dim]
        @param pair_to_fuse: boolean tensor : [batch, seq_len-1]
        @return: fusion_repreentation : [batch, seq_len-nb_fusion, dim]
        >>> input_frame = torch.randn((2, 5, 2))
        >>> pair_score = torch.tensor([[False,True, False, True], [True, False, False, False]])
        >>> fusion = WordSpeechBinaryFusion(score_model=torch.nn.Linear(2*2,1), combine_model=torch.nn.Linear(2*2,2))
        >>> fusion.compute_word_rep_step(input_frame, pair_score)
        """

        extracted_pairs = input_frame.unfold(dimension=1, size=2, step=1)
        # get the pairs, but no idea to which element of the batch it belong...
        extracted_pairs = extracted_pairs[pair_to_fuse]
        extracted_pairs = extracted_pairs.reshape((extracted_pairs.shape[0], -1))
        new_rep = self.combine_model(extracted_pairs)
        num_fused_pairs = pair_to_fuse.sum(dim=1)
        min_fused_pairs = num_fused_pairs.min().item()

        new_x = torch.zeros(
            input_frame.size(0),
            input_frame.size(1) - min_fused_pairs,
            input_frame.size(2),
            device=input_frame.device,
        )
        index_to_replace = (pair_to_fuse == True).nonzero(as_tuple=False)
        for i in range(input_frame.size(0)):
            nb_replaced = 0
            replaced_indice = []
            last_y = -1
            for y in range(input_frame.size(1)):
                replaced = False
                for pos, x in enumerate(index_to_replace):
                    # replace by fused rep
                    if i == x[0] and y == x[1] - nb_replaced:
                        new_x[i, y] = new_rep[pos]
                        nb_replaced += 1
                        last_y = y
                        replaced = True
                        replaced_indice.append(x[1].item())
                        replaced_indice.append(x[1].item() + 1)
                if not replaced and y not in replaced_indice:
                    last_y = last_y + 1
                    new_x[i, last_y] = input_frame[i, y]

        return new_x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
