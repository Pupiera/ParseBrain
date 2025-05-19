import torch
from torch.nn.utils.rnn import pack_padded_sequence

class ViterbiLoss(torch.nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map = {}, length_normalization = False):
        """
        :param tag_map: tag map
        """
        super(ViterbiLoss, self).__init__()
        self.length_normalization = length_normalization
        if len(tag_map) == 0:
            print("tag_map not given in ViterbiLoss constructor, use .set_tag_map")
            return
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def set_tag_map(self, tag_map):
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def format_targets(self, targets, lengths):
        formated_targets = torch.zeros_like(targets, device = targets.device)
        for y, (t, l) in enumerate(zip(targets, lengths)):
            for i in range(l): 
                if i == 0:
                    prev = 0
                formated_targets[y, i] = prev * self.tagset_size + t[i]
                prev = t[i]
        return formated_targets
                    



    def forward(self, scores, targets, lengths):
        """
        Forward propagation.

        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :param lengths: sequence lengths
        :return: viterbi loss

        Need the scores to be sorted in reverse length order (longest at index 0)

        return 0 when the all_paths_score == gold_score, ie: all the probability are on the gold path.

        """

        batch_size = scores.size(0)
        word_pad_len = scores.size(1)
        device = scores.device


        # Gold score

        formated_targets = self.format_targets(targets, lengths)

        #targets = targets.unsqueeze(2)
        targets = formated_targets.unsqueeze(2)
        #it also take the scores of the padding... (but it is removed by pack_padded_sequence)
        # how does it know which one to take from the shape (batch, seq_len, 1) in a tensor fo shape (batch, seq_len, tag_set, tag_set ) ? 
        scores_at_targets = torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(
            2)  # (batch_size, word_pad_len)

        # Everything is already sorted by lengths
        scores_at_targets = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)

        # i think this should be also log sum exp (cause we want it to be equal to all path value if all the score is on the gold path, ie loss should be 0)
        gold_score = scores_at_targets.data.sum() 

        #normalize score by length
        '''
        print(scores_at_targets.data.shape)
        gold_score = gold_score / scores_at_targets.data.shape[0]
        '''
        # All paths' scores

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(device)


        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # normalize loss by length
        #print(scores_upto_t.shape)
        #all_paths_scores = scores_upto_t[:self.end_tag] / lengths
        # We only need the final accumulated scores at the <end> tag
        #all_paths_scores = scores_upto_t.sum()
        all_paths_scores = scores_upto_t[:, self.end_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        #length normalization
        if self.length_normalization: 
            viterbi_loss = viterbi_loss / lengths.sum()
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss




def log_sum_exp(tensor, dim):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))


if __name__ == "__main__":

    # shape (2, 4, 5, 5)
    scores = torch.tensor([[[[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]],

         [[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]]],


        [[[0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 1.],
          [0., 0., 0., 0., 0.]]]])
    gold_sup = torch.tensor([[1,2,2,4], [1,2,3,4]])
    tag_set = {"<start>": 0,
            "begin": 1,
            "inside": 2,
            "outside": 3,
            "<end>": 4}
    loss = ViterbiLoss(tag_set)
    val_loss = loss(scores, gold_sup, torch.tensor([4, 4]))
    print(f"loss : {val_loss}")

