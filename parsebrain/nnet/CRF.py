import torch

class CRF(torch.nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of previous NN output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = torch.nn.Linear(hidden_dim, self.tagset_size)
        self.transition = torch.nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        """
        Forward propagation.

        :param feats: tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(
            0)  # (batch_size, timesteps, tagset_size, tagset_size)
        return crf_scores

