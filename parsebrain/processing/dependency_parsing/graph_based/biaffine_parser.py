import torch
from typing import Final

# Note:  BiAffine layer from original paper


class BiAffine(torch.nn.Module):
    """Biaffine attention layer.
    Inputs
    ------
    - `d` a tensor of shape `batch_size×num_dependents×input_dim
    - `h` a tensor of shape `batch_size×num_heads×input_dim
    Outputs
    -------
    A tensor of shape `batch_size×num_dependents×num_heads×output_dim`.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool):
        super(BiAffine, self).__init__()
        self.input_dim: Final[int] = input_dim
        self.output_dim: Final[int] = output_dim
        self.bias: Final[bool] = bias
        weight_input = input_dim + 1 if bias else input_dim
        self.weight = torch.nn.Parameter(
            torch.empty(output_dim, weight_input, weight_input)
        )
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, d: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.bias:
            d = torch.cat((d, d.new_ones((*d.shape[:-1], 1))), dim=-1)
            h = torch.cat((h, h.new_ones((*h.shape[:-1], 1))), dim=-1)
        return torch.einsum("bxi,oij,byj->bxyo", d, self.weight, h)


# Slightly simplified code from HOPSparser
class BiAffineParser(torch.nn.Module):
    def __init__(
        self,
        pos_tagger: torch.nn.Module,
        arc_h: torch.nn.Module,
        arc_d: torch.nn.Module,
        lab_h: torch.nn.Module,
        lab_d: torch.nn.Module,
        biased_biaffine: bool,
        input_size: int,
        num_labels: int,
    ):
        super().__init__()
        self.pos_tagger = pos_tagger

        # Arc MLPs
        self.arc_h = arc_h
        self.arc_d = arc_d
        # Label MLPs
        self.lab_h = lab_h
        self.lab_d = lab_d
        self.input_size = input_size
        self.num_labels = num_labels
        # BiAffine layers
        self.arc_biaffine = BiAffine(self.input_size, 1, bias=biased_biaffine)
        self.lab_biaffine = BiAffine(
            self.input_size, self.num_labels, bias=biased_biaffine
        )

    def forward(self, inpt: torch.Tensor):
        """
        inpt: The batch_first representation of each word of the sentence
        """
        pos_scores = self.pos_tagger(inpt)
        arc_h = self.arc_h(inpt)
        arc_d = self.arc_d(inpt)
        lab_h = self.lab_h(inpt)
        lab_d = self.lab_d(inpt)

        arc_scores = self.arc_biaffine(arc_d, arc_h).squeeze(-1)
        lab_scores = self.lab_biaffine(lab_d, lab_h)

        return pos_scores, arc_scores, lab_scores
