import torch
from torch.nn.utils.rnn import pad_sequence


class WordSpeechCtC(torch.nn.Module):
    def __init__(
        self, RepFusionModel, repFusionHidden, repFusionBidirectional, repFusionLayers
    ):
        super().__init__()
        self.RepFusionModel = RepFusionModel
        self.repFusionLayers = repFusionLayers
        self.repFusionBidirectional = repFusionBidirectional
        self.repFusionHidden = repFusionHidden

    def forward(self, frame_input, mapFrameToWord, device):
        """
        Compute the word audio embedding
        Parameters
        ----------
        frame_input : The encoder representation output ( 1 frame per 20ms with wav2vec)
        mapFrameToWord : The mapping of frame to word from the CTC module.

        Returns
        batch : The padded batch of word audio embedding [batch, max(seq_len)]
        seq_len : the length of each element of the batch
        -------

        """
        batch = []
        hidden_size = self.repFusionHidden
        is_bidirectional = self.repFusionBidirectional
        n_layers = self.repFusionLayers
        nb_hidden = n_layers * (1 + is_bidirectional)
        for i, (rep, map) in enumerate(
            zip(frame_input, mapFrameToWord)
        ):  # for 1 element on the batch do :
            map = torch.Tensor(map)
            uniq = torch.unique(map)
            fusionedRep = []
            # init hidden to zeros for each sentence
            hidden = torch.zeros(nb_hidden, 1, hidden_size, device=device)
            # init hidden to zeros for each sentence
            cell = torch.zeros(nb_hidden, 1, hidden_size, device=device)
            # For each word find the limit of the relevant sequence of frames
            for e in uniq:
                # ignore 0, if empty tensor, try with everything (i.e correspond to transition of words)
                if e.item() == 0 and len(uniq) > 1:
                    continue
                relevant_column = (map == e).nonzero(as_tuple=False)
                min = torch.min(relevant_column)
                max = torch.max(relevant_column)
                # should not break autograd https://discuss.pytorch.org/t/select-columns-from-a-batch-of-matrices-by-index/85321/3
                frames = rep[min : max + 1, :].unsqueeze(0)
                # extract feature from all the relevant audio frame representation
                _, (hidden, cell) = self.RepFusionModel(frames, (hidden, cell))
                if is_bidirectional:
                    fusionedRep.append(torch.cat((hidden[-2], hidden[-1]), dim=1))
                else:
                    fusionedRep.append(hidden[-1])
            batch.append(torch.stack(fusionedRep))
        seq_len = [len(e) for e in batch]
        batch = torch.reshape(
            pad_sequence(batch, batch_first=True),
            (len(mapFrameToWord), -1, hidden_size * (1 + is_bidirectional)),
        )
        return batch, torch.Tensor(seq_len)
