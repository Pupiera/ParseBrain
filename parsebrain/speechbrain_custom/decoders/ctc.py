import torch
from itertools import groupby
from speechbrain.dataio.dataio import length_to_mask


def filter_ctc_output(string_pred, blank_id=-1, space_id=1):
    """Apply CTC output merge and filter rules.
    Removes the blank symbol and output repetitions.
    Arguments
    ---------
    string_pred : list
        A list containing the output strings/ints predicted by the CTC system.
    blank_id : int, string
        The id of the blank.
    Returns
    -------
    list
        The output predicted by CTC without the blank symbol and
        the repetitions.
    list
        The maping of frame to word ( Feature grouping)
    Example
    -------
    >>> string_pred = ['a','a','blank','b','b','blank','c']
    >>> string_out = filter_ctc_output(string_pred, blank_id='blank')
    >>> print(string_out)
    ['a', 'b', 'c']
    """
    #import pudb; pudb.set_trace()
    if isinstance(string_pred, list):
        map_frame_to_ASR = []
        # Filter the repetitions
        string_out = []
        if string_pred[0] == space_id:
            cpt = 0
        else:
            cpt = 1
        previous_space = True
        for i, v in enumerate(string_pred):
            if i == 0 or v != string_pred[i - 1]:  # filter repetition
                string_out.append(v)
                # v is a new_space
                if v == space_id and not (len(string_out) == 2 and string_out[0] == blank_id) :  # filter on space_id
                    previous_space = True
                    cpt += 1
                    map_frame_to_ASR.append(0)  # blank ==0,
                    continue
                # v is at least a second space in a row
                if previous_space and v == space_id:
                    map_frame_to_ASR.append(0)
                    continue
                # v is a blank after a space or is the first element of the sequence
                if (
                    previous_space
                    and v == blank_id
                    or (i == 0 and v == blank_id)
                ):
                    map_frame_to_ASR.append(0)
                else:
                    previous_space = False
                    map_frame_to_ASR.append(cpt)
            else:
                if v == space_id:
                    map_frame_to_ASR.append(0)
                elif (
                    v == blank_id and previous_space == True
                ):  # deal with case of space followed by blank.
                    map_frame_to_ASR.append(0)
                else:
                    map_frame_to_ASR.append(cpt)

        # Remove duplicates, similar to uniq in unix. Take first element of uniq key. For data aug, i guess ?
        string_out = [i[0] for i in groupby(string_out)]

        # Filter the blank symbol. Original code
        string_out = list(filter(lambda elem: elem != blank_id, string_out))
        # remove following space_id. Should remove empty string
        string_out = [
            e
            for i, e in enumerate(string_out)
            if (i == 0) or not (e == space_id and string_out[i - 1] == space_id)
        ]
        if len(string_out) >0 and string_out[0] == space_id:
            string_out = string_out[1:]
        
        # print(string_out)
    else:
        raise ValueError("filter_ctc_out can only filter python lists")
    return string_out, map_frame_to_ASR


def ctc_greedy_decode(
    probabilities,
    seq_lens,
    space_id=1,
    blank_id=-1,
):
    """Greedy decode a batch of probabilities and apply CTC rules.
    Arguments
    ---------
    probabilities : torch.tensor
        Output probabilities (or log-probabilities) from the network with shape
        [batch, probabilities, time]
    seq_lens : torch.tensor
        Relative true sequence lengths (to deal with padded inputs),
        the longest sequence has length 1.0, others a value between zero and one
        shape [batch, lengths].
    blank_id : int, string
        The blank symbol/index. Default: -1. If a negative number is given,
        it is assumed to mean counting down from the maximum possible index,
        so that -1 refers to the maximum possible index.
    Returns
    -------
    list
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.
    Example
    -------
    >>> import torch
    >>> probs = torch.tensor([[[0.3, 0.7], [0.0, 0.0]],
    ...                       [[0.2, 0.8], [0.9, 0.1]]])
    >>> lens = torch.tensor([0.51, 1.0])
    >>> blank_id = 0
    >>> ctc_greedy_decode(probs, lens, blank_id)
    [[1], [1]]
    """
    if isinstance(blank_id, int) and blank_id < 0:
        blank_id = probabilities.shape[-1] + blank_id
    batch_max_len = probabilities.shape[1]
    batch_outputs = []
    map_batch = []
    for seq, seq_len in zip(probabilities, seq_lens):
        actual_size = int(torch.round(seq_len * batch_max_len))
        scores, predictions = torch.max(seq.narrow(0, 0, actual_size), dim=1)
        out, map = filter_ctc_output(
            predictions.tolist(), blank_id=blank_id, space_id=space_id
        )
        batch_outputs.append(out)
        map_batch.append(map)
    return batch_outputs, map_batch
