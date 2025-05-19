import torch


#ToDo : Add way to add constraint during decoding. 
# The constraint can be represented as a map containing the previous element and the banned elements in a list ? 
# Then when you generate the probabilty during the decoding, we set these prob at 0 if it's in the ['previous_element']

#Not sure that the start and end tag are needed when you are not generating it in a seq2seq way...

class ViterbiDecoder():
    """
    Viterbi Decoder.

    A I can't be preceded by a O, it need to be inside the word (either a B or I must be before) 
    >>> dic = {0 : '<start>', 1 : 'B', 2 : 'I', 3: 'O', 4 : '<end>'}
    >>> reversed_dic = {item : key for key, item in dic.items()}
    >>> viterbi = ViterbiDecoder(reversed_dic)
    >>> viterbi_ban = ViterbiDecoder(reversed_dic, {2: 3})
    >>> scores = torch.ones((2, 4, 5, 5))
    >>> length = torch.tensor([4, 3])
    >>> decoded = viterbi.decode(scores, length)
    >>> decoded_ban = viterbi_ban.decode(scores, length)
    """

    def __init__(self, tag_map = {}, banned_sequence = {}):
        """
        :param tag_map: tag map
        """
        self.banned_sequence = banned_sequence
        if len(tag_map) == 0:
            print("Viterbi tag map not initialized, set it by using .set_tag_map()")
            return
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def set_tag_map(self, tag_map):
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']


    def decode(self, scores, lengths):
        """
        :param scores: CRF scores
        :param lengths: sequence lengths
        :return: decoded sequences
        """
        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size, device = scores.device)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, max(lengths), self.tagset_size), dtype=torch.long, device = scores.device) * self.end_tag
        set_trace()

        #to ban transition, the easiest way is to set the banned transition scores at very low scores.
        if len(self.banned_sequence) >0:
            for i in range(self.tagset_size):
                if i in self.banned_sequence.keys():
                    # shape of score (batch, len_seq, prev_tag, current_tag)
                    scores[:, :, self.banned_sequence[i], i] = -1000

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.tagset_size),
                                                               dtype=torch.long) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t], backpointers[:batch_size_t, t, :] = torch.max(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)


        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long, device = scores.device)
        pointer = torch.ones((batch_size, 1),
                             dtype=torch.long, device = scores.device) * self.end_tag  # the pointers at the ends are all <end> tags

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long, device = scores.device) * self.start_tag)

        # Remove the <starts> at the beginning, and append with <ends> (to compare to targets, if any)
        decoded = torch.cat([decoded[:, 1:], torch.ones((batch_size, 1), dtype=torch.long, device = scores.device) * self.start_tag],
                            dim=1)

        return decoded

if __name__ == "__main__":
    import doctest
    from pudb import set_trace
    
    #doctest.testmod()
    dic = {0 : '<start>', 1 : 'B', 2 : 'I', 3: 'O', 4 : '<end>'}
    reversed_dic = {item : key for key, item in dic.items()}
    viterbi = ViterbiDecoder(reversed_dic)
    # added constraint, a 2 cannot be preceded by a 3
    viterbi_ban = ViterbiDecoder(reversed_dic, {2: [0, 3, 4]})
    #shape (2,4,5,5)
    scores = torch.tensor([[[[0.8842, 0.6705, 0.7865, 0.3143, 0.6859],
          [0.3889, 0.6416, 0.8733, 0.2831, 0.0618],
          [0.1497, 0.2180, 0.3276, 0.2372, 0.5161],
          [0.2636, 0.3752, 0.1952, 0.5136, 0.6829],
          [0.1066, 0.3951, 0.1629, 0.7904, 0.3340]],

         [[0.6429, 0.3017, 0.3177, 0.3453, 0.9375],
          [0.5780, 0.2236, 0.3234, 0.0692, 0.4139],
          [0.3607, 0.5354, 0.3918, 0.2165, 0.6987],
          [0.3816, 0.4068, 0.9704, 0.1058, 0.9590],
          [0.7699, 0.7560, 0.2428, 0.0715, 0.3746]],

         [[0.3139, 0.4555, 0.9564, 0.5373, 0.8168],
          [0.5462, 0.8605, 0.4142, 0.1867, 0.8482],
          [0.5923, 0.9498, 0.2611, 0.7489, 0.1628],
          [0.4146, 0.5166, 0.5106, 0.0476, 0.8524],
          [0.4480, 0.1900, 0.0069, 0.8102, 0.6484]],

         [[0.6174, 0.0881, 0.8861, 0.0718, 0.0392],
          [0.3713, 0.7228, 0.3432, 0.7857, 0.9894],
          [0.5530, 0.7182, 0.5191, 0.7696, 0.2384],
          [0.8237, 0.7264, 0.6397, 0.7907, 0.5739],
          [0.2137, 0.2996, 0.6468, 0.4328, 0.3962]]],


        [[[0.3884, 0.8282, 0.4442, 0.3307, 0.4457],
          [0.3538, 0.2891, 0.1741, 0.3474, 0.7148],
          [0.9761, 0.6556, 0.9897, 0.6562, 0.1717],
          [0.0452, 0.5307, 0.7648, 0.0080, 0.6761],
          [0.1761, 0.6168, 0.7971, 0.4067, 0.6102]],

         [[0.1926, 0.1667, 0.1086, 0.1730, 0.3902],
          [0.7456, 0.6228, 0.6473, 0.3862, 0.7708],
          [0.8916, 0.8094, 0.7682, 0.2532, 0.9950],
          [0.5128, 0.9330, 0.1249, 0.6762, 0.3299],
          [0.3131, 0.9582, 0.8749, 0.1623, 0.6920]],

         [[0.0913, 0.5704, 0.6036, 0.4190, 0.8383],
          [0.9998, 0.8383, 0.2463, 0.6109, 0.1863],
          [0.9942, 0.5123, 0.4500, 0.6351, 0.1967],
          [0.6846, 0.5183, 0.5987, 0.9147, 0.9731],
          [0.6829, 0.0467, 0.2632, 0.6872, 0.1613]],

         [[0.0480, 0.8749, 0.3634, 0.2227, 0.0480],
          [0.0833, 0.4620, 0.5594, 0.2886, 0.3749],
          [0.3741, 0.2131, 0.6381, 0.8125, 0.5150],
          [0.7253, 0.9907, 0.7185, 0.8509, 0.3763],
          [0.0112, 0.6033, 0.2343, 0.6124, 0.8933]]]])
    length = torch.tensor([4, 3])
    decoded = viterbi.decode(scores, length)
    print(f"decoded {decoded}")
    print("with BAN")
    decoded_ban = viterbi_ban.decode(scores, length)
    print(f"decoded_ban {decoded_ban}")
