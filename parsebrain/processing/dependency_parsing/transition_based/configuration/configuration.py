from typing import List

from torch import Tensor


class Configuration:
    def __init__(self, features: List["Tensor"], features_string: List["Word"]):
        self.buffer = features
        self.buffer_string = features_string
        self.stack = []
        self.stack_string = []
        self.arc = []

    def add_features(self, features: List["Tensor"], features_string: List["Word"]):
        self.buffer = features
        self.buffer_string = features_string

    def shift_buffer(self):
        self.buffer = self.buffer[1:]
        self.buffer_string = self.buffer_string[1:]


class GoldConfiguration:
    """
    This class contains the information about the gold data.
    It only uses the position of the word in the sentence to identify them.
    This remove the ambiguity if there is multiple occurrence of the same word.
    """

    def __init__(self, gov: List[int] = None, label: List[str] = None):
        """
        >>> gov = [2,0,2,3]
        >>> lab = ['X','root','Y', 'Z']
        >>> g_c = GoldConfiguration(gov, lab)
        >>> g_c.heads
        {1: 2, 2: 0, 3: 2, 4: 3}
        >>> g_c.deps
        {1: [], 2: [1, 3], 3: [4], 4: []}
        >>> g_c.label
        {1: 'X', 2: 'root', 3: 'Y', 4: 'Z'}
        """
        self.heads = {}  # the head of a given word
        self.deps = {}  # the list of dependent of a given word (can be empty)
        if gov is not None:
            for i, g in enumerate(gov):
                g = int(g)
                self.heads[i + 1] = g
            depss = []
            for i, _ in enumerate(gov):
                tmp = []
                for y, gg in enumerate(gov):
                    gg = int(gg)
                    if i + 1 == gg:
                        tmp.append(y + 1)
                depss.append(tmp)
            for i, d in enumerate(depss):
                self.deps[i + 1] = d
            if label is not None:
                self.label = {}  # the label between words i and the head.
                for i, l in enumerate(label):
                    self.label[i + 1] = l


class GoldConfigurationASR:
    def __init__(self, alignment: List, gov: List[int], label: List[str]):
        """
        >>> from speechbrain.utils.edit_distance import wer_details_for_batch
        >>> ids = [['utt1'], ['utt2']]
        >>> refs = [[['aa','bb','dd','c']]]
        >>> hyps = [[['aa','c']]]
        >>> gov = [0, 1, 2, 3]
        >>> label = ["root", "XX","YY", "ZZ"]
        >>> wer_details = []
        >>> for ids_batch, refs_batch, hyps_batch in zip(ids, refs, hyps):
        ...     details = wer_details_for_batch(ids_batch, refs_batch, hyps_batch, compute_alignments=True)
        ...     wer_details.extend(details)
        >>> x = GoldConfigurationASR(wer_details[0]['alignment'], gov, label)
        >>> x.heads
        {1: 0, 2: 1}
        >>> x.label
        {1: 'root', 2: 'DELETION'}
        >>> refs = [[['aa', 'bb', 'cc','dd']]]
        >>> hyps = [[['bb','cc','dd']]]
        >>> gov = [0, 1, 2, 1]
        >>> label = ["root", "XX","YY", "ZZ"]
        >>> wer_details = []
        >>> for ids_batch, refs_batch, hyps_batch in zip(ids, refs, hyps):
        ...     details = wer_details_for_batch(ids_batch, refs_batch, hyps_batch, compute_alignments=True)
        ...     wer_details.extend(details)
        >>> x = GoldConfigurationASR(wer_details[0]['alignment'], gov, label)
        >>> x.heads
        {1: 0, 2: 1, 3: 1}
        >>> x.label
        {1: 'root', 2: 'YY', 3: 'DELETION'}
        """
        self.heads = {}  # the head of a given word
        self.deps = {}  # the list of dependent of a given word (can be empty)
        self.label = {}
        self.dist_from_root = {}
        root_indice = -1
        root_status = None
        for i in range(len(gov)):
            if gov[i] == 0:
                root_indice = i
                print(alignment[root_indice])
                root_status = alignment[root_indice][0] != "D"

        for i, (alig, g, l) in enumerate(zip(alignment, gov, label)):
            if alig[0] == "=" or alig == "S":
                position_system = alig[2] + 1
                if g == 0:
                    self.heads[position_system] = 0
                    self.label[position_system] = l
                    self.dist_from_root[position_system] = 0
                    continue
                original_head_status = alignment[g - 1]

                if original_head_status[0] == "=" or original_head_status[0] == "S":
                    # keep original head
                    self.heads[position_system] = alignment[g - 1][2]
                    self.label[position_system] = l
                    self.dist_from_root[
                        position_system
                    ] = 99999  # not a candidate for root.
                elif original_head_status[0] == "D":
                    # recursively find new head
                    current_head_indice = g - 1
                    dist_from_root = 0
                    while original_head_status[0] == "D" and current_head_indice != -1:
                        dist_from_root += 1
                        current_head_indice = gov[current_head_indice] - 1
                        original_head_status = alignment[current_head_indice]
                    if current_head_indice == -1:
                        self.heads[position_system] = 0
                        self.label[position_system] = "DELETION"
                        self.dist_from_root[position_system] = dist_from_root
                        continue
                    self.heads[position_system] = current_head_indice + 1
                    self.label[position_system] = "DELETION"
                # insertion is not possible here. (original head can't be attached to non existant head in gold)
            elif alig[0] == "D":
                # do nothing ? if deleted element was root need to find a new one.
                pass
            elif alig[0] == "I":
                # attach to root
                position_system = alig[2] + 1
                if root_status:
                    self.heads[position_system] = root_indice
                    self.label[position_system] = "INSERTION"
                else:
                    # in case if this is the only words in the sentence...
                    self.heads[position_system] = 0
                    self.label[position_system] = "INSERTION"

                self.dist_from_root = 99999
        self.collapse_root()

    def has_multiple_roots(self):
        return len([item for key, item in self.heads.items() if item == 0]) > 1

    def collapse_root(self):
        root_indice = min(self.dist_from_root, key=self.dist_from_root.get)
        self.label[root_indice] = "root"
        for key, item in self.heads.items():
            if item == 0 and key != root_indice:
                self.heads[key] = root_indice


class Word:
    """
    Contains the word (string)
    """

    def __init__(self, word: str, position: int):
        self.word = word
        self.position = position

    def __str__(self):
        return f" Word (word :{self.word}, position : {self.position})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
