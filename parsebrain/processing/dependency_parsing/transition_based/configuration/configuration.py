from typing import List
import torch


class Configuration:
    def __init__(
        self,
        features: torch.Tensor,
        features_string: List["Word"],
        root_embedding: torch.Tensor,
    ):
        self.buffer = features
        self.buffer_string = features_string
        self.stack = []
        self.stack_string = []
        self.stack2 = []
        self.stack2_string = []
        self.arc = []
        self.has_root = False
        # root until linked is as if there is already one element in the stack.
        if root_embedding is None:
            try:
                self.root = torch.zeros(features[0].shape)
            # not a tensor, used for oracle (no need for features)
            except AttributeError:
                self.root = 0
        else:
            self.root = root_embedding
        # this value will be the value in one arc of the tree.
        self.root_token = Word("ROOT", 0)

    def add_features(self, features: List["Tensor"], features_string: List["Word"]):
        self.buffer = features
        self.buffer_string = features_string

    def shift_buffer(self):
        self.buffer = self.buffer[1:]
        self.buffer_string = self.buffer_string[1:]

    def __str__(self):
        return (
            f"buffer : {[str(x) for x in self.buffer_string]}, "
            f"stack : {[str(x) for x in self.stack_string]},"
            f" stack2 : {[str(x) for x in self.stack2_string]},"
            f" arcs : {[str(x) for x in self.arc]}"
        )


class GoldConfiguration:
    """
    This class contains the information about the gold data.
    It only uses the position of the word in the sentence to identify them.
    This remove the ambiguity if there is multiple occurrence of the same word.
    """

    def __init__(
        self, gov: List[int] = None, label: List[str] = None, sent_id: str = None
    ) -> object:
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
        self.sent_id = sent_id
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
