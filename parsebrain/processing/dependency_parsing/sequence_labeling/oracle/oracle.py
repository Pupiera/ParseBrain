from typing import List

from parsebrain.processing.dependency_parsing.sequence_labeling.oracle.reader import (
    Reader,
)


class Oracle:
    def __init__(self, reader_alig: Reader):
        self.reader_alig = reader_alig

    def set_alphabet(self, alphabet, reverse):
        self.alphabet = alphabet
        self.reverse = reverse
        self.reader_alig.set_alphabet(alphabet, reverse)

    def find_best_tree_from_alignment(
        self,
        alignment,
        dep2label_gov: List,
        dep2label_dep: List = None,
        gold_pos: List = None,
    ):
        raise NotImplementedError()
