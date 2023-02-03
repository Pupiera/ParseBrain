from typing import List

from parsebrain.processing.dependency_parsing.sequence_labeling.alignment_oracle.reader import (
    Reader,
)


class Oracle:
    def __init__(self, reader_alig: Reader):
        self.reader_alig = reader_alig

    def set_alphabet(self, alphabet, reverse):
        self.alphabet = alphabet
        self.reverse = reverse

    def find_best_tree_from_alignment(
        self,
        alignment,
        original_gov: List,
        original_dep: List = None,
        original_pos: List = None,
    ):
        raise NotImplementedError()

    def read_alignment(
        self, alignment: str, original_gov: List, original_dep: List, original_pos: List
    ):
        """
        Implement default behavior, ie : just read the file.
        "INSERTION" token will be in place of inserted tokens. (default supervision, need to tokenize it)
        @param alignment:
        @param original_gov:
        @param original_dep:
        @param original_pos:
        @return:
        """
        raise NotImplementedError()
