from typing import List


class Reader:
    def __init__(self):
        self.reverse = None
        self.alphabet = None

    def read(
        self, alignment, original_gov: List, original_dep: List, original_pos: List
    ):
        raise NotImplementedError()

    def set_alphabet(self, alphabet: dict, reverse: dict):
        self.alphabet = alphabet
        self.reverse = reverse
