from typing import List


class Encoder:
    def encode(self, sentence: dict):
        """
        sentence [dict]: [sentence represented as {index_word: {"id":int,"word":str,"lemma":str,"pos":str,"head":int }}]
        """
        raise NotImplementedError()

    def decode(self, sentence: List):
        raise NotImplementedError()
