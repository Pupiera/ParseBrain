from typing import List

from parsebrain.processing.dependency_parsing.sequence_labeling.encoding.decoder import (
    Decoder,
)
import parsebrain.processing.dependency_parsing.sequence_labeling.encoding.postprocessing as postprocessing

import torch
from parsebrain.processing.dependency_parsing.sequence_labeling.encoding.encoder_relative_position import (
    RelPosEncoding,
)


class DecoderRelPos(Decoder):
    """
    Class taking probabilities of each label and the alphabet and converting it to conll text.
    """

    def __init__(self, eval_st):

        self.encoding = RelPosEncoding()
        self.l = postprocessing.LabelPostProcessor_dep2Label()
        self.p = postprocessing.CoNLLPostProcessor(eval_st)
        self.decoded_sentences = {}
        self.nb_of_sentence = 1

    def set_alphabet(self, alphabet):
        self.gov_alphabet = alphabet[0]
        self.dep_alphabet = alphabet[1]
        self.pos_alphabet = alphabet[2]
        self.reverse_gov_alphabet = {
            value: key for (key, value) in self.gov_alphabet.items()
        }
        self.reverse_dep_alphabet = {
            value: key for (key, value) in self.dep_alphabet.items()
        }
        self.reverse_pos_alphabet = {
            value: key for (key, value) in self.pos_alphabet.items()
        }

    def decode(self, list_prob: List, predicted_words: List[List], sent_ids: List, prob=True):
        """
            Decode the probability into a dependency tree
        Parameters
        ----------
        list_prob : List containing the three following tensor
            p_govLabel : tensor of form [batch, seq_len, len(self.gov_alphabet)]
            p_depLabel : tensor of form [batch, seq_len, len(self.dep_alphabet)]
            p_posLabel : tensor of form [batch, seq_len, len(self.pos_alphabet)]
        predicted_words : list of list of predicted words by ASR.
        sent_ids : List containig the sent_id of each sentence processed in the current batch
        Returns: For each element of the batch (sentence) a list of string, where each element correspond to a line
        in a conllu format.
        -------
        """
        p_govLabel = list_prob[0]
        p_depLabel = list_prob[1]
        p_posLabel = list_prob[2]

        for p_deps, p_govs, words, p_poss, sent_id in zip(
            p_depLabel, p_govLabel, predicted_words, p_posLabel, sent_ids
        ):  # for each element of the batch
            to_decode = {0: ["-BOS-", "-BOS-", "-BOS-", "-BOS-", "-BOS-"]}
            # for each element of seqlen
            # print(f"{sent_id} \t {words} \t {p_govLabel.shape}")
            for i, (p_dep, p_gov, p_pos, word) in enumerate(
                zip(p_deps, p_govs, p_poss, words)
            ):
                if prob :
                    dep = self.reverse_dep_alphabet[torch.argmax(p_dep).item()]
                    gov = self.reverse_gov_alphabet[torch.argmax(p_gov).item()].split("@")
                    pos = self.reverse_pos_alphabet[torch.argmax(p_pos).item()]
                else:
                    #already decoded
                    dep = self.reverse_dep_alphabet[p_dep]
                    gov = self.reverse_gov_alphabet[p_gov].split("@")
                    pos = self.reverse_pos_alphabet[p_pos]
                # if word == "":
                #    word = "[EMPTY_ASR_WRD]"
                to_decode[i + 1] = [word, pos, gov[0], dep, gov[-1]]
            # to_decode[len(to_decode)]=["-EOS-",	"-EOS-","-EOS-","-EOS-","-EOS-"]
            # print(to_decode)
            decoded_words, headless_words = self.encoding.decode(to_decode)
            # POSTPROCESSING
            if not self.l.has_root(decoded_words):
                self.l.find_candidates_for_root(decoded_words, headless_words)
            if not self.l.has_root(decoded_words):
                self.l.assign_root_index_one(decoded_words, headless_words)
            if self.l.has_multiple_roots(decoded_words):
                self.l.choose_root_from_multiple_candidates(decoded_words)
            self.l.assign_headless_words_to_root(decoded_words, headless_words)
            self.l.find_cycles(decoded_words)
            self.p.convert_labels2conllu(decoded_words)
            self.decoded_sentences.update({sent_id: decoded_words})
            self.nb_of_sentence += 1

    def writeToCoNLLU(self, pathFile, order):
        """
        Write the tree stored through the decode function to the pathfile with a specific order
        Parameters
        ----------
        pathFile
        order

        Returns
        -------

        """
        # print(f"self.decoded_sentences {self.decoded_sentences}")
        self.p.write_to_file(self.decoded_sentences, pathFile, order)

    def evaluateCoNLLU(self, goldPath, predictedPathFile, alig_path=None, gold_segmentation = False):
        if not gold_segmentation:
            assert alig_path is not None, "if gold_segmentation is False, an alignment file path must be provided"
        return self.p.evaluate_dependencies(goldPath, predictedPathFile, alig_path, gold_segmentation)
