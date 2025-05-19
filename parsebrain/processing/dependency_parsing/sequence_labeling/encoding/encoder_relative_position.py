from parsebrain.processing.dependency_parsing.sequence_labeling.encoding.postprocessing import LabelPostProcessor_dep2Label
from parsebrain.processing.dependency_parsing.sequence_labeling.encoding.encoder import (
    Encoder,
)


class RelPosEncoding(Encoder):
    """
    From Dep2Label paper (Viable Dependency Parsing as Sequence Labeling):
    https://github.com/mstrise/dep2label
    https://aclanthology.org/N19-1077/
    """

    def encodeFromList(self, wrds, poss, govs, deps):

        """
        *
        Parameters
        ----------
        wrds : List of words
        poss : List of POS
        govs : List of Head /Governor
        deps : List of type of dependence e.g: aux, subj ...
        All list must have the same size and should because each element represent one word of the sentence.
        Returns a List of dep2label to be used for the tagging task.
        2 task Separated by "{}". here the task are the 1 : relative_position_head + postag_gov and 2: the type of dep
        -------
        >>> wrds = ['mais', 'on', "s'", 'amuse', 'beaucoup', "c'", 'est', "l'", 'école', "d'", 'ingénieur', 'euh', 'le', 'retour', 'des', 'de', 'la', 'vie', 'sociale', 'de', 'les', 'amis', 'et', 'tout', 'ça', 'donc', 'ça', "c'", 'était', 'chouette']
        >>> poss = ['CCONJ', 'PRON', 'PRON', 'VERB', 'ADV', 'PRON', 'VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'INTJ', 'DET', 'NOUN', 'ADP+DET', 'ADP', 'DET', 'NOUN', 'ADJ', 'ADP', 'DET', 'NOUN', 'CCONJ', 'ADJ', 'PRON', 'CCONJ', 'PRON', 'PRON', 'VERB', 'ADJ']
        >>> govs = ['4', '4', '4', '0', '4', '7', '14', '9', '7', '11', '9', '14', '14', '4', '15', '18', '18', '14', '18', '22', '22', '15', '25', '25', '22', '29', '29', '29', '22', '29']
        >>> deps = ['cc', 'nsubj', 'expl:comp', 'root', 'advmod', 'nsubj', 'reparandum', 'det', 'xcomp', 'case', 'nmod', 'discourse', 'det', 'conj', 'case+reparandum', 'case', 'det', 'nmod', 'amod', 'case', 'det', 'appos', 'cc', 'amod', 'conj', 'cc', 'dislocated', 'nsubj', 'conj', 'xcomp']
        >>> encoding = RelPosEncoding()
        >>> encoding.encodeFromList(wrds, poss, govs, deps)
        """
        list_label = []
        for i, (wrd, pos, gov, dep) in enumerate(zip(wrds, poss, govs, deps)):
            if gov == "0":
                label = "-1@ROOT{}root"
                full_label = str(wrd + "\t" + pos + "\t" + label)
                list_label.append(full_label)
                continue
            gov = int(gov) - 1
            if gov == i:
                raise ValueError(f"Governor and line index is the same, check your conllu file for sentence : {' '.join(wrds)}")
            if i < gov:
                relative_position_head = 1
                postag_gov = poss[gov]
                for w in range(i + 1, gov):
                    pos_word_w = poss[w]
                    if (
                        pos_word_w == postag_gov
                    ):  # Not the next wrd with this POSTAG but the one after
                        relative_position_head += 1
                label = str(
                    "+" + repr(relative_position_head) + "@" + postag_gov + "{}" + dep
                )
                full_label = str(wrd + "\t" + pos + "\t" + label)
                list_label.append(full_label)
            elif i > gov:
                relative_position_head = 1
                postag_gov = poss[gov]
                for w in range(gov + 1, i):
                    pos_word_w = poss[w]
                    if pos_word_w == postag_gov:
                        relative_position_head += 1
                label = str(
                    "-" + repr(relative_position_head) + "@" + postag_gov + "{}" + dep
                )
                full_label = str(wrd + "\t" + pos + "\t" + label)
                list_label.append(full_label)
        return list_label

    def encode(self, sentence):
        """[summary]

        Args:
            sentence [dict]: [sentence represented as {index_word: {"id":int,"word":str,"lemma":str,"pos":str,"head":int }}]
        Returns:
            words_with_labels [dict]: [word with its PoS and label]
        """
        task = "2-task-combined"
        words_with_labels = {}
        l = LabelPostProcessor_dep2Label()
        # combined 2-task label: x@x{}x
        words_with_labels = l.tag_BOS(task, words_with_labels)

        for index_word in sentence:
            if not index_word == 0:
                word = sentence[index_word]
                head = word["head"]
                if index_word < head:
                    relative_position_head = 1
                    head_word = sentence[head]
                    postag_head = head_word["pos"]
                    for w in range(index_word + 1, head):
                        word_i = sentence[w]
                        postag_word_i = word_i["pos"]
                        if postag_word_i == postag_head:
                            relative_position_head += 1

                    label = str(
                        "+"
                        + repr(relative_position_head)
                        + "@"
                        + postag_head
                        + "{}"
                        + word["deprel"]
                    )
                    full_label = str(word["word"] + "\t" + word["pos"] + "\t" + label)
                    words_with_labels.update({index_word: full_label})
                elif index_word > head:
                    relative_position_head = 1
                    head_word = sentence[head]
                    postag_head = head_word["pos"]
                    for w in range(head + 1, index_word):
                        word_i = sentence[w]
                        postag_word_i = word_i["pos"]
                        if postag_word_i == postag_head:
                            relative_position_head += 1

                    label = str(
                        "-"
                        + repr(relative_position_head)
                        + "@"
                        + postag_head
                        + "{}"
                        + word["deprel"]
                    )
                    full_label = str(word["word"] + "\t" + word["pos"] + "\t" + label)
                    words_with_labels.update({index_word: full_label})

        words_with_labels = l.tag_EOS(task, words_with_labels)
        return words_with_labels

    def decode(self, sentence):
        """[summary]

        Args:
            sentence [dict]: [int: ["word", "PoS", "rel.position", "deprel","head's PoS"]]

        Returns:
            decoded_words [dict]: [words with assigned head]
            unassigned_word [dict]: [words for which head assignment failed]
        """
        decoded_words = {}  # 1 : ['The', 'DT', '+1', 'det', 'NN']
        unassigned_word = {}

        def assignHeadLeft(word_index, info_word, postag_head, decoded, abs_posit):
            count_posit = 0
            # find postag_head with the relative position -1,-2....
            for index in range(word_index - 1, -1, -1):
                word_candidate = decoded[index]
                postag_candidate = word_candidate[1]
                if postag_candidate == postag_head:
                    count_posit += 1
                    if abs_posit == count_posit:
                        head_word = {
                            1: word_index,
                            2: info_word[0],
                            3: "_",
                            4: info_word[1],
                            5: index,
                            6: info_word[3],
                        }
                        return head_word

        def assignHeadRight(word_index, info_word, postag_head, decoded, abs_posit):
            count_posit = 0
            # find postag_head with the relative position +1,+2....
            for index in range(word_index + 1, len(decoded)):
                word_candidate = decoded[index]
                postag_candidate = word_candidate[1]
                if postag_candidate == postag_head:
                    count_posit += 1
                    if abs_posit == count_posit:
                        head_word = {
                            1: word_index,
                            2: info_word[0],
                            3: "_",
                            4: info_word[1],
                            5: index,
                            6: info_word[3],
                        }
                        return head_word

        for word_index in sentence:
            if not word_index == 0:
                word_line = sentence.get(word_index)
                info_word = word_line
                found_head = False
                if not word_line[2] == "-EOS-" and not word_line[2] == "-BOS-":
                    rel_pos_head = int(word_line[2])
                    pos_head = word_line[4]
                    abs_posit = abs(rel_pos_head)
                    positminus1 = abs_posit - 1
                    positplus1 = abs_posit + 1
                    if rel_pos_head < 0:
                        head_word = assignHeadLeft(
                            word_index, info_word, pos_head, sentence, abs_posit
                        )
                        if head_word:
                            decoded_words.update({word_index: head_word})
                            found_head = True

                        elif not positminus1 == 0:
                            head_word = assignHeadLeft(
                                word_index, info_word, pos_head, sentence, positminus1
                            )
                            if head_word:
                                decoded_words.update({word_index: head_word})
                                found_head = True
                        else:
                            head_word = assignHeadLeft(
                                word_index, info_word, pos_head, sentence, positplus1
                            )
                            if head_word:
                                found_head = True
                                decoded_words.update({word_index: head_word})

                        # find postag_head with the relative position +1,+2....
                    elif rel_pos_head > 0:
                        found_head = False
                        head_word = assignHeadRight(
                            word_index, info_word, pos_head, sentence, abs_posit
                        )
                        if head_word:
                            decoded_words.update({word_index: head_word})
                            found_head = True

                        elif not positminus1 == 0:
                            head_word = assignHeadRight(
                                word_index, info_word, pos_head, sentence, positminus1
                            )
                            if head_word:
                                decoded_words.update({word_index: head_word})
                                found_head = True
                        else:
                            head_word = assignHeadRight(
                                word_index, info_word, pos_head, sentence, positplus1
                            )
                            if head_word:
                                decoded_words.update({word_index: head_word})
                                found_head = True
                    if not found_head:
                        head_word = {
                            1: word_index,
                            2: info_word[0],
                            3: "_",
                            4: info_word[1],
                            5: -1,
                            6: info_word[3],
                        }
                        unassigned_word.update({word_index: head_word})

                        decoded_words.update({word_index: head_word})
                else:
                    head_word = {
                        1: word_index,
                        2: info_word[0],
                        3: "_",
                        4: info_word[1],
                        5: -1,
                        6: "root",
                    }
                    unassigned_word.update({word_index: head_word})
                    decoded_words.update({word_index: head_word})

        return decoded_words, unassigned_word




if __name__ == "__main__":
    #import doctest

    #doctest.testmod()
    wrds = ['mais', 'on', "s'", 'amuse', 'beaucoup', "c'", 'est', "l'", 'école', "d'", 'ingénieur', 'euh', 'le', 'retour', 'des', 'de', 'la', 'vie', 'sociale', 'de', 'les', 'amis', 'et', 'tout', 'ça', 'donc', 'ça', "c'", 'était', 'chouette']
    poss = ['CCONJ', 'PRON', 'PRON', 'VERB', 'ADV', 'PRON', 'VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'INTJ', 'DET', 'NOUN', 'ADP+DET', 'ADP', 'DET', 'NOUN', 'ADJ', 'ADP', 'DET', 'NOUN', 'CCONJ', 'ADJ', 'PRON', 'CCONJ', 'PRON', 'PRON', 'VERB', 'ADJ']
    govs = ['4', '4', '4', '0', '4', '7', '14', '9', '7', '11', '9', '14', '14', '4', '15', '18', '18', '14', '18', '22', '22', '15', '25', '25', '22', '29', '29', '29', '22', '29']
    deps = ['cc', 'nsubj', 'expl:comp', 'root', 'advmod', 'nsubj', 'reparandum', 'det', 'xcomp', 'case', 'nmod', 'discourse', 'det', 'conj', 'case+reparandum', 'case', 'det', 'nmod', 'amod', 'case', 'det', 'appos', 'cc', 'amod', 'conj', 'cc', 'dislocated', 'nsubj', 'conj', 'xcomp']
    encoding = RelPosEncoding()
    from pudb import set_trace; set_trace()
    encoding.encodeFromList(wrds, poss, govs, deps)
