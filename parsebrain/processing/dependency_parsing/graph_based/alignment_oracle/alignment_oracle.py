from typing import List

from parsebrain.processing.dependency_parsing.sequence_labeling.alignment_oracle.reader import (
    Reader,
)
from parsebrain.processing.dependency_parsing.sequence_labeling.alignment_oracle import (
    Oracle,
)

import torch

# todo : move the reader to a better path.


class AlignmentOracle(Oracle):
    def read_alignment(
        self, alignment: str, original_gov: List, original_dep: List, original_pos: List
    ):
        """

        @param alignment:
        @param original_gov:
        @param original_dep:
        @param original_pos:
        @return:
        >>> from parsebrain.processing.dependency_parsing.sequence_labeling.alignment_oracle.reader import ReaderSpeechbrain
        >>> alphabet = [{"DELETION": 10,"INSERTION":11 }, {"DELETION": 10,"INSERTION":11 }]
        >>> reverse = [{10: "DELETION", 11:"INSERTION"}, {10: "DELETION", 11:"INSERTION"}]
        >>> alignment = [[('I', None, 0), ('=', 0, 1), ('=', 1, 2), ('D', 2, None), ('D', 3, None)],
        ... [('S', 0, 0), ('D', 1, None), ('D', 2, None), ('D', 3, None)]]
        >>> original_gov = [[2, 0, 2, 3, 0], [2, 0, 2, 3, 0]]
        >>> original_dep = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
        >>> original_pos = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
        >>> oracle = AlignmentOracle(ReaderSpeechbrain())
        >>> oracle.set_alphabet(alphabet=alphabet, reverse=reverse )
        >>> oracle.read_alignment(alignment, original_gov, original_dep, original_pos)
        ([['I', 'C', 'C', 'D', 'D'], ['S', 'D', 'D', 'D']], [[-1, 3, 0, 3, 4], [2, 0, 2, 3]], [[11, 1, 2, 3, 4], [1, 2, 3, 4]], [[11, 1, 2, 3, 4], [1, 2, 3, 4]])
        """
        types, govs, deps, poss = self.reader_alig.read(
            alignment, original_gov, original_dep, original_pos
        )
        corrected_head = []
        # compute how many token have been inserted at which point and add it to the gov to keep pointing on the correct index 
        for y, typ in enumerate(types):
            for i, t in enumerate(typ):
                if 'I' in t:
                    govs[y][i] = -1
                    deps[y][i] = self.alphabet[0]["INSERTION"]
                    poss[y][i] = self.alphabet[1]["INSERTION"]
                    for index_gov, g in enumerate(govs[y]):
                        if 'INSERTION' == g:
                            continue
                        if int(g) > i:
                            govs[y][index_gov]+=1
        return types, govs, deps, poss


        '''
        for y, typ in enumerate(types):
            c = 0
            corr = []
            for i, t in enumerate(typ):
                if "I" in t:
                    # if insertion, the head is the root, will be colapsed to the actual root later
                    govs[y][i] = -1
                    deps[y][i] = self.alphabet[0]["INSERTION"]
                    poss[y][i] = self.alphabet[1]["INSERTION"]
                    c += 1
                corr.append(c)
            corrected_head.append(corr)
        for i, g in enumerate(govs):
            for y in range(len(g)):
                if govs[i][y] == 0 or govs[i][y] == -1:
                    continue
                h = govs[i][y] - 1
                c_add = corrected_head[i][h]
                govs[i][y] += c_add
        '''
        return types, govs, deps, poss

    def find_best_tree_from_alignment(
        self,
        alignment,
        original_gov: List,
        original_dep: List = None,
        original_pos: List = None,
        pytorch_mode=True
    ):
        """
        The two file sequence to be aligned: 
        ie the first sentence in the alignment 
        need to be the same as the first in gov, dep and pos

        @param alignment:
        @param original_gov:
        @param original_dep:
        @param original_pos:
        @return:
        >>> from parsebrain.processing.dependency_parsing.sequence_labeling.alignment_oracle.reader import ReaderSpeechbrain
        >>> alphabet = [{"DELETION": 10,"INSERTION":11 }, {"DELETION": 10,"INSERTION":11 }]
        >>> reverse = [{10: "DELETION", 11:"INSERTION"}, {10: "DELETION", 11:"INSERTION"}]
        >>> alignment = [[('I', None, 0), ('=', 0, 1), ('=', 1, 2), ('D', 2, None), ('D', 3, None)],
        ... [('S', 0, 0), ('D', 1, None), ('D', 2, None), ('D', 3, None)]]
        >>> original_gov = [[2, 0, 2, 3, 0], [2, 0, 2, 3, 0]]
        >>> original_dep = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
        >>> original_pos = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
        >>> oracle = AlignmentOracle(ReaderSpeechbrain())
        >>> oracle.set_alphabet(alphabet=alphabet, reverse=reverse )
        >>> oracle.find_best_tree_from_alignment(alignment, original_gov, original_dep, original_pos)
        ([tensor([3, 3, 0]), tensor([0])], [tensor([11,  1,  2]), tensor([10])], [tensor([11,  1,  2]), tensor([1])])
        >>> alignment =[[('D', 0, None), ('D', 1, None), ('=', 2, 0), ('S', 3, 1)]]
        >>> original_gov = [[2, 0, 4, 2]]
        >>> original_dep = [[1, 2, 3, 4]]
        >>> original_pos = [[1, 2, 3, 4]]
        >>> oracle.find_best_tree_from_alignment(alignment, original_gov, original_dep, original_pos)
        ([tensor([2, 0])], [tensor([ 3, 10])], [tensor([3, 4])])
        >>> alignment = [[('S', 0, 0), ('S', 1, 1), ('D', 2, None), ('D', 3, None), ('=', 4, 2)]]
        >>> original_gov = [[3, 3, 0, 3, 4]]
        >>> original_dep = [[1, 2, 3, 4, 5]]
        >>> original_pos = [[1, 2, 3, 4, 5]]
        >>> oracle.find_best_tree_from_alignment(alignment, original_gov, original_dep, original_pos)
        ([tensor([0, 1, 1])], [tensor([10, 10, 10])], [tensor([1, 2, 5])])
        >>> alignment = [[('S', 0, 0), ('I', None, 1), ('=', 1, 2), ('D', 2, None), ('D', 3, None), ('D', 4, None)]]
        >>> original_gov= [[4, 4, 4, 0, 4]]
        >>> oracle.find_best_tree_from_alignment(alignment, original_gov, original_dep, original_pos)
        ([tensor([0, 1, 1])], [tensor([10, 11, 10])], [tensor([ 1, 11,  2])])
        """
        types, govs, deps, poss = self.read_alignment(
            alignment, original_gov, original_dep, original_pos
        )
        new_gold_gov = []
        new_gold_dep = []
        new_gold_POS = []
        # for each sentence
        for typ, gov, dep, pos in zip(types, govs, deps, poss):
            has_root = False
            new_gov = []
            new_dep = []
            new_pos = []
            dist_from_root = []
            # for each tokens of the sentence
            for i, (t, g, d, p) in enumerate(zip(typ, gov, dep, pos)):
                # case deletion
                if t == "D":
                    dist_from_root.append(999999)
                    continue
                # case Insertion
                if t == "I":
                    new_gov.append(-1)
                    new_dep.append(d)
                    new_pos.append(p)
                    dist_from_root.append(999999)
                    continue
                if g == 0:
                    has_root=True
                    dist = 0
                else:
                    # if parent is alive, not root
                    dist = 99999
                    if typ[g - 1] == "D":
                        g, d, dist = self.find_closest_parent(typ, gov, i)
                new_gov.append(g)
                new_dep.append(d)
                new_pos.append(p)
                dist_from_root.append(dist)
            if self.multiple_root(new_gov):
                new_gov = self.colapse_root(new_gov, dist_from_root)
            try:
                root_position = new_gov.index(0)
            except ValueError as e:
                # if this is raised then there is a cycle between two.
                # The first element become the root by default.
                new_gov[0] = 0
                root_position = 0
            new_gov = self.colapse_unconected_root(new_gov, root_position)
            new_gov = self.adapt_head_deletion(new_gov, typ)
            if pytorch_mode:
                new_gold_gov.append(torch.tensor(new_gov))
                new_gold_dep.append(torch.tensor(new_dep))
                new_gold_POS.append(torch.tensor(new_pos))
            else:
                new_gold_gov.append(new_gov)
                new_gold_dep.append(new_dep)
                new_gold_POS.append(new_pos)
                
        return new_gold_gov, new_gold_dep, new_gold_POS

    def find_closest_parent(self, typ, gov, index_child):
        g = gov[index_child]
        p_t = typ[g - 1]
        p_gov = gov[g - 1]
        dist = 1
        while p_t == "D":
            dist += 1
            # infinite loop, information lost, attach to root
            if dist > 10000:
                g = -1
                dist = 999999
                break
            # if deleted head was root, then this is a root candidate
            if p_gov == 0:
                g = 0
                break
            else:
                g = p_gov

            p_t = typ[g - 1]
            p_gov = gov[g - 1]
        d = self.alphabet[0]["DELETION"]
        return g, d, dist

    def multiple_root(self, gov):
        return gov.count(0) > 1

    def colapse_root(self, new_gov, dist_from_root):
        new_root_index = dist_from_root.index(min(dist_from_root))
        for i, gov in enumerate(new_gov):
            if gov == 0:
                if i == new_root_index:
                    continue
                new_gov[i] = new_root_index + 1
        return new_gov

    def colapse_unconected_root(self, new_gov, root_index):
        for i, g in enumerate(new_gov):
            if g == -1:
                new_gov[i] = root_index + 1
        return new_gov

    def adapt_head_deletion(self, new_gov, type):
        """
        This function change the value of the head to reflect the change in the numerical value
        caused by deletion
        ie : with alig :[D, D, S, C] and gov : [0, 1, 4, 2], without this function you get
        gov [4, 0] and 4 is out of scope. This function adapt the value to be
        gov [2, 0] which is in scope.
        This need to be done based on the value of the head.
        @param new_gov:
        @param type:
        @return:
        """
        correcting_val = []
        c = 0
        for t in type:
            if t == "D":
                c += 1
            correcting_val.append(c)
        i = 0
        for t in type:
            if t == "D":
                continue
            if t == "I":
                i += 1
                continue
            index_head_to_edit = new_gov[i]
            if index_head_to_edit == 0:
                i += 1
                continue
            new_gov[i] -= correcting_val[index_head_to_edit - 1]
            i += 1
        return new_gov


if __name__ == "__main__":
    import doctest

    doctest.testmod()
