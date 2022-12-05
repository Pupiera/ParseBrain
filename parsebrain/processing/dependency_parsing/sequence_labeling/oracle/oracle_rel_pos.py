from typing import List

from parsebrain.processing.dependency_parsing.sequence_labeling.oracle.reader import (
    Reader,
)
from parsebrain.processing.dependency_parsing.sequence_labeling.oracle.oracle import (
    Oracle,
)
import torch


class OracleRelPos(Oracle):
    def __init__(self, reader_alig: Reader):
        super().__init__(reader_alig)

    def find_best_tree_from_alignment(
        self,
        alignment,
        dep2label_gov: List,
        dep2label_dep: List = None,
        gold_pos: List = None,
    ):

        new_gold_gov = []
        new_gold_dep = []
        new_gold_POS = []

        # Replace this by dict ?
        types, govs, deps, poss = self.reader_alig.read(
            alignment, dep2label_gov, dep2label_dep, gold_pos
        )

        for typ, gov, dep, pos in zip(types, govs, deps, poss):  # for each sentence
            # print(f"{typ} {[self.reverse[0].get(g.item()) for g in gov]} {[self.reverse[1].get(d.item()) for d in dep]} {[self.reverse[2].get(p.item()) for p in pos]}")
            has_root = False
            new_gov = []
            new_dep = []
            new_pos = []
            dist_from_root = []
            try:
                pos_trad = [self.reverse[2][p.item()] for p in pos]
            except AttributeError:  # Not tensor
                pos_trad = [self.reverse[2][p] for p in pos]
            for i, (t, g, d, p) in enumerate(
                zip(typ, gov, dep, pos)
            ):  # for each tokens of the sentence
                try:
                    g = g.item()
                    d = d.item()
                    p = p.item()
                except AttributeError:  # not tensor
                    pass
                if g == 0:
                    print(alignment)
                if (
                    t == "D"
                ):  # Deleted tokens do not appears in the new best dependency tree
                    dist_from_root.append(999999)
                    continue
                if t == "I":
                    new_gov.append(
                        self.alphabet[0]["-1@ROOT"]
                    )  # insertion will be attach to root later
                    new_dep.append(d)
                    new_pos.append(p)
                    dist_from_root.append(999999)
                    continue
                # to do : debug this. behavior should be : if parent is deleted, attach to parent parent recursively.
                # while p_t == 'D' or not p_gov == "-1@ROOT": # while parent is deleted and current gov is not root
                if g == self.alphabet[0]["-1@ROOT"]:
                    has_root = True
                    dist = 0
                else:
                    dist = 99999  # if parent is alive, not root
                    p_t, p_gov, index = self.find_parent_type(g, pos_trad, typ, gov, i)
                    # print(f"pt : {p_t}, p_gov : {self.reverse[0].get(p_gov.item())}")
                    if (
                        p_t == "D"
                    ):  # if parent missing, need to attach child to something else.
                        g, d, dist = self.find_closest_parent(typ, gov, pos_trad, i)
                new_gov.append(g)
                new_dep.append(d)
                new_pos.append(p)
                dist_from_root.append(dist)
            if self.multiple_root(new_gov):
                new_gov = self.collapse_root(new_gov, pos_trad, typ, dist_from_root)
            try:
                new_gold_gov.append(torch.tensor(new_gov))
                new_gold_dep.append(torch.tensor(new_dep))
                new_gold_POS.append(torch.tensor(new_pos))
            except RuntimeError as e:
                print(new_gov)
                raise RuntimeError() from e
        return new_gold_gov, new_gold_dep, new_gold_POS

    def find_closest_parent(self, typ, gov, pos_trad, index_child):
        """
        From the segmentation and gov, find the closest node to this element
        Parameters
        ----------
        typ
        gov
        pos_trad
        index_child

        Returns
        -------

        """
        try:
            g = gov[index_child].item()
        except AttributeError:
            g = gov[index_child]
        p_t, p_gov, index = self.find_parent_type(g, pos_trad, typ, gov, index_child)
        dist = 1
        while (
            p_t == "D"
        ):  # recursive call to deal with case where a lot of word are missing
            dist += 1
            if dist > 10000:  # infinite loop, information lost, attach to root
                g = self.alphabet[0]["-1@ROOT"]
                dist = 999999
                break
            if (
                p_gov == self.alphabet[0]["-1@ROOT"]
            ):  # promote 1st element linked to missing root as root
                g = self.alphabet[0]["-1@ROOT"]
                break
            else:
                g = self.alphabet[0]["-1@DELETION"]

            p_t, p_gov, index = self.find_parent_type(p_gov, pos_trad, typ, gov, index)
        if (
            index == index_child
        ):  # check if cycle. Create root, will be dealt with in root resolution
            dist = 999999
            g = self.alphabet[0]["-1@ROOT"]
        elif g != self.alphabet[0]["-1@ROOT"]:
            dist = 999999  # not root
            g = self.create_new_label_from_index(pos_trad, typ, index, index_child)
        d = self.alphabet[1]["DELETION"]
        return g, d, dist

    def multiple_root(self, new_gov):
        """
        Detect if there is mulitple root in the tree
        Parameters
        ----------
        new_gov

        Returns
        Bool : true if mulitple root, false otherwise
        -------

        """
        root = self.alphabet[0]["-1@ROOT"]
        root_exist = False
        for gov in new_gov:
            if gov == root:
                if root_exist:
                    return True
                root_exist = True
        return False

    def collapse_root(self, new_gov, pos_list, typ, dist_from_root):
        """
        Collapse the multiple root to a single one. The one with the closest distance to the old root.
        If multiple candidate, leftmost. => should probably be changed to the one with the biggest subtree ?
        Parameters
        ----------
        new_gov
        pos_list
        typ
        dist_from_root

        Returns
        -------

        """
        new_root_index = dist_from_root.index(min(dist_from_root))
        root = self.alphabet[0]["-1@ROOT"]
        for i, gov in enumerate(new_gov):
            if gov == root:
                if i == new_root_index:
                    continue
                new_gov[i] = self.create_new_label_from_index(
                    pos_list, typ, new_root_index, i
                )
        return new_gov

    def create_new_label_from_index(self, pos_list, typ, target_index, index_child):
        """
        generate new Label from the transformed element
        Parameters
        ----------
        pos_list
        typ
        target_index
        index_child

        Returns
        -------

        """
        if typ[target_index] == "D":
            raise ValueError("index of parent link to a deleted element")
        direction = target_index - index_child
        cpt = 0
        if direction > 0:
            for i in range(index_child + 1, target_index + 1):
                if pos_list[i] == pos_list[target_index] and typ[i] != "D":
                    cpt += 1
        else:
            for i in reversed(range(target_index, index_child)):
                if pos_list[i] == pos_list[target_index] and typ[i] != "D":
                    cpt -= 1
        try:
            assert cpt != 0
        except AssertionError as e:
            print(pos_list)
            print(typ)
            print(direction)
            print(target_index)
            print(index_child)
            raise AssertionError() from e
        result = f"{cpt}@{pos_list[target_index]}"
        if cpt > 0:
            result = "+" + result
        return self.alphabet[0][result]

    def find_parent_type(self, head, pos, typ, govList, index):
        """
        Find if the direct parrent of a node is deleted, substitued or correct.
        If deleted, we use this information to construct new tree
        Parameters
        ----------
        head
        pos
        typ
        govList
        index

        Returns
        -------

        """
        assert len(pos) == len(govList) and len(pos) == len(typ)
        fields = self.reverse[0][head].split("@")
        # print(f" index : {index} {fields}")
        # print(govList)
        # print([self.reverse[2].get(p.item()) for p in pos])
        try:
            cpt = int(fields[0])
        except ValueError as e:
            print(
                f"head: {head}, pos: {pos}, typ: {typ}, govList: {govList}, index: {index}"
            )
            raise ValueError() from e
        posToFind = fields[1]
        if posToFind == "ROOT":
            return "C", "ROOT", index
        if cpt > 0:
            for i in range(
                index + 1, len(govList)
            ):  # start iteration after the index position to the end of sentence
                if pos[i] == posToFind:
                    cpt -= 1
                if cpt == 0:
                    # print(i)
                    try:
                        return typ[i], govList[i].item(), i
                    except AttributeError:  # not tensor
                        return typ[i], govList[i], i
        else:
            for i in reversed(range(0, index)):
                if pos[i] == posToFind:
                    cpt += 1
                if cpt == 0:
                    # print(i)
                    try:
                        return typ[i], govList[i].item(), i
                    except AttributeError:  # not tensor
                        return typ[i], govList[i], i
        raise RuntimeError(
            f"Dynamic oracle did not manage to find head of one element.{fields[0]}@{posToFind}. index : {index}, POS {pos}"
        )

    ###############################################################################################"""
    # TODO: Move this to test package
    import unittest

    class OracleRelPosTest(unittest.TestCase):
        def setUp(self):
            alphabet_gov = {
                "-1@ROOT": 0,
                "-1@VRB": 1,
                "+1@VRB": 2,
                "+2@VRB": 3,
                "-2@VRB": 4,
                "-1@NOUN": 5,
                "+1@NOUN": 6,
                "+2@NOUN": 7,
                "-2@NOUN": 8,
                "-1@CLS": 9,
                "+1@CLS": 10,
                "-1@DELETION": 11,
                "-3@VRB": 12,
                "-1@PRQ": 13,
            }

            alphabet_dep = {"dep": 0, "aux": 1, "DELETION": 2, "INSERTION": 3}
            POS_LIST = ["VRB", "NOUN", "CLS", "ADV", "PRQ", "CSU", "COO", "DET"]
            alphabet_POS = {key: i for i, key in enumerate(POS_LIST)}
            self.alphabet = [alphabet_gov, alphabet_dep, alphabet_POS]

            self.reverse = [
                {item: key for (key, item) in alphabet_gov.items()},
                {item: key for (key, item) in alphabet_dep.items()},
                {item: key for (key, item) in alphabet_POS.items()},
            ]
            self.d_oracle = OracleRelPos(self.alphabet, self.reverse)

        def test_find_parent_type(self):
            POS = ["CLS", "VRB", "NOUN", "NOUN"]  # dummy phrase like, i am SURNAME NAME
            HEAD = ["+1@VRB", "-1@ROOT", "-1@VRB", "-1@NOUN"]
            TYP = ["C", "C", "C", "C"]
            tensor_HEAD = torch.tensor([self.alphabet[0][h] for h in HEAD])
            p_t, p_gov, index = self.d_oracle.find_parent_type(
                self.alphabet[0][HEAD[0]], POS, TYP, tensor_HEAD, 0
            )
            self.assertEqual("C", p_t)
            self.assertEqual(tensor_HEAD[1].item(), p_gov)
            self.assertEqual(1, index)
            p_t, p_gov, index = self.d_oracle.find_parent_type(
                self.alphabet[0][HEAD[2]], POS, TYP, tensor_HEAD, 2
            )
            self.assertEqual("C", p_t)
            self.assertEqual(tensor_HEAD[1].item(), p_gov)
            self.assertEqual(1, index)

        def test_find_parent_type_MiddleDeleted(self):
            POS = [
                "CLS",
                "VRB",
                "VRB",
                "VRB",
                "NOUN",
            ]  # dummy phrase like, i am SURNAME NAME
            HEAD = ["+2@VRB", "+1@VRB", "-1@ROOT", "-1@VRB", "-2@VRB"]
            TYP = ["C", "D", "C", "D", "C"]
            tensor_HEAD = torch.tensor([self.alphabet[0][h] for h in HEAD])
            p_t, p_gov, index = self.d_oracle.find_parent_type(
                self.alphabet[0][HEAD[0]], POS, TYP, tensor_HEAD, 0
            )
            self.assertEqual("C", p_t)
            self.assertEqual(tensor_HEAD[2].item(), p_gov)
            self.assertEqual(2, index)
            p_t, p_gov, index = self.d_oracle.find_parent_type(
                self.alphabet[0][HEAD[4]], POS, TYP, tensor_HEAD, 4
            )
            self.assertEqual("C", p_t)
            self.assertEqual(tensor_HEAD[2].item(), p_gov)
            self.assertEqual(2, index)

        def test_find_parent_type_Jump(self):
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            HEAD = ["+2@VRB", "-1@ROOT", "-1@VRB", "-2@VRB", "-1@VRB"]
            TYP = ["C", "C", "C", "C", "C"]
            tensor_HEAD = torch.tensor([self.alphabet[0][h] for h in HEAD])
            p_t, p_gov, index = self.d_oracle.find_parent_type(
                self.alphabet[0][HEAD[0]], POS, TYP, tensor_HEAD, 0
            )  # forward
            self.assertEqual("C", p_t)
            self.assertEqual(tensor_HEAD[2].item(), p_gov)
            self.assertEqual(2, index)
            p_t, p_gov, index = self.d_oracle.find_parent_type(
                self.alphabet[0][HEAD[3]], POS, TYP, tensor_HEAD, 3
            )  # backward
            self.assertEqual("C", p_t)
            self.assertEqual(tensor_HEAD[1].item(), p_gov)
            self.assertEqual(1, index)

        def test_create_new_label_from_index(self):
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["C", "C", "C", "C", "C"]
            x = self.d_oracle.create_new_label_from_index(POS, TYP, 1, 0)
            self.assertEqual(self.alphabet[0]["+1@VRB"], x)
            x = self.d_oracle.create_new_label_from_index(POS, TYP, 2, 0)
            self.assertEqual(self.alphabet[0]["+2@VRB"], x)
            x = self.d_oracle.create_new_label_from_index(POS, TYP, 0, 1)
            self.assertEqual(self.alphabet[0]["-1@VRB"], x)
            x = self.d_oracle.create_new_label_from_index(POS, TYP, 0, 2)
            self.assertEqual(self.alphabet[0]["-2@VRB"], x)

        def test_create_new_label_from_index_middle_deletion(self):
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["C", "D", "C", "D", "C"]
            x = self.d_oracle.create_new_label_from_index(POS, TYP, 2, 0)
            self.assertEqual(self.alphabet[0]["+1@VRB"], x)
            x = self.d_oracle.create_new_label_from_index(POS, TYP, 2, 4)
            self.assertEqual(self.alphabet[0]["-1@VRB"], x)
            POS = ["ADV", "PRQ", "CSU", "COO", "DET", "DET", "NOM"]
            TYP = ["C", "D", "S", "C", "C", "C", "S"]
            with self.assertRaises(ValueError):
                x = self.d_oracle.create_new_label_from_index(POS, TYP, 1, 2)

        def test_multiple_root(self):
            HEAD = ["+2@VRB", "-1@ROOT", "-1@VRB", "-2@VRB", "-1@VRB"]
            HEAD_n = [self.alphabet[0][x] for x in HEAD]
            self.assertFalse(self.d_oracle.multiple_root(HEAD_n))
            HEAD = ["+2@VRB", "-1@ROOT", "-1@ROOT", "-2@VRB", "-1@VRB"]
            HEAD_n = [self.alphabet[0][x] for x in HEAD]
            self.assertTrue(self.d_oracle.multiple_root(HEAD_n))

        def test_collapse_roots(self):
            HEAD = ["+2@VRB", "-1@ROOT", "-1@ROOT", "-2@VRB", "-1@VRB"]
            HEAD_n = [self.alphabet[0][x] for x in HEAD]
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["C", "C", "C", "C", "C"]
            DIST = [99999, 1, 2, 99999, 99999]
            x = self.d_oracle.collapse_root(HEAD_n, POS, TYP, DIST)
            res = ["+2@VRB", "-1@ROOT", "-1@VRB", "-2@VRB", "-1@VRB"]
            res_n = [self.alphabet[0][x] for x in res]
            self.assertEqual(res_n, x)

        def test_find_closest_parent(self):
            # one deletion
            HEAD = ["+2@VRB", "-1@ROOT", "-1@VRB", "-2@VRB", "-1@VRB"]
            HEAD_n = torch.tensor([self.alphabet[0][x] for x in HEAD])
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["C", "C", "D", "C", "C"]
            gov, dep, dist = self.d_oracle.find_closest_parent(TYP, HEAD_n, POS, 0)
            self.assertEqual(self.alphabet[0]["+1@VRB"], gov)
            self.assertEqual(self.alphabet[1]["DELETION"], dep)
            self.assertEqual(999999, dist)
            # multiple deletion
            HEAD = ["+2@VRB", "-1@ROOT", "-1@VRB", "-3@VRB", "-1@VRB"]
            HEAD_n = torch.tensor([self.alphabet[0][x] for x in HEAD])
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["D", "C", "D", "C", "C"]
            gov, dep, dist = self.d_oracle.find_closest_parent(TYP, HEAD_n, POS, 3)
            self.assertEqual(self.alphabet[0]["-1@VRB"], gov)
            self.assertEqual(self.alphabet[1]["DELETION"], dep)
            self.assertEqual(999999, dist)
            # missing root and multiple deletion
            HEAD = ["+2@VRB", "-1@ROOT", "-1@VRB", "-3@VRB", "-1@VRB"]
            HEAD_n = torch.tensor([self.alphabet[0][x] for x in HEAD])
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["C", "D", "D", "C", "C"]
            gov, dep, dist = self.d_oracle.find_closest_parent(TYP, HEAD_n, POS, 0)
            self.assertEqual(self.alphabet[0]["-1@ROOT"], gov)
            self.assertEqual(self.alphabet[1]["DELETION"], dep)
            self.assertEqual(3, dist)
            # cycle coming back to itself
            HEAD = ["+2@VRB", "-1@ROOT", "-2@VRB", "-3@VRB", "-1@VRB"]
            HEAD_n = torch.tensor([self.alphabet[0][x] for x in HEAD])
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["C", "D", "D", "C", "C"]
            gov, dep, dist = self.d_oracle.find_closest_parent(TYP, HEAD_n, POS, 0)
            self.assertEqual(self.alphabet[0]["-1@ROOT"], gov)
            self.assertEqual(self.alphabet[1]["DELETION"], dep)
            self.assertEqual(999999, dist)
            # infinite loop cycle
            HEAD = ["+2@VRB", "-1@ROOT", "-2@VRB", "-3@VRB", "-1@VRB"]
            HEAD_n = torch.tensor([self.alphabet[0][x] for x in HEAD])
            POS = ["VRB", "VRB", "VRB", "VRB", "NOUN"]
            TYP = ["D", "D", "D", "C", "C"]
            gov, dep, dist = self.d_oracle.find_closest_parent(TYP, HEAD_n, POS, 3)
            self.assertEqual(self.alphabet[0]["-1@ROOT"], gov)
            self.assertEqual(self.alphabet[1]["DELETION"], dep)
            self.assertEqual(999999, dist)
