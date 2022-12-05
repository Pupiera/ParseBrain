from typing import List

from .reader import Reader


class ReaderSclite(Reader):
    def read(
        self, alignment, original_gov: List, original_dep: List, original_pos: List
    ):
        govs = []
        deps = []
        poss = []
        types = []
        index_gold = 0
        index_sent = 0
        for line in alignment.split("\n"):
            if line.startswith("<") or line == "\n" or line == "":
                continue
            typ = []
            gov = []
            dep = []
            pos = []
            tokens = line.split(":")
            for i, t in enumerate(tokens):
                fields = t.split(",")
                typ.append(fields[0])
                if fields[0] == "D":  # Deletion. Only present in gold
                    gov.append(original_gov[index_sent][index_gold])
                    dep.append(original_dep[index_sent][index_gold])
                    pos.append(original_pos[index_sent][index_gold])
                    index_gold += 1
                elif fields[0] == "I":  # insertion. Only present in predictions
                    gov.append(self.alphabet[0].get("-1@INSERTION"))
                    dep.append(self.alphabet[1].get("INSERTION"))
                    pos.append(self.alphabet[2].get("INSERTION"))
                else:  # Correct or substitution
                    gov.append(original_gov[index_sent][index_gold])
                    dep.append(original_dep[index_sent][index_gold])
                    pos.append(original_pos[index_sent][index_gold])
                    index_gold += 1
            types.append(typ)
            govs.append(gov)
            deps.append(dep)
            poss.append(pos)
            index_gold = 0
            index_sent += 1
        return types, govs, deps, poss
