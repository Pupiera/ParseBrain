from typing import List
from .reader import Reader


class ReaderSpeechbrain(Reader):
    def read(
        self, alignment, original_gov: List, original_dep: List, original_pos: List
    ):
        """
        Return the corresponding annotation based on the current alignment (ie if a word was added, add anotation too so
        it's still one-one.

        >>> from speechbrain.utils.edit_distance import wer_details_for_batch
        >>> ref = [['this', 'is', 'a', 'test'], ['doctest','is', 'usefull']]
        >>> hyp = [['this', 'a', 'tests'], ['doctest','is' , 'very', 'usefull', 'nice']]
        >>> wer_details = wer_details_for_batch(ids=['test1','test2'], refs = ref, hyps=hyp, compute_alignments=True)
        >>> wer_alig = [a['alignment'] for a in wer_details]
        >>> og_gov = [[1,2,3,4],[4,5,6]]
        >>> og_dep = [[1,2,3,4], [4,5,6]]
        >>> og_pos = [[1,2,3,4], [4,5,6]]
        >>> reader = ReaderSpeechbrain()
        >>> alphabet =[{'-1@INSERTION': 0}, {'INSERTION':0}, {'INSERTION':0}]
        >>> reverse = [{v : k for k, v in alphabet[0].items()},
        ...            {v : k for k, v in alphabet[1].items()},
        ...            {v : k for k, v in alphabet[2].items()}]
        >>> reader.set_alphabet(alphabet, reverse)
        >>> reader.read(wer_alig, og_gov, og_dep, og_pos)
        """
        govs = []
        deps = []
        poss = []
        types = []
        for alig, gov, dep, pos in zip(
            alignment, original_gov, original_dep, original_pos
        ):
            tmp_gov = []
            tmp_dep = []
            tmp_pos = []
            tmp_type = []
            for a in alig:
                gold_index = a[1]
                if a[0] == "I":
                    tmp_gov.append(self.alphabet[0]["-1@INSERTION"])
                    tmp_dep.append(self.alphabet[1]["INSERTION"])
                    tmp_pos.append(self.alphabet[2]["INSERTION"])
                    tmp_type.append("I")
                else:
                    tmp_gov.append(gov[gold_index])
                    tmp_dep.append(dep[gold_index])
                    tmp_pos.append(pos[gold_index])
                if a[0] == "=":
                    tmp_type.append("C")
                elif a[0] == "D":
                    tmp_type.append("D")
                elif a[0] == "S":
                    tmp_type.append("S")
            govs.append(tmp_gov)
            deps.append(tmp_dep)
            poss.append(tmp_pos)
            types.append(tmp_type)
        return types, govs, deps, poss


if __name__ == "__main__":
    import doctest

    doctest.testmod()
