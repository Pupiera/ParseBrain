from typing import List
from parsebrain.processing.dependency_parsing.sequence_labeling.alignment_oracle.reader import (
    Reader,
)

# todo : make this independent from RELPOS (remove the "-1@INSERTION", cleanly)
# Moreover, this is tailored for case where there is three task, need to make it more agile.


class ReaderSpeechbrain(Reader):
    def read(
        self, alignment: str, original_gov: List, original_dep: List, original_pos: List
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
        >>> reader.read(wer_alig, og_gov, og_dep, og_pos)
        ([['C', 'D', 'C', 'S'], ['C', 'C', 'I', 'C', 'I']], [[1, 2, 3, 4], [4, 5, 'INSERTION', 6, 'INSERTION']], [[1, 2, 3, 4], [4, 5, 'INSERTION', 6, 'INSERTION']], [[1, 2, 3, 4], [4, 5, 'INSERTION', 6, 'INSERTION']])
        """
        govs = []
        deps = []
        poss = []
        types = []
        for index_error ,(alig, gov, dep, pos) in enumerate(zip(
            alignment, original_gov, original_dep, original_pos
        )):
            tmp_gov = []
            tmp_dep = []
            tmp_pos = []
            tmp_type = []
            for a in alig:
                gold_index = a[1]
                if a[0] == "I":
                    tmp_gov.append("INSERTION")
                    tmp_dep.append("INSERTION")
                    tmp_pos.append("INSERTION")
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
