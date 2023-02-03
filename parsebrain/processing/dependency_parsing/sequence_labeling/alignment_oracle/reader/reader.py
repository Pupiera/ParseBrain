from typing import List


class Reader:
    def read(
        self, alignment, original_gov: List, original_dep: List, original_pos: List
    ):
        raise NotImplementedError()
