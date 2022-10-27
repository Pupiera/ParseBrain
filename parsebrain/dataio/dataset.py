"""
Extension of dynamicItemDataset for conllu file
"""
import logging
from typing import List

import speechbrain as sb

from parsebrain.dataio.dataio import load_data_conllu

logger = logging.getLogger(__name__)


class DynamicItemDatasetConllu(sb.dataio.dataset.DynamicItemDataset):
    @classmethod
    def from_conllu(
            cls,
            conllu_path: str,
            keys: List,
            replacements: dict = {},
            dynamic_items: List = [],
            output_keys: List = [],
    ):
        data = load_data_conllu(conllu_path, keys, replacements)
        return cls(data, dynamic_items, output_keys)
