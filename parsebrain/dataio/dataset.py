"""
Extension of dynamicItemDataset for conllu file
"""
from parsebrain.dataio.dataio import load_data_conllu
import logging
import speechbrain as sb

logger = logging.getLogger(__name__)


class DynamicItemDatasetConllu(sb.dataio.dataset.DynamicItemDataset):
    @classmethod
    def from_conllu(
            cls, conllu_path, keys, replacements={}, dynamic_items=[], output_keys=[]
    ):
        data = load_data_conllu(conllu_path, keys, replacements)
        return cls(data, dynamic_items, output_keys)
