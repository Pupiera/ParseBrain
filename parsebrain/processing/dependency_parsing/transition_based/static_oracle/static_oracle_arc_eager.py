from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
    Word,
    GoldConfiguration,
)
from parsebrain.processing.dependency_parsing.transition_based.transition import (
    ArcEagerTransition,
)
from parsebrain.processing.dependency_parsing.transition_based.static_oracle import (
    StaticOracle,
)

"""
The function compute_sequence is very generic. Maybe for other static oracle, 
we can create subclass from it and only define best_decision
Note for the arc eager oracle, if the sentence is non-projective, it won't manage to create tree. 
"""


class StaticOracleArcEager(StaticOracle):
    def __init__(self, padding_value: int = 1000):
        super().__init__(padding_value)
        self.transition = ArcEagerTransition()

    def compute_sequence(
        self, batch_gold_config: List[GoldConfiguration]
    ) -> torch.Tensor:
        """
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import GoldConfiguration
        >>> head1 = [2, 0, 2, 3]
        >>> head2 = [0, 3, 1, 1]
        >>> g_c = [GoldConfiguration(head1), GoldConfiguration(head2)]
        >>> s_oracle = StaticOracleArcEager()
        >>> s_oracle.compute_sequence(g_c)
        tensor([[ 0,  1,  2,  2,  2, -1],
                [ 2,  0,  1,  2,  3,  2]])
        """

        static_sequence = []
        for gc in batch_gold_config:
            s_seq = []
            # we create a new config, only to compute the gold sequence
            # This will not update the config we will use for the parse
            current_config = Configuration(
                [x for x in gc.heads.keys()],
                [Word(str(x), x) for x in gc.heads.keys()],
            )
            while not self.transition.is_terminal(current_config):
                decision = self.best_decision(current_config, gc)
                self.transition.apply_decision(decision, current_config)
                s_seq.append(torch.tensor(decision))
            static_sequence.append(torch.stack(s_seq))
        return pad_sequence(
            static_sequence, batch_first=True, padding_value=self.padding_value
        )

    def best_decision(
        self, config: Configuration, gold_config: GoldConfiguration
    ) -> int:
        # when buffer is empty, can only reduce.
        if len(config.buffer_string) == 0:
            return self.transition.REDUCE

        buffer_first = config.buffer_string[0]
        if len(config.stack_string) > 0:
            stack_top = config.stack_string[-1]
        # if root and nothing on stack, create the arc
        elif gold_config.heads[buffer_first.position] == 0 and not config.has_root:
            return self.transition.RIGHT
        elif self.transition.shift_condition(config):
            return self.transition.SHIFT
        else:
            raise IndexError(
                "Static Oracle Arc-Eager : Stack is empty but shift transition not valid in current configuration"
            )
        if gold_config.heads[buffer_first.position] == stack_top.position:
            return self.transition.RIGHT
        elif gold_config.heads[stack_top.position] == buffer_first.position:
            return self.transition.LEFT
        elif (
            self.transition.reduce_condition(config)
            and len(
                [
                    x
                    for x in config.buffer_string
                    if x.position in gold_config.deps[stack_top.position]
                ]
            )
            == 0
        ):
            return self.transition.REDUCE
        else:
            return self.transition.SHIFT


if __name__ == "__main__":
    import doctest

    doctest.testmod()
