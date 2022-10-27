from typing import List

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
    Word,
    GoldConfiguration,
)
from parsebrain.processing.dependency_parsing.transition_based.transition import (
    ArcEagerTransition,
)

"""
The function compute_sequence is very generic. Maybe for other static oracle, 
we can create subclass from it and only define best_decision
"""


class StaticOracle_ArcEager:
    def __init__(self):
        self.transition = ArcEagerTransition()

    def compute_sequence(
        self, batch_gold_config: List[GoldConfiguration]
    ) -> List[List[int]]:
        """
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import GoldConfiguration
        >>> head1 = [2, 0, 2, 3]
        >>> head2 = [0, 3, 1, 1]
        >>> g_c = [GoldConfiguration(head1), GoldConfiguration(head2)]
        >>> s_oracle = StaticOracle_ArcEager()
        >>> s_oracle.compute_sequence(g_c)
        [[0, 1, 0, 2, 2], [0, 0, 1, 2, 3, 2]]
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
                s_seq.append(decision)
            static_sequence.append(s_seq)
        return static_sequence

    def best_decision(
        self, config: Configuration, gold_config: GoldConfiguration
    ) -> int:
        try:
            stack_top = config.stack_string[-1]
        except IndexError:
            if self.transition.shift_condition(config):
                return self.transition.SHIFT
            else:
                raise IndexError(
                    "Static Oracle Arc-Eager : Stack is empty but shift transition not valid in current configuration"
                )
        buffer_first = config.buffer_string[0]
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
