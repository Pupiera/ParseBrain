from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    GoldConfiguration,
    Word,
    Configuration,
)
from parsebrain.processing.dependency_parsing.transition_based.transition import (
    CovingtonTransition,
)
from parsebrain.processing.dependency_parsing.transition_based.dynamic_oracle import (
    DynamicOracle,
)
from tarjan import tarjan
from copy import deepcopy

"""
Need to think how to cleanly deal 
"""


# Dynamic alignment_oracle from "A Dynamic Oracle for Arc-Eager Dependency Parsing" Goldberg, Yoav, and Joakim Nivre

# Maybe need to rework the actions the parser can take in a specific config. ( to not have multiple implementation of
# those rules)


class DynamicOracleCovington(DynamicOracle):
    def get_oracle_move_from_config_tree(
        self, configuration: Configuration, gold_tree: GoldConfiguration
    ):
        """
        need to run loss computation 4 time, not very efficient...
        @param configuration:
        @param gold_tree:
        @return:
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import Configuration
        >>> from parsebrain.processing.dependency_parsing.transition_based.transition import CovingtonTransition
        >>> config = Configuration(["j","ai","vu","jamais","eu","ça","pour","un","une","un","colloque"],
        ... ["j","ai","vu","jamais","eu","ça","pour","un","une","un","colloque"], root_embedding=[0])
        >>> config.buffer_string = [Word("j",1),Word("ai",2),Word("vu",3),
        ... Word("jamais",4), Word("eu",5),Word("ça",6),Word("pour",7),
        ... Word("un",8),Word("une",9),Word("un",10),Word("colloque",11)]
        >>> config_gold = GoldConfiguration()
        >>> heads = [5, 3, 5, 5, 0, 5, 5, 11, 8, 9, 7]
        >>> for i, h in enumerate(heads): config_gold.heads[i+1] = h
        >>> deps = [[], [], [2], [], [1,3,4,6,7], [], [11], [9], [10], [], [8]]
        >>> for i, d in enumerate(deps): config_gold.deps[i+1] = d
        >>> alignment_oracle = DynamicOracleCovington()
        >>> m = alignment_oracle.get_oracle_move_from_config_tree(config, config_gold)
        >>> print(m)
        0
        >>> transiton = CovingtonTransition()
        >>> list_decision = []
        >>> while len(config.buffer) != 0:
        ...     decision = alignment_oracle.get_oracle_move_from_config_tree(config, config_gold)
        ...     list_decision.append(decision)
        ...     config = transiton.apply_decision(decision, config)
        >>> print(list_decision)
        [0, 0, 1, 0, 0, 1, 1, 3, 1, 2, 0, 2, 0, 3, 2, 0, 0, 2, 0, 2, 0, 3, 3, 1, 2, 0]
        """
        # if config is terminal:
        if CovingtonTransition.is_terminal(configuration):
            return self.padding_value  # padding
        if len(configuration.stack_string) == 0 and configuration.has_root:
            return CovingtonTransition.SHIFT

        decision_cost = [99999, 99999, 99999, 99999]
        # avoid doing computation if not valid
        if CovingtonTransition.shift_condition(configuration):
            decision_cost[CovingtonTransition.SHIFT] = self.compute_shift_cost(
                configuration, gold_tree
            )
        if CovingtonTransition.left_arc_condition(configuration):
            decision_cost[CovingtonTransition.LEFT] = self.compute_left_arc_cost(
                configuration, gold_tree
            )
        if CovingtonTransition.right_arc_condition(configuration):
            decision_cost[CovingtonTransition.RIGHT] = self.compute_right_arc_cost(
                configuration, gold_tree
            )
        if CovingtonTransition.no_arc_condition(configuration):
            decision_cost[CovingtonTransition.NO_ARC] = self.compute_no_arc_cost(
                configuration, gold_tree
            )

        min_cost = min(decision_cost)
        return decision_cost.index(min_cost)

    def compute_shift_cost(
        self, configuration: Configuration, gold_tree: GoldConfiguration
    ):
        """
        Copy config, and shift then compute the loss
        @param configuration:
        @param gold_tree:
        @return:
        """
        if not CovingtonTransition.shift_condition(configuration):
            return 99999
        # create a light copy of config
        new_conf = self.copy_config(configuration)
        new_conf = CovingtonTransition.shift(new_conf)
        return self.compute_loss_from_config(new_conf, gold_tree)

    def compute_left_arc_cost(
        self, configuration: Configuration, gold_tree: GoldConfiguration
    ):
        if not CovingtonTransition.left_arc_condition(configuration):
            return 99999
            # create a light copy of config

        new_conf = self.copy_config(configuration)
        new_conf = CovingtonTransition.left_arc(new_conf)
        return self.compute_loss_from_config(new_conf, gold_tree)

    def compute_right_arc_cost(
        self, configuration: Configuration, gold_tree: GoldConfiguration
    ):
        if not CovingtonTransition.right_arc_condition(configuration):
            return 99999
            # create a light copy of config
        new_conf = self.copy_config(configuration)
        new_conf = CovingtonTransition.right_arc(new_conf)
        return self.compute_loss_from_config(new_conf, gold_tree)

    def compute_no_arc_cost(
        self, configuration: Configuration, gold_tree: GoldConfiguration
    ):
        if not CovingtonTransition.no_arc_condition(configuration):
            return 99999
            # create a light copy of config
        new_conf = self.copy_config(configuration)
        new_conf = CovingtonTransition.no_arc(new_conf)
        return self.compute_loss_from_config(new_conf, gold_tree)

    def compute_loss_from_config(
        self, configuration: Configuration, gold_tree: GoldConfiguration
    ):
        """
        Algorithm 1 of "An Efficient Dynamic Oracle for Unrestricted Non-Projective Parsing"
        @param configuration:
        @param gold_tree:
        @return:
        """
        # When nothing on stack, not computable..., well i guess this means that this must be a shift ?
        U = set([])
        gold_arc = set([(h, d) for d, h in gold_tree.heads.items()])
        A = set([(x.head.position, x.dependent.position) for x in configuration.arc])
        gold_arc_not_done = gold_arc.difference(A)
        # case where finishing by shift, loss of this config is equal to the number of gold_arc not done
        if len(configuration.buffer_string) == 0:
            return len(gold_arc_not_done)

        wj = configuration.buffer_string[0]
        if len(configuration.stack_string) > 0:
            wi = configuration.stack_string[-1]
        # else take the root to do this loss computation
        else:
            wi = configuration.root_token
        # x -> y
        for x, y in gold_arc_not_done:
            left = min(x, y)
            right = max(x, y)
            if (
                wj.position > right
                or (wj.position == right and wi.position < left)
                or self.has_head_not_gold(y, configuration.arc, x)
                or self.weakly_connected(A, x, y)
            ):
                U.add((x, y))

        I = gold_arc.difference(U)
        return len(U) + self.count_cycles(A, I)

    def has_head_not_gold(self, to_check, arcs, gold_head):
        """
        Check if a word has the correct head
        @param to_check:
        @param arcs:
        @param gold_head:
        @return:
        """
        for a in arcs:
            if a.dependent.position == to_check:
                # early stoping, no need to iterate the rest, just check if the head is gold.
                return not (a.head.position == gold_head)
        return False

    # TODO: This can be done more efficient
    # TODO: check this function, there is an infinite loop here.
    # O(n^2)
    def weakly_connected(self, arcs, head, dependent):
        """
        "can also be resolved in O(1), by querying the disjoint set data structure
        that implementations of the Covington algorithm"
        Todo check this implementation
        @param arcs:
        @param head:
        @param dependent:
        @return:
        """
        weakly_connected = False
        end_path = False
        parent = head
        y = dependent
        A = arcs
        while parent != 0 and not weakly_connected and not end_path and A != set([]):
            if (parent, y) in A:
                weakly_connected = True
                break
            else:

                for (a, b) in A:
                    if b == parent:
                        parent = a
                        break
                    else:
                        end_path = True

        return weakly_connected

    def count_cycles(self, arcs, I):
        """
        Tarjan (1972) implementation at https://github.com/bwesterb/py-tarjan/
        O(n)
        @param arcs:
        @param I:
        @return:
        """
        A = arcs.union(I)
        d = {}
        for a, b in A:
            if a not in d:
                d[a] = [b]
            else:
                d[a].append(b)

        return sum([1 for e in tarjan(d) if len(e) > 1])

    def copy_config(self, configuration):
        """
        Custom copy of config, to avoid cloning the heavy tensor,
        we just copy the surface level information (_string)
        @param configuration:
        @return:
        """
        new_conf = Configuration(
            features=None,
            features_string=configuration.buffer_string,
            root_embedding=[0],
        )
        new_conf.stack = deepcopy(configuration.stack_string)
        new_conf.stack_string = deepcopy(configuration.stack_string)
        new_conf.stack2 = deepcopy(configuration.stack2_string)
        new_conf.stack2_string = deepcopy(configuration.stack2_string)
        new_conf.buffer = deepcopy(configuration.buffer_string)
        new_conf.buffer_string = deepcopy(configuration.buffer_string)
        new_conf.arc = deepcopy(configuration.arc)

        return new_conf

    def compute_label(
        self,
        configuration: Configuration,
        gold_configuration: GoldConfiguration,
        decision: int,
    ):
        transition = CovingtonTransition()
        # get info of first element of stack
        if len(configuration.stack_string) > 0:
            stack_pos = configuration.stack_string[-1].position
        # root info not yet applied so still false
        elif not configuration.has_root:
            stack_pos = 0
        else:
            return self.padding_value
        # get info of first element of buffer
        try:
            buffer_pos = configuration.buffer_string[0].position
        except IndexError:
            return self.padding_value
        # if decision is right arc, stack elt is head
        if decision == transition.RIGHT:
            # Check if this arc exist in gold config
            if gold_configuration.heads[buffer_pos] == stack_pos:
                return gold_configuration.label[buffer_pos]
        # if decision is left arc, buffer elt is head
        elif decision == transition.LEFT:
            # Check if this arc exist in gold config
            if stack_pos == 0:
                return self.padding_value
            if gold_configuration.heads[stack_pos] == buffer_pos:
                return gold_configuration.label[stack_pos]
        # if not return -1 (reinforce current prediction of model)
        return self.padding_value


if __name__ == "__main__":
    import doctest

    doctest.testmod()
