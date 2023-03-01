from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    GoldConfiguration,
    Word,
    Configuration,
)
from parsebrain.processing.dependency_parsing.transition_based.transition import (
    CovingtonTransition,
)
from .dynamic_oracle import DynamicOracle
from tarjan import tarjan

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
        need to run loss computation 5 time, not very efficient...
        @param configuration:
        @param gold_tree:
        @return:
        """

        current_loss = self.compute_loss_from_config(configuration, gold_tree)
        if len(configuration.buffer) == 0:  # if config is terminal with Arc eager:
            return self.padding_value  # padding
        decision_cost = [
            self.compute_shift_cost(configuration, gold_tree)
            - current_loss
            + int(not CovingtonTransition.shift_condition(configuration)) * 99999,
            self.compute_left_arc_cost(configuration, gold_tree)
            + int(not CovingtonTransition.left_arc_condition(configuration)) * 99999,
            self.compute_right_arc_cost(configuration, gold_tree)
            + int(not CovingtonTransition.right_arc_condition(configuration)) * 99999,
            self.compute_no_arc_cost(configuration, gold_tree)
            + int(not CovingtonTransition.no_arc_condition(configuration)) * 99999,
        ]
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
        U = set([])
        gold_arc = set([(h, d) for h, d in enumerate(gold_tree.heads)])
        A = set([(x.head.position, x.dependent.position) for x in configuration.arc])
        gold_arc_not_done = gold_arc.difference(A)
        wj = configuration.buffer_string[0]
        wi = configuration.stack_string[0][-1]
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
        new_conf = Configuration(
            features=None, features_string=configuration.buffer_string
        )
        new_conf.stack_string = configuration.stack_string
        new_conf.arc = configuration.arc
        return new_conf
