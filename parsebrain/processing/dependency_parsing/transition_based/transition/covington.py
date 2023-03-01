import torch

from processing.dependency_parsing.transition_based.configuration import Configuration
from transition import Transition
from tarjan import tarjan
from arc import LeftArc, RightArc


class CovingtonTransition(Transition):
    def __init__(self):
        super().__init__()
        self.SHIFT = 0
        self.LEFT = 1
        self.RIGHT = 2
        self.NO_ARC = 3

    def get_transition_dict(self) -> dict:
        return {
            "SHIFT": self.SHIFT,
            "NO_ARC": self.NO_ARC,
            "LEFT": self.LEFT,
            "RIGHT": self.RIGHT,
        }

    def get_relation_from_decision(
        self, decision: int, config: Configuration
    ) -> (torch.Tensor, torch.tensor):
        pass

    def is_decision_valid(self, decision: int, config: Configuration) -> bool:
        if self.is_terminal(config):
            return False
        if decision == self.SHIFT:
            return self.shift_condition(config)
        elif decision == self.NO_ARC:
            return self.no_arc_condition(config)
        elif decision == self.LEFT:
            return self.left_arc_condition(config)
        elif decision == self.RIGHT:
            return self.right_arc_condition(config)
        else:
            raise ValueError(
                f"Decision {decision} out of scope for arc-eager transition"
            )

    def apply_decision(self, decision: int, config: Configuration) -> Configuration:
        if self.is_terminal(config):
            return config
        if decision == self.SHIFT:
            return self.shift(config)
        elif decision == self.NO_ARC:
            return self.no_arc(config)
        elif decision == self.LEFT:
            return self.left_arc(config)
        elif decision == self.RIGHT:
            return self.right_arc(config)
        else:
            raise ValueError(
                f"Decision number ({decision}) is out of scope for covington transition"
            )

    def is_terminal(self, config: Configuration) -> bool:
        return len(config.buffer_string) == 0

    def update_tree(self, decision: int, config: Configuration, tree: dict) -> dict:
        pass

    def require_label(self, decision: int) -> bool:
        return decision in [self.RIGHT, self.LEFT]

    @staticmethod
    def shift_condition(config):
        return len(config.buffer_string) != 0

    @staticmethod
    def shift(config: Configuration):
        """
        Shift: 〈λ1, λ2, j|B, A〉 ⇒ 〈λ1 · λ2|j, [], B, A
        @param config:
        @return:
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf.stack, conf.stack_string, conf.stack2, conf.stack2_string
        (['Hey'], ['Hey'], [], [])
        >>> conf = CovingtonTransition.no_arc(conf)
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf.stack, conf.stack_string, conf.stack2, conf.stack2_string
        (['Hey', 'Parsing'], ['Hey', 'Parsing'], [], [])
        """

        wj = config.buffer[0]
        wj_string = config.buffer_string[0]

        config.stack2.append(wj)
        config.stack2_string.append(wj_string)

        config.stack.extend(config.stack2)
        config.stack_string.extend(config.stack2_string)

        config.stack2 = []
        config.stack2_string = []

        config.shift_buffer()
        return config

    @staticmethod
    def left_arc_condition(config):
        """
        〈λ1|i, λ2, j|B, A〉 ⇒ 〈λ1, i|λ2, j|B, A ∪ {j → i}〉
        only if not_exist(k) | k → i ∈ A (single-head) and i →∗ j !∈ A (acyclicity)
        @param config:
        @return:
        """

        wi_string = config.stack_string[-1]
        wj_string = config.buffer_string
        # single head and acyclicity check
        if CovingtonTransition.has_head(
            wi_string, config.arc
        ) or CovingtonTransition.cycle_check(
            head=wj_string, dependent=wi_string, arcs=config.arc
        ):
            return False
        return True

    @staticmethod
    def cycle_check(head, dependent, arcs):
        """

        @param head:
        @param dependent:
        @param arcs:
        @return: True if the addition of an arc between head and dependent would cause a cycle, False otherwise
        >>> from processing.dependency_parsing.transition_based.configuration import Word
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],[Word("Hey",1), Word("Parsing",2), Word("Is",3), Word("Fun",4)])
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf = CovingtonTransition.left_arc(conf)
        >>> conf = CovingtonTransition.shift(conf)
        >>> CovingtonTransition.cycle_check(2,1, conf.arc)
        False
        >>> conf = CovingtonTransition.left_arc(conf)
        >>> CovingtonTransition.cycle_check(1,3, conf.arc)
        True
        """
        # no cycle possible, so should be 0 before adding the potential arcs
        A = set([(a.head.position, a.dependent.position) for a in arcs])
        A.add((head, dependent))
        d = {}
        for a, b in A:
            if a not in d:
                d[a] = [b]
            else:
                d[a].append(b)

        x = sum([1 for e in tarjan(d) if len(e) > 1])
        return x != 0

    @staticmethod
    def left_arc(config):
        """
        〈λ1|i, λ2, j|B, A〉 ⇒ 〈λ1, i|λ2, j|B, A ∪ {j → i}〉
         only if not_exist(k) | k → i ∈ A (single-head) and i →∗ j !∈ A (acyclicity)
        @param config:
        @return:

        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf = CovingtonTransition.left_arc(conf)
        >>> conf.stack, conf.stack_string, conf.stack2, conf.stack2_string, conf.buffer, conf.buffer_string, str(conf.arc[0])
        ([], [], ['Hey'], ['Hey'], ['Parsing', 'Is', 'Fun'], ['Parsing', 'Is', 'Fun'], 'arc : Parsing -> Hey')

        """
        wj = config.buffer[0]
        wj_string = config.buffer_string[0]

        wi = config.stack.pop()
        wi_string = config.stack_string.pop()

        config.stack2.insert(0, wi)
        config.stack2_string.insert(0, wi_string)

        config.arc.append(LeftArc(head=wj_string, dependent=wi_string))
        return config

    @staticmethod
    def right_arc_condition(config):
        """
        Right-Arc: 〈λ1|i, λ2, j|B, A〉 ⇒ 〈λ1, i|λ2, j|B, A ∪ {i → j}〉
        only if not_exist(k) | k → i ∈ A (single-head) and i →∗ j !∈ A (acyclicity)
        @param config:
        @return:
        """
        wi_string = config.stack_string[-1]
        wj_string = config.buffer_string[0]
        # single head and acyclicity check
        if CovingtonTransition.has_head(
            wj_string, config.arc
        ) or CovingtonTransition.cycle_check(
            head=wi_string, dependent=wj_string, arcs=config.arc
        ):
            return False
        return True

    @staticmethod
    def right_arc(config):
        """
        Right-Arc: 〈λ1|i, λ2, j|B, A〉 ⇒ 〈λ1, i|λ2, j|B, A ∪ {i → j}〉
        only if not_exist(k) | k → i ∈ A (single-head) and i →∗ j !∈ A (acyclicity)
        @param config:
        @return:

        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf = CovingtonTransition.right_arc(conf)
        >>> conf.stack, conf.stack_string, conf.stack2, conf.stack2_string, conf.buffer, conf.buffer_string, str(conf.arc[0])
        ([], [], ['Hey'], ['Hey'], ['Parsing', 'Is', 'Fun'], ['Parsing', 'Is', 'Fun'], 'arc : Hey -> Parsing')
        """
        wj = config.buffer[0]
        wj_string = config.buffer_string[0]

        wi = config.stack.pop()
        wi_string = config.stack_string.pop()

        config.stack2.insert(0, wi)
        config.stack2_string.insert(0, wi_string)

        config.arc.append(RightArc(head=wi_string, dependent=wj_string))
        return config

    @staticmethod
    def no_arc_condition(config):
        return len(config.stack) >= 1

    @staticmethod
    def no_arc(config):
        """

        @param config:
        @return:

        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf = CovingtonTransition.no_arc(conf)
        >>> conf.stack, conf.stack_string, conf.stack2, conf.stack2_string, conf.buffer, conf.buffer_string
        ([], [], ['Hey'], ['Hey'], ['Parsing', 'Is', 'Fun'], ['Parsing', 'Is', 'Fun'])
        """
        wi = config.stack.pop()
        wi_string = config.stack_string.pop()

        config.stack2.insert(0, wi)
        config.stack2_string.insert(0, wi_string)
        return config


if __name__ == "__main__":
    import doctest

    doctest.testmod()
