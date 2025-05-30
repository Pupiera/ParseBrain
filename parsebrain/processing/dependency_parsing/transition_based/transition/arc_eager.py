from typing import List

import torch

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
    Word,
)
from .arc import Arc, RightArc, LeftArc
from parsebrain.processing.dependency_parsing.transition_based.transition.transition import (
    Transition,
)


class ArcEagerTransition(Transition):
    SHIFT = 0
    LEFT = 1
    RIGHT = 2
    REDUCE = 3

    @classmethod
    def get_transition_dict(cls) -> dict:
        return {
            "SHIFT": cls.SHIFT,
            "REDUCE": cls.REDUCE,
            "LEFT": cls.LEFT,
            "RIGHT": cls.RIGHT,
        }

    # Some function may need to work on batch of tensor...
    # Not really liking to mix tensor optimization and transition logic...

    def require_label(self, decision: int) -> bool:
        return decision == self.LEFT or decision == self.RIGHT

    def get_relation_from_decision(
        self, decision: int, config: Configuration
    ) -> (torch.Tensor, torch.Tensor):
        """
        Maube this is a specfic computer and should not be here.

        """
        if len(config.stack) > 0:
            stack = config.stack[-1]
        elif not config.has_root:
            stack = config.root.squeeze()
        else:
            device = config.buffer[0].device
            stack = torch.zeros(config.buffer[0].size()).to(device)
        if len(config.buffer) > 0:
            buffer = config.buffer[0]
        else:
            device = config.stack[0].device
            buffer = torch.zeros(config.stack[0].size()).to(device)
        if decision == self.RIGHT:
            return stack, buffer
        elif decision == self.LEFT:
            return buffer, stack
        else:  # if not valid, default = right
            return stack, buffer

    def update_tree(self, decision: int, config: Configuration, tree: dict) -> int:
        """
        Update a dictionary with for a given word, it's position and the position of the head.
        For each word a dictionary is created and will be updated in another function to get the
        label of syntactic relationship.
        """
        key = None
        try:
            stack_head = config.stack_string[-1]
            buffer_head = config.buffer_string[0]
        except IndexError:
            return key
        if decision == self.RIGHT:
            key = buffer_head.position
            tree[key] = {"head": stack_head.position}
        elif decision == self.LEFT:
            key = stack_head.position
            tree[key] = {"head": buffer_head.position}
        return key

    @classmethod
    def apply_decision(cls, decision: int, config: Configuration) -> Configuration:
        """
        apply the given decision
        In case where the statie is terminal do nothing (Padding)
        """
        # todo: if nothing is valid, backup solution. (this means the parser did some very bad thing, such as only shifting)
        if cls.is_terminal(config):
            return config

        if decision == cls.SHIFT:
            return cls.shift(config)
        elif decision == cls.REDUCE:
            return cls.reduce(config)
        elif decision == cls.LEFT:
            return cls.left_arc(config)
        elif decision == cls.RIGHT:
            return cls.right_arc(config)
        else:
            raise ValueError(
                f"Decision number ({decision}) is out of scope for arc-eager transition"
            )

    def is_decision_valid(self, decision: int, config: Configuration) -> bool:
        if self.is_terminal(config):
            return False
        if decision == self.SHIFT:
            return self.shift_condition(config)
        elif decision == self.REDUCE:
            return self.reduce_condition(config)
        elif decision == self.LEFT:
            return self.left_arc_condition(config)
        elif decision == self.RIGHT:
            return self.right_arc_condition(config)
        else:
            raise ValueError(
                f"Decision {decision} out of scope for arc-eager transition"
            )

    @staticmethod
    def is_terminal(config: Configuration) -> bool:
        """
        Condition is terminal if buffer is empty
        (can't create any new arc if everything is on the stack)
        Using buffer string to ignore potentially padded element in buffer.
        """
        return len(config.buffer_string) == 0

    # maybe move this to config ?
    @staticmethod
    def has_head(wi: Word, arc: List[Arc]) -> bool:
        """
        The condition HEAD(wi ) is true in a configuration (σ, β, A) if A contains an arc wk → wi
        Check if wi already has a HEAD. ie. check if wi is the dependent in a dependency.
        :param wi:
        :param arc:
        :return:
        """
        for a in arc:
            if a.dependent == wi:
                return True
        return False

    @staticmethod
    def shift_condition(config: Configuration) -> bool:
        """
        Condition to fullfill for shift to be available
        :return:
        """
        if len(config.buffer_string) >= 2:
            return True
        elif len(config.buffer_string) == 1 and (
            ArcEagerTransition.has_head(config.buffer_string[0], config.arc)
            or len(config.stack_string) == 0
        ):
            return True
        return False

    @staticmethod
    def shift(config: Configuration) -> Configuration:
        """
        Pop first element of buffer on the top of the stack
        Shift: (σ, wi|β, A) ⇒ (σ|wi, β, A)
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf.buffer, conf.stack.pop()
        (['Parsing', 'Is', 'Fun'], 'Hey')
        >>> conf.buffer_string, conf.stack_string.pop()
        (['Parsing', 'Is', 'Fun'], 'Hey')
        >>>
        """

        config.stack.append(config.buffer[0])
        config.stack_string.append(config.buffer_string[0])
        config.shift_buffer()
        return config

    @staticmethod
    def reduce_condition(config: Configuration) -> bool:
        # safeguard against reduce to remove case where we get a isolated word.
        # if len of stack and len of buffer is 1, can't reduce
        # unless the last element in buffer already has an head.
        # or the root has yet to be chosen.
        # No point in reducing to the root. If it already has an head then it is the root choosen with Right arc
        # if no root then one will appear with left arc from the rest of the sentence and it will be poped naturally
        # Note that this only work because arc eager deal only with projective sentence.
        if len(config.stack_string) == 1:
            return False
        """
        if (
            len(config.stack_string) == 1
            and len(config.buffer_string) == 1
            and config.has_root
            and not ArcEagerTransition.has_head(config.buffer_string[0], config.arc)
        ):
            return False
        """
        try:
            wi = config.stack_string[-1]
        except IndexError:
            return False
        return ArcEagerTransition.has_head(wi, config.arc)

    @staticmethod
    def reduce(config: Configuration) -> Configuration:
        """
        Reduce: (σ|wi, β, A) ⇒ (σ, β, A)    HEAD(wi)
        :return:

        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.shift(conf)
        >>> conf = x.reduce(conf)
        >>> conf.buffer, conf.stack.pop()
        (['Is', 'Fun'], 'Hey')
        >>> conf.buffer_string, conf.stack_string.pop()
        (['Is', 'Fun'], 'Hey')
        >>>
        """

        config.stack.pop()
        config.stack_string.pop()
        return config

    @staticmethod
    def right_arc_condition(config: Configuration) -> bool:
        """
        A right arc can be created if there is an element in the stack (futur head)
        and an element in the buffer (futur dependent).
        If the stack if empty and the root has yet to be chosen then the root relation can be formed.
        The stack need to be empty because the root is the first element in the stack (if there is element in the stack
        the root is not on top of the stack)
        """
        if (len(config.stack_string) == 0 and config.has_root) or len(
            config.buffer_string
        ) == 0:
            return False
        if (
            len(config.stack_string) > 0
            or (not config.has_root and len(config.stack_string) == 0)
        ) and len(config.buffer_string) >= 1:
            return True
        return False

    @staticmethod
    def right_arc(config: Configuration) -> Configuration:
        """
        Right-Arc: (σ|wi, wj|β, A) ⇒ (σ|wi|wj, β, A ∪ {wi → wj})
        :return:
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.right_arc(conf)
        >>> conf.buffer, conf.stack, conf.arc[-1].head, conf.arc[-1].dependent
        (['Is', 'Fun'], ['Hey', 'Parsing'], 'Hey', 'Parsing')
        >>> conf.buffer_string, conf.stack_string, conf.arc[-1].head, conf.arc[-1].dependent
        (['Is', 'Fun'], ['Hey', 'Parsing'], 'Hey', 'Parsing')
        >>>
        """
        # if root case
        if not config.has_root and len(config.stack_string) == 0:
            wi = config.root
            # need to make this value not hardcoded
            wi_string = config.root_token
            config.has_root = True
        else:
            wi = config.stack[-1]
            wi_string = config.stack_string[-1]

        wj = config.buffer[0]
        wj_string = config.buffer_string[0]
        config.stack.append(wj)
        config.stack_string.append(wj_string)
        config.arc.append(RightArc(head=wi_string, dependent=wj_string))
        config.shift_buffer()
        return config

    @staticmethod
    def left_arc_condition(config: Configuration) -> bool:
        # safeguard against isolated word in the end
        if len(config.stack_string) == 0 or len(config.buffer_string) == 0:
            return False
        # if its the last element in the buffer has no head and there is only one element left in the stack and the root
        # has already been chosen, then can't use the left arc (no word left in the stack to create an arc after...)
        if (
            len(config.stack_string) == 1
            and len(config.buffer_string) == 1
            and config.has_root
            and not ArcEagerTransition.has_head(config.buffer_string[0], config.arc)
        ):
            return False
        wi = config.stack_string[-1]
        return not ArcEagerTransition.has_head(wi, config.arc)

    @staticmethod
    def left_arc(config: Configuration) -> Configuration:
        """
        Left-Arc: (σ|wi, wj|β, A) ⇒ (σ, wj|β, A ∪ {wi ← wj})    ¬HEAD(wi )
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"])
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.left_arc(conf)
        >>> conf.buffer, conf.stack, conf.arc[-1].head, conf.arc[-1].dependent
        (['Parsing', 'Is', 'Fun'], [], 'Parsing', 'Hey')
        >>> conf.buffer_string, conf.stack_string, conf.arc[-1].head, conf.arc[-1].dependent
        (['Parsing', 'Is', 'Fun'], [], 'Parsing', 'Hey')
        >>>
        :return:
        """

        wi = config.stack.pop()
        wi_string = config.stack_string.pop()
        wj_string = config.buffer_string[0]
        config.arc.append(LeftArc(head=wj_string, dependent=wi_string))
        return config


if __name__ == "__main__":
    import doctest

    doctest.testmod()
