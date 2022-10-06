from .transition import Transition
from .arc import Right_Arc, Left_Arc
from parsebrain.processing.dependency_parsing.transition_based.configuration import Configuration


class ArcEagerTransition(Transition):

    def __init__(self):
        super().__init__()
        self.SHIFT = 0
        self.LEFT = 1
        self.RIGHT = 2
        self.REDUCE = 3


    def get_transition_dict(self):
        return {"SHIFT": self.SHIFT,
                "REDUCE": self.REDUCE,
                "LEFT": self.LEFT,
                "RIGHT": self.RIGHT}

    def get_relation_from_decision(self, decision, config):
        if decision == self.RIGHT:
            return config.stack[0], config.buffer[0]
        elif decision == self.LEFT:
            return config.buffer[0], config.stack[0]
        else: # if not valid, default = right
            return config.stack[0], config.buffer[0]

    def apply_decision(self, decision, config):
        '''
        apply the given decision
        In case where the state is terminal do nothing (Padding)
        '''
        if self.is_terminal(config):
            return config

        if decision == self.SHIFT:
            return self.shift(config)
        elif decision == self.REDUCE:
            return self.reduce(config)
        elif decision == self.LEFT:
            return self.left_arc(config)
        elif decision == self.RIGHT:
            return self.right_arc(config)
        else:
            raise ValueError(f"Decision number ({decision}) is out of scope for arc-eager transition")

    def is_decision_valid(self, decision, config):
        if decision == self.SHIFT:
            return self.shift_condition(config)
        elif decision == self.REDUCE:
            return self.reduce_condition(config)
        elif decision == self.LEFT:
            return self.left_arc_condition(config)
        elif decision == self.RIGHT:
            return self.right_arc_condition(config)
        else:
            raise ValueError(f"Decision {decision} out of scope for arc-eager transition")

    def is_terminal(self, config):
        """
        Condition is terminal if buffer is empty
        (can't create any new arc if everything is on the stack)
        """
        return len(config.buffer) == 0

    @staticmethod
    def has_head(wi, arc):
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
    def shift_condition(config):
        """
        Condition to fullfill for shift to be available
        :return:
        """
        return len(config.buffer) != 0

    @staticmethod
    def shift(config):
        """
        Pop first element of buffer on the top of the stack
        Shift: (σ, wi|β, A) ⇒ (σ|wi, β, A)
        >>> conf = Configuration()
        >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
        >>> conf.buffer_string = ["Hey", "Parsing", "Is", "Fun"]
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
    def reduce_condition(config):
        try:
            if len(config.stack[0]) == 0:
                return False
            wi = config.stack_string[-1]
        except IndexError:
            return False
        return ArcEagerTransition.has_head(wi, config.arc)

    @staticmethod
    def reduce(config):
        """
        Reduce: (σ|wi, β, A) ⇒ (σ, β, A)    HEAD(wi)
        :return:

        >>> conf = Configuration()
        >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
        >>> conf.buffer_string = ["Hey", "Parsing", "Is", "Fun"]
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
    def right_arc_condition(config):
        if len(config.stack) > 0 and len(config.buffer) > 0:
            return True
        return False

    @staticmethod
    def right_arc(config):
        """
        Right-Arc: (σ|wi, wj|β, A) ⇒ (σ|wi|wj, β, A ∪ {wi → wj})
        :return:
        >>> conf = Configuration()
        >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
        >>> conf.buffer_string = ["Hey", "Parsing", "Is", "Fun"]
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.right_arc(conf)
        >>> conf.buffer, conf.stack, conf.arc[-1].head, conf.arc[-1].dependent
        (['Is', 'Fun'], ['Hey', 'Parsing'], 'Hey', 'Parsing')
        >>> conf.buffer_string, conf.stack_string, conf.arc[-1].head, conf.arc[-1].dependent
        (['Is', 'Fun'], ['Hey', 'Parsing'], 'Hey', 'Parsing')
        >>>
        """

        wi = config.stack[0]
        wi_string = config.stack_string[0]
        wj = config.buffer[0]
        wj_string = config.buffer_string[0]
        config.stack.append(wj)
        config.stack_string.append(wj_string)
        config.arc.append(Right_Arc(head=wi_string, dependent=wj_string))
        config.shift_buffer()
        return config

    @staticmethod
    def left_arc_condition(config):
        try:
            wi = config.stack_string[-1]
        except IndexError:
            return False
        return not ArcEagerTransition.has_head(wi, config.arc)

    @staticmethod
    def left_arc(config):
        """
        Left-Arc: (σ|wi, wj|β, A) ⇒ (σ, wj|β, A ∪ {wi ← wj})    ¬HEAD(wi )
        >>> conf = Configuration()
        >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
        >>> conf.buffer_string = ["Hey", "Parsing", "Is", "Fun"]
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.left_arc(conf)
        >>> conf.buffer, conf.stack, conf.arc[-1].head, conf.arc[-1].dependent
        (['Parsing', 'Is', 'Fun'], [], 'Parsing', 'Hey')
        >>>
        >>> conf.buffer_string, conf.stack_string, conf.arc[-1].head, conf.arc[-1].dependent
        (['Parsing', 'Is', 'Fun'], [], 'Parsing', 'Hey')
        >>>
        :return:
        """

        wi = config.stack.pop()
        wi_string = config.stack_string.pop()
        wj = config.buffer[0]
        wj_string = config.buffer_string[0]
        config.arc.append(Left_Arc(head=wj_string, dependent=wi_string))
        return config


if __name__ == "__main__":
    import doctest
    doctest.testmod()
