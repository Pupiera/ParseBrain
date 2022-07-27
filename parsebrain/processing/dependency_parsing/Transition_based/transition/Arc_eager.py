from Transition import Transition
from Arc import Right_Arc, Left_Arc
from Configuration import Configuration


# use docTest

class ArcEagerTransition(Transition):

    def __init__(self):
        super().__init__()
        self.SHIFT = 0
        self.REDUCE = 1
        self.LEFT = 2
        self.RIGHT = 3

    def get_transition_dict(self):
        return {"SHIFT": self.SHIFT,
                "REDUCE": self.REDUCE,
                "LEFT": self.LEFT,
                "RIGHT": self.RIGHT}

    def apply_decision(self, decision, config):
        if decision == self.SHIFT:
            return self.shift(config)
        elif decision == self.REDUCE:
            return self.reduce(config)
        elif decision == self.LEFT:
            return self.left_arc(config)
        elif decision == self.RIGHT:
            return self.right_arc(config)
        else:
            raise ValueError(f"Decision {decision} out of scope for arc-eager transition")

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
        return not (len(config.buffer) == 0 and len(config.stack) == 0)

    @staticmethod
    def has_head(wi, arc):
        """
        The condition HEAD(wi ) is true in a configuration (σ, β, A) if A contains an arc wk → wi
        Check if wi already has a HEAD.
        :param wi:
        :param arc:
        :return:
        """
        for a in arc:
            if a.head == wi:
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
            >>> from collections import deque
            >>> conf = Configuration()
            >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
            >>> x = ArcEagerTransition()
            >>> conf = x.shift(conf)
            >>> conf.buffer, conf.stack.pop()
            (['Parsing', 'Is', 'Fun'], 'Hey')
            >>>

        :return:
        """
        config.stack.append(config.buffer[0])
        config.shift_buffer()
        return config

    def reduce_condition(self, config):
        if len(config.stack[0]) == 0:
            return False
        wi = config.stack[0]
        return self.has_head(wi, config.arc)

    @staticmethod
    def reduce(config):
        """
        Reduce: (σ|wi, β, A) ⇒ (σ, β, A)    HEAD(wi)
        :return:

        >>> conf = Configuration()
        >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.shift(conf)
        >>> conf = x.reduce(conf)
        >>> conf.buffer, conf.stack.pop()
        (['Is', 'Fun'], 'Hey')
        >>>

        """
        config.stack.pop()
        return config

    @staticmethod
    def right_arc_condition(config):
        return True

    @staticmethod
    def right_arc(config):
        """
        Right-Arc: (σ|wi, wj|β, A) ⇒ (σ|wi|wj, β, A ∪ {wi → wj})
        :return:
        >>> conf = Configuration()
        >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.right_arc(conf)
        >>> conf.buffer, conf.stack, conf.arc[-1].head, conf.arc[-1].dependent
        (['Is', 'Fun'], deque(['Hey', 'Parsing']), 'Hey', 'Parsing')
        >>>
        """
        wi = config.stack[0]
        wj = config.buffer[0]
        config.stack.append(wj)
        config.arc.append(Right_Arc(head=wi, dependent=wj))
        config.shift_buffer()
        return config

    def left_arc_condition(self, config):
        wi = config.stack[0]
        return not self.has_head(wi, config.arc)

    @staticmethod
    def left_arc(config):
        """
        Left-Arc: (σ|wi, wj|β, A) ⇒ (σ, wj|β, A ∪ {wi ← wj})    ¬HEAD(wi )
        >>> from collections import deque
        >>> conf = Configuration()
        >>> conf.buffer = ["Hey", "Parsing", "Is", "Fun"]
        >>> x = ArcEagerTransition()
        >>> conf = x.shift(conf)
        >>> conf = x.left_arc(conf)
        >>> conf.buffer, conf.stack, conf.arc[-1].head, conf.arc[-1].dependent
        (['Parsing', 'Is', 'Fun'], deque([]), 'Parsing', 'Hey')
        >>>
        :return:
        """
        wi = config.stack.pop()
        wj = config.buffer[0]
        config.arc.append(Left_Arc(head=wj, dependent=wi))
        return config


if __name__ == "__main__":
    import doctest

    doctest.testmod()
