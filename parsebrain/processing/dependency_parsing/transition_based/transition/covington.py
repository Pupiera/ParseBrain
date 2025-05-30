import torch

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
)
from parsebrain.processing.dependency_parsing.transition_based.transition.transition import (
    Transition,
)
from tarjan import tarjan
from parsebrain.processing.dependency_parsing.transition_based.transition.arc import (
    LeftArc,
    RightArc,
)

from itertools import chain


class CovingtonTransition(Transition):
    SHIFT = 0
    LEFT = 1
    RIGHT = 2
    NO_ARC = 3

    def __init__(self):
        super().__init__()

    @classmethod
    def get_transition_dict(cls) -> dict:
        return {
            "SHIFT": cls.SHIFT,
            "NO_ARC": cls.NO_ARC,
            "LEFT": cls.LEFT,
            "RIGHT": cls.RIGHT,
        }

    def get_relation_from_decision(
        self, decision: int, config: Configuration
    ) -> (torch.Tensor, torch.tensor):
        """
        Return the head an the dependent based on the decision taken, used to compute the label
        @param decision:
        @param config:
        @return:
        """
        if len(config.stack) > 0:
            stack = config.stack[-1]
        elif not config.has_root:
            stack = config.root.squeeze()
        else:
            if len(config.buffer) >1:
                device = config.buffer[0].device
                size = config.buffer[0].size()
            elif len(config.stack2) >1:
                device = config.stack2[0].device
                size = config.stack2[0].size()
            stack = torch.zeros(size).to(device)
        if len(config.buffer) > 0:
            buffer = config.buffer[0]
        elif len(config.stack) >0:
            device = config.stack[0].device
            buffer = torch.zeros(config.stack[0].size()).to(device)
        else:
            device = config.stack2[0].device
            buffer = torch.zeros(config.stack2[0].size()).to(device)
        if decision == self.RIGHT:
            return stack, buffer
        elif decision == self.LEFT:
            return buffer, stack
        else:  # if not valid, default = right and will be padded
            return stack, buffer

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
            # ToDo: add default behavior, link to current root.
            #import pudb; pudb.set_trace()
            return self.create_arc_to_root(config)
        

    def create_arc_to_root(self, config):
        # 1. check if blocking element is on buffer or stack
        if len(config.stack_string)>=1 and not CovingtonTransition.has_head(config.stack_string[-1], config.arc):
            w_to_get_head = config.stack_string.pop()
            wi = config.stack.pop()
            if config.has_root:
                w_root = self.get_head_root(config)
                try:
                    config.arc.append(LeftArc(head=w_root, dependent=w_to_get_head))
                except AttributeError as e:
                    import pudb; pudb.set_trace()
                    raise e
            else:
                wi_string = config.root_token
                config.arc.append(RightArc(head=wi_string, dependent=w_to_get_head))
                config.has_root = True
            config.stack2.insert(0, wi)
            config.stack2_string.insert(0, w_to_get_head)
        elif len(config.buffer_string)>=1 and not CovingtonTransition.has_head(config.buffer_string[0], config.arc):
            w_to_get_head = config.buffer_string[0]
            wi = config.buffer[0]
            config.shift_buffer()
            if config.has_root:
                wi_string = self.get_head_root(config)
            else:
                wi_string = config.root_token
                config.has_root = True

            config.arc.append(RightArc(head=wi_string, dependent=w_to_get_head))            
        return config


    def get_head_root(self, config):
        for arc in config.arc:
            if arc.head.position ==0:
                return arc.dependent

    @staticmethod
    def is_terminal(config: Configuration) -> bool:
        if len(config.buffer_string) >1:
            return False
        if len(config.buffer_string) == 0:
            return True
        if CovingtonTransition.has_head(config.buffer_string[0], config.arc):
            # just missing the root transition, will be on the last missing one in the stack
            if not config.has_root and CovingtonTransition.count_stack_no_head(config) ==1:
                return True
            if config.has_root and CovingtonTransition.count_stack_no_head(config) == 0:
                return True
        return False
        #return len(config.buffer_string) == 0

    def update_tree(self, decision: int, config: Configuration, tree: dict) -> dict:
        pass

    def require_label(self, decision: int) -> bool:
        return decision in [self.RIGHT, self.LEFT, -1]

    @staticmethod
    def shift_condition(config):
        if len(config.buffer_string) == 0:
            return False
        if len(config.buffer_string) >= 2:
            return True
        # case where len(config.buffer_string) == 1
        wi = config.buffer_string[0]
        # if last shift, need to have a head (won't be able to get it after the shift...)
        # and CovingtonTransition.all_stack_has_head(config):
        if CovingtonTransition.has_head(wi, config.arc) and CovingtonTransition.all_stack_has_head(config):
            return True
        else:
            return False

    @staticmethod
    def all_stack_has_head(config):
        """
        No point in checking the one in stack 2, they need a shift to be accessible.
        @param config:
        @return:
        """
        for w in config.stack_string:
            if not CovingtonTransition.has_head(w, config.arc):
                return False
        return True
    
    @staticmethod
    def count_stack_no_head(config):
        count = 0
        for w in config.stack_string:
            if not CovingtonTransition.has_head(w, config.arc):
                count+=1
        return count


    @staticmethod
    def shift(config: Configuration):
        """
        Shift: 〈λ1, λ2, j|B, A〉 ⇒ 〈λ1 · λ2|j, [], B, A
        @param config:
        @return:
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import Word
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"], root_embedding=[0])
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
        try:
            # can't be the root here, since it mean the root would be a dependent
            wi_string = config.stack_string[-1]
            wj_string = config.buffer_string[0]
        except IndexError:
            return False
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
        return true if adding this arc would create a cycle.
        @param head:
        @param dependent:
        @param arcs:
        @return: True if the addition of an arc between head and dependent would cause a cycle, False otherwise
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import Word
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],[Word("Hey",1), Word("Parsing",2), Word("Is",3), Word("Fun",4)], root_embedding=[0])
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf = CovingtonTransition.left_arc(conf)
        >>> conf = CovingtonTransition.shift(conf)
        >>> CovingtonTransition.cycle_check(conf.buffer_string[0], conf.stack_string[-1], conf.arc)
        False
        >>> conf = CovingtonTransition.left_arc(conf)
        >>> CovingtonTransition.cycle_check(conf.stack_string[-1],conf.buffer_string[0], conf.arc) # if do right arc
        True
        """
        # no cycle possible, so should be 0 before adding the potential arcs
        A = set([(a.head.position, a.dependent.position) for a in arcs])
        A.add((head.position, dependent.position))
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
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import Word
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"], root_embedding=[0])
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
        if len(config.stack_string) > 0:
            wi_string = config.stack_string[-1]
        elif not config.has_root:
            wi_string = config.root_token
        else:
            return False
        try:
            wj_string = config.buffer_string[0]
        except IndexError:
            return False
        # single head and acyclicity check
        if CovingtonTransition.has_head(
            wj_string, config.arc
        ) or CovingtonTransition.cycle_check(
            head=wi_string, dependent=wj_string, arcs=config.arc
        ):
            return False
        if len(config.buffer_string) ==1 and CovingtonTransition.count_number_word_with_no_head(config) == 1 and not config.has_root:
            return True

        if len(config.buffer_string) ==1 and not CovingtonTransition.has_head(wi_string, config.arc):
            return False

        return True
    
    @staticmethod
    def count_number_word_with_no_head(config):
        return sum([not CovingtonTransition.has_head(x, config.arc) for x in chain(config.buffer_string, config.stack_string, config.stack2_string)])


    @staticmethod
    def right_arc(config):
        """
        Right-Arc: 〈λ1|i, λ2, j|B, A〉 ⇒ 〈λ1, i|λ2, j|B, A ∪ {i → j}〉
        only if not_exist(k) | k → i ∈ A (single-head) and i →∗ j !∈ A (acyclicity)
        @param config:
        @return:
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"], root_embedding=[0])
        >>> conf = CovingtonTransition.shift(conf)
        >>> conf = CovingtonTransition.right_arc(conf)
        >>> conf.stack, conf.stack_string, conf.stack2, conf.stack2_string, conf.buffer, conf.buffer_string, str(conf.arc[0])
        ([], [], ['Hey'], ['Hey'], ['Parsing', 'Is', 'Fun'], ['Parsing', 'Is', 'Fun'], 'arc : Hey -> Parsing')
        """
        wj_string = config.buffer_string[0]
        try:
            wj = config.buffer[0]
        except IndexError as e:
            print(config.buffer_string)
            print(len(config.buffer))
            raise e

        if len(config.stack_string) > 0:
            wi = config.stack.pop()
            wi_string = config.stack_string.pop()
        elif not config.has_root:
            # dont append root to the stack (disable multiple root)
            wi = config.root
            wi_string = config.root_token
            config.arc.append(RightArc(head=wi_string, dependent=wj_string))
            config.has_root = True
            return config
        else:
            raise IndexError("no element in stack and root already chosen.")

        config.stack2.insert(0, wi)
        config.stack2_string.insert(0, wi_string)

        config.arc.append(RightArc(head=wi_string, dependent=wj_string))
        return config

    @staticmethod
    def no_arc_condition(config):
        # Specific case to avoid having nothing in the stack and only one element wihout head in the buffer
        # This situation would be a dead end for the parser (can't shift because the buffer element would never have a head
        # and can't right arc since there is nothing on the stack.
        if (
            len(config.buffer_string) == 1
            and len(config.stack_string) == 1
            and config.has_root
            and not CovingtonTransition.has_head(config.buffer_string[0], config.arc)
        ):
            return False
        # case where we need to force left arc, if the element go to stack2 without a head now, it will never have one since it can't go back into stack1
        if (len(config.buffer_string) == 1 
        and len(config.stack_string)>=1 
        and not CovingtonTransition.has_head(config.stack_string[-1], config.arc)):
            return False

        return len(config.stack_string) >= 1

    @staticmethod
    def no_arc(config):
        """
        〈λ1|i, λ2, B, A〉 ⇒ 〈λ1, i|λ2, B, A〉
        @param config:
        @return:
        >>> from parsebrain.processing.dependency_parsing.transition_based.configuration import Word
        >>> conf = Configuration(["Hey", "Parsing", "Is", "Fun"],["Hey", "Parsing", "Is", "Fun"], root_embedding=[0])
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
