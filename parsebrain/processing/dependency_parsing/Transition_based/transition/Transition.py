class Transition:
    def __init__(self):
        pass

    def is_decision_valid(self, decision, config):
        raise NotImplementedError

    def apply_decision(self, decision, config):
        raise NotImplementedError

    def is_terminal(self, config):
        raise NotImplementedError
