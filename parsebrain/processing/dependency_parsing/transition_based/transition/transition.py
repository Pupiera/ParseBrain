class Transition:
    def __init__(self):
        pass

    def get_relation_from_decision(self, decision, config):
    raise NotImplementedError("Subclass need to implement this function")

    def is_decision_valid(self, decision, config):
        raise NotImplementedError("Subclass need to implement this function")

    def apply_decision(self, decision, config):
        raise NotImplementedError("Subclass need to implement this function")

    def is_terminal(self, config):
        raise NotImplementedError("Subclass need to implement this function")

    def update_tree(self, decision, config, tree):
        raise NotImplementedError("Subclass need to implement this function")

    def require_label(self, decision):
        raise NotImplementedError("Subclass need to implement this function")
