from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
)


class Transition:
    def __init__(self):
        pass

    def get_relation_from_decision(self, decision: int, config: Configuration):
        raise NotImplementedError("Subclass need to implement this function")

    def is_decision_valid(self, decision: int, config: Configuration):
        raise NotImplementedError("Subclass need to implement this function")

    def apply_decision(self, decision: int, config: Configuration):
    raise NotImplementedError("Subclass need to implement this function")

    def is_terminal(self, config: Configuration):
    raise NotImplementedError("Subclass need to implement this function")

    def update_tree(self, decision: int, config: Configuration, tree: dict):
    raise NotImplementedError("Subclass need to implement this function")

    def require_label(self, decision: int):
    raise NotImplementedError("Subclass need to implement this function")
