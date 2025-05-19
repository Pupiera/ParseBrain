class Arc:
    def __init__(self, head, dependent):
        if head is None or dependent is None:
            raise AttributeError("Head or dependent is None when creating arc")

        self.head = head
        self.dependent = dependent

    def __str__(self):
        return f"arc : {self.head} -> {self.dependent}"


class LeftArc(Arc):
    def __init__(self, head, dependent):
        super().__init__(head, dependent)


class RightArc(Arc):
    def __init__(self, head, dependent):
        super().__init__(head, dependent)
