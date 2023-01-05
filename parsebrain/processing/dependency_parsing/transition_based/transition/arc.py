class Arc:
    def __init__(self, head: int, dependent: int):
        self.head = head
        self.dependent = dependent


class LeftArc(Arc):
    def __init__(self, head: int, dependent: int):
        super().__init__(head, dependent)


class RightArc(Arc):
    def __init__(self, head: int, dependent: int):
        super().__init__(head, dependent)
