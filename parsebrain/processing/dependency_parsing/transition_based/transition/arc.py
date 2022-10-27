class Arc:
    def __init__(self, head: int, dependent: int):
        self.head = head
        self.dependent = dependent


class Left_Arc(Arc):
    def __init__(self, head: int, dependent: int):
        super().__init__(head, dependent)


class Right_Arc(Arc):
    def __init__(self, head: int, dependent: int):
        super().__init__(head, dependent)
