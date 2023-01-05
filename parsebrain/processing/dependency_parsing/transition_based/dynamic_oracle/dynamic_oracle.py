class DynamicOracle:
    def __init__(self, padding_value: int = 1000):
        self.padding_value = padding_value

    def get_oracle_move_from_config_tree(self, configuration, gold_tree):
        raise NotImplementedError
