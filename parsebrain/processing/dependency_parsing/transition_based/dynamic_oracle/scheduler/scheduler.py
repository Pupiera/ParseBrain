class Scheduler:
    def update_rate(self, rate: float) -> float:
        raise NotImplementedError


class LinearScheduler(Scheduler):
    def __init__(self, growth_rate: float):
        self.growth_rate = growth_rate

    def update_rate(self, rate: float) -> float:
        new_rate = rate + self.growth_rate
        if new_rate > 1:
            return 1.0
        else:
            return new_rate
