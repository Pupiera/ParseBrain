from collections import deque


class Configuration:
    def __init__(self):
        self.buffer = []
        self.stack = deque()
        self.arc = []

    def add_features(self, features):
        self.buffer = features

    def shift_buffer(self):
        self.buffer = self.buffer[1:]
