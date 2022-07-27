class Configuration:
    def __init__(self):
        self.buffer = []
        self.stack = []
        self.arc = []

    def add_features(self, features):
        self.buffer = features

    def shift_buffer(self):
        self.buffer = self.buffer[1:]
