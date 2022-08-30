class Configuration:
    def __init__(self):
        self.buffer = []
        self.buffer_string = []
        self.stack = []
        self.stack_string = []
        self.arc = []

    def add_features(self, features):
        self.buffer = features

    def shift_buffer(self):
        self.buffer = self.buffer[1:]
        self.buffer_string = self.buffer_string[1:]
