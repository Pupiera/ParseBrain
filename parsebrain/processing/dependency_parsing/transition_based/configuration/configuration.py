from collections import defaultdict


class Configuration:
    def __init__(self, features, features_string):
        self.buffer = features
        self.buffer_string = features_string
        self.stack = []
        self.stack_string = []
        self.arc = []

    def add_features(self, features, features_string):
        self.buffer = features
        self.buffer_string = features_string

    def shift_buffer(self):
        self.buffer = self.buffer[1:]
        self.buffer_string = self.buffer_string[1:]


class GoldConfiguration:
    '''
    This class contains the information about the gold data.
    It only uses the position of the word in the sentence to identify them.
    This remove the ambiguity if there is multiple occurrence of the same word.
    '''

    def __init__(self):
        self.heads = {}  # the head of a given word
        self.deps = defaultdict(lambda: [])  # the list of dependent of a given word (can be empty)


class Word:
    '''
    Contains the word (string)
    '''

    def __init__(self, word, position):
        self.word = word
        self.position = position
