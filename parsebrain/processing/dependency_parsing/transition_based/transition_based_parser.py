from .transition import Transition
import torch


# maybe inherit from nn.module ?

class TransitionBasedParser:
    def __init__(self, neural_network, transition: Transition, features_computer, dynamic_oracle):
        self.config = None
        self.parser_neural_network = neural_network
        self.transition = transition
        self.features_computer = features_computer
        self.dynamic_oracle = dynamic_oracle

    def parse(self, features, config):
        '''
        Parse one sentence
        TO-DO: Make it batchable
        :param features: embedding from features extractor
        :param config:
        :return:
        '''
        self.config = config
        self.config.add_features(features)
        # To do: Do this at batch level.
        while not self.config.is_terminal():
            features = self.compute_features()  # is it really needed ?
            decision_score = self.decision_score(features)  # size = number of transitions
            self.config.apply_decision(decision_score)

    def decision_score(self, x):
        return self.parser_neural_network(x)

    def compute_features(self):
        '''
        strategy to compute features
        e.g : concat embedding of first buffer word to the embedding of the first 3 element of the stack
        :return:
        '''
        return self.features_computer.compute_feature(self.config)

    def apply_decision(self, decision_score):
        '''
        Function
        :param decision_score:
        :return:
        '''
        # order decision from most likely to the least likely
        _, best_decision = torch.sort(decision_score, descending=True)
        # Apply first best applicable decision
        for d in best_decision:
            if self.transition.is_decision_valid(d, self.config):
                self.config = \
                    self.transition.apply_decision(d, self.config)
                return

    def is_terminal(self):
        '''
        Return whether the configuration is terminal or not.
        :return:
        '''
        return self.transition.is_terminal(self.config)
