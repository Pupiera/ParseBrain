from .transition import Transition
import torch


# maybe inherit from nn.module ?

class TransitionBasedParser:
    def __init__(self, neural_network, transition: Transition, features_computer, dynamic_oracle):
        self.parser_neural_network = neural_network
        self.transition = transition
        self.features_computer = features_computer
        self.dynamic_oracle = dynamic_oracle

    def parse(self, config):
        '''
        Parse one sentence
        TO-DO: Make it batchable
        :param features: embedding from features extractor
        :param config:
        :return:
        '''
        # To do: Do this at batch level.
        list_decision_taken = []
        while not self.is_terminal(config):
            features = self.compute_features(config)  # is it really needed ?
            decision_score = self.decision_score(features)  # size = number of transitions
            config, decision_taken = self.apply_decision(decision_score, config)

    def decision_score(self, x):
        return self.parser_neural_network(x)

    def compute_features(self, config):
        '''
        strategy to compute features
        e.g : concat embedding of first buffer word to the embedding of the first 3 element of the stack
        :param config:
        :return:
        '''
        return self.features_computer.compute_feature(config)

    def apply_decision(self, decision_score, config):
        '''
        Function
        :param decision_score:
        :param config:
        :return:
        '''
        # order decision from most likely to the least likely
        _, best_decision = torch.sort(decision_score, descending=True)
        # Apply first best applicable decision
        for d in best_decision:
            if self.transition.is_decision_valid(d, config):
                config = \
                    self.transition.apply_decision(d, config)
                return config, d

    def is_terminal(self, config):
        '''
        Return whether the configuration is terminal or not.
        :param config:
        :return:
        '''
        return self.transition.is_terminal(config)
