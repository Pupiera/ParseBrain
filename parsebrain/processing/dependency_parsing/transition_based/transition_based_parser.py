from .transition import Transition
import torch
import speechbrain as sb

# maybe inherit from nn.module ?
# ToDo Need to find a clean way to introduce the dynamic oracle. Extremely important to get good supervision


class TransitionBasedParser:
    def __init__(self, neural_network, transition: Transition, features_computer, dynamic_oracle):
        self.parser_neural_network = neural_network
        self.transition = transition
        self.features_computer = features_computer
        self.dynamic_oracle = dynamic_oracle
        self.device = None

    def parse(self, config, stage, gold_config=None):
        '''
        Parse one sentence
        TO-DO: Make it batchable
        :param config:
        :return:
        '''
        # To do: Do this at batch level. or not ?
        self.device = config.buffer.get_device()
        list_decision_taken = []
        dynamic_oracle_decision = []
        config.buffer = config.buffer.squeeze()
        while not self._is_terminal(config):
            features = self._compute_features(config)
            decision_score = self._decision_score(features)  # size = number of transitions
            if stage == sb.Stage.TRAIN and gold_config is not None:
                dynamic_oracle_decision.append(
                    self.dynamic_oracle.get_oracle_move_from_config_tree(config, gold_config ))
            config, decision_taken = self._apply_decision(decision_score, config)
            list_decision_taken.append(decision_taken)
        return list_decision_taken, dynamic_oracle_decision

    def _decision_score(self, x):
        x = x.unsqueeze(0) #simulate batch for the moment
        return self.parser_neural_network(x)

    def _compute_features(self, config):
        '''
        strategy to compute features
        e.g : concat embedding of first buffer word to the embedding of the first 3 element of the stack
        :param config:
        :return:
        '''
        return self.features_computer.compute_feature(config, self.device)

    def _apply_decision(self, decision_score, config):
        '''
        Function
        :param decision_score:
        :param config:
        :return:
        '''
        # order decision from most likely to the least likely
        _, best_decision = torch.sort(decision_score, descending=True)
        best_decision = best_decision.squeeze()
        # Apply first best applicable decision
        for d in best_decision:
            if self.transition.is_decision_valid(d, config):
                config = \
                    self.transition.apply_decision(d, config)
                return config, d

    def _is_terminal(self, config):
        '''
        Return whether the configuration is terminal or not.
        :param config:
        :return:
        '''
        return self.transition.is_terminal(config)
