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
        self.device = config[0].buffer.get_device()

        list_decision_taken = []
        list_decision_score = [[] for _ in range(len(config))]
        dynamic_oracle_decision = [[] for _ in range(len(config))]
        #config.buffer = config.buffer.squeeze()
        while any([not self._is_terminal(conf) for conf in config]) :
            features = [self._compute_features(conf) for conf in config]
            #print(f"features {features}")
            features = torch.stack(features)
            #print(features.shape)
            decision_score = self._decision_score(features)  # shape = [batch, nb_decision]
            for i, d_score in enumerate(decision_score):
                list_decision_score[i].append(d_score)
            if stage == sb.Stage.TRAIN and gold_config is not None:
                dynamic_oracle_decision = self._get_oracle_move_from_config_tree(config, gold_config, dynamic_oracle_decision)
            config, decision_taken = self._apply_decision(decision_score, config)
            #list_decision_score.append(decision_score)
            list_decision_taken.append(decision_taken)
        list_decision_score = [torch.stack(x) for x in list_decision_score]
        #print(list_decision_score)
        return torch.stack(list_decision_score).to(self.device), list_decision_taken,  torch.tensor(dynamic_oracle_decision).to(self.device)

    def _decision_score(self, x):
        return self.parser_neural_network(x)


    def _get_oracle_move_from_config_tree(self, config, gold_config, dynamic_oracle_decision):
        for i in range(len(config)):
            decision = self.dynamic_oracle.get_oracle_move_from_config_tree(config[i], gold_config[i])
            dynamic_oracle_decision[i].append(decision)
        return dynamic_oracle_decision

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
        if decision_score.shape[0] == 1:
            for d in best_decision:
                if self.transition.is_decision_valid(d, config):
                    config = \
                        self.transition.apply_decision(d, config)
                    return config, d
        else:
            _, best_decision = torch.sort(decision_score, descending=True)
            decision = [[] for i in range(decision_score.shape[0])]
            for i, b in enumerate(best_decision): # for each element of the batch
                for d in b:
                    if self.transition.is_decision_valid(d, config[i]):
                        config[i] = \
                            self.transition.apply_decision(d, config[i])
                        decision[i].append(d)
                        break

            return config, decision

    def _is_terminal(self, config):
        '''
        Return whether the configuration is terminal or not.
        :param config:
        :return:
        '''
        return self.transition.is_terminal(config)
