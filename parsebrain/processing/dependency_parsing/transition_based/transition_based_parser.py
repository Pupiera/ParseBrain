from .transition import Transition
import torch
import speechbrain as sb


# maybe inherit from nn.module ?
# ToDO Maybe the parser should also construct the tree -> how to store them ? (dict: words -> {head, label} )?, only apply on right arc or left arc
# ToDO : Heuristics to connect every words (if every words are shifted for example) or add condition for shift ? Not sure if possible...

class TransitionBasedParser:
    def __init__(self, neural_network, label_neural_network,
                 transition: Transition, features_computer,
                 dynamic_oracle, label_policie):
        self.parser_neural_network = neural_network
        self.label_neural_network = label_neural_network
        self.transition = transition
        self.features_computer = features_computer
        self.dynamic_oracle = dynamic_oracle
        self.label_policie = label_policie
        self.device = None

    def parse(self, config, stage, gold_config=None):
        '''
        Parse one sentence
        TO-DO: Make it batchable
        :param config:
        :return:
        '''
        self.device = config[0].buffer.get_device()

        list_decision_taken = []
        list_decision_score = [[] for _ in range(len(config))]
        dynamic_oracle_decision = [[] for _ in range(len(config))]
        list_label_decision_score = [[] for _ in range(len(config))]
        dynamic_oracle_label_decision = [[] for _ in range(len(config))]
        while any([not self._is_terminal(conf) for conf in config]):
            # Batched decision score for UAS
            features = [self._compute_features(conf) for conf in config]
            features = torch.stack(features)
            decision_score = self._decision_score(features)  # shape = [batch, nb_decision]
            # append it to the correct sequence of decision
            for i, d_score in enumerate(decision_score):
                list_decision_score[i].append(d_score)
            if stage == sb.Stage.TRAIN and gold_config is not None:
                dynamic_oracle_decision = self._get_oracle_move_from_config_tree(config,
                                                                                 gold_config, dynamic_oracle_decision)
                dynamic_oracle_label = None  # output the correct label if the arc exist, output either : -1 or the predicted label if arc does not exist;
            # batched decision apply
            next_config, decision_taken = self._apply_decision(decision_score, config)
            list_decision_taken.append(decision_taken)

            # Based on the decision taken, create feature then compute the label
            label_score = self._compute_label(config, decision_taken)  # Maybe add the decision ?
            for i, l_score in enumerate(label_score):
                list_label_decision_score[i].append(l_score)

            # For the dynamic oracle and the label. If the arc between the two words does not exist
            # keep predicted label (since there is no good label)
            dynamic_oracle_label = self.dynamic_oracle.compute_label(config, gold_config, decision_taken)
            # In the case where the arc does not exist, reinforce current predictions
            if dynamic_oracle_label == -1:
                dynamic_oracle_label = torch.argmax(label_score)

            config = next_config

        list_decision_score = [torch.stack(x) for x in list_decision_score]
        # print(list_decision_score)
        return torch.stack(list_decision_score).to(self.device), list_decision_taken, torch.tensor(
            dynamic_oracle_decision).to(self.device), torch.stack(list_label_decision_score).to(
            self.device), torch.tensor(dynamic_oracle_label_decision)

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
            # unbactched config
            for d in best_decision:
                if self.transition.is_decision_valid(d, config):
                    config = \
                        self.transition.apply_decision(d, config)
                    return config, d
        else:
            # Batched config
            _, best_decision = torch.sort(decision_score, descending=True)
            decision = [[] for i in range(decision_score.shape[0])]
            for i, b in enumerate(best_decision):  # for each element of the batch
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

    def _compute_label(self, config, decision):
        '''
        This function compute the label for each element of the batch.
        Maybe compute for all, but discard the prediction of not applicable one ?
        '''
        # fill mask with 1 when decision taken can't create a label (ie not an arc)
        mask = [1 if d is not self.transition.RIGHT or d is not self.transition.LEFT
                else 0 for d in decision]
        batch_rep = []
        for c, d in zip(config, decision):
            batch_rep.append(self.label_policie.compute_representation(c, d))
        label_score = self.label_neural_network(torch.tensor(batch_rep))
        return label_score, mask
