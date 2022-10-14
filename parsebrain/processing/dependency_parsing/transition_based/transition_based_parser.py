import speechbrain as sb
import torch

from .transition import Transition


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
        :return: A dictionary with the following key:
        "decision_score": tensor log prob of each parsing decision (batch, seq, decision)
        "decision_taken": tensor of each decision taken by the system after verification of applicability (batch, seq)
        "oracle_parsing": tensor of best possible decision the model could have taken at this step
        "label_score": tensor log prob of each label decision (batch, seq, label)
        "oracle_label": tensor of best possible label the model should have taken at this step
        "mask_label" : mask the label taken when it is not possible to take a label with the current parsing decision.
        '''
        self.device = config[0].buffer.get_device()

        list_decision_taken = [[] for _ in range(len(config))]
        list_decision_score = [[] for _ in range(len(config))]
        dynamic_oracle_decision = [[] for _ in range(len(config))]
        list_label_decision_score = [[] for _ in range(len(config))]
        dynamic_oracle_label_decision = [[] for _ in range(len(config))]
        list_mask_label = [[] for _ in range(len(config))]
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
            # batched find best decision
            decision_taken = self._get_best_valid_decision(decision_score, config)
            for i, d in enumerate(
                    decision_taken):  # ToDo: find a way to do this kind of operation with matrice operation. (to remove loops)
                list_decision_taken[i].append(d)

            # Based on the decision taken, create feature then compute the label
            label_score, mask_label = self._compute_label(config, decision_taken)
            for i, l_score in enumerate(label_score):
                list_label_decision_score[i].append(l_score)

                # For the dynamic oracle and the label. If the arc between the two words does not exist
                # keep predicted label (since there is no good label)
                dynamic_oracle_label = self.dynamic_oracle.compute_label(config[i], gold_config[i], decision_taken[i])
                # In the case where the arc does not exist, reinforce current predictions
                if dynamic_oracle_label == -1:
                    dynamic_oracle_label = torch.argmax(l_score)
                dynamic_oracle_label_decision[i].append(dynamic_oracle_label)
                list_mask_label[i].append(mask_label[i])
            # batched update config for each decision now that all prediction has been done
            config = self._apply_decision(decision_taken, config)

        list_decision_score = [torch.stack(x) for x in list_decision_score]
        list_decision_taken = [torch.stack(x) for x in list_decision_taken]
        list_label_decision_score = [torch.stack(x) for x in list_label_decision_score]

        return {"decision_score": torch.stack(list_decision_score).to(self.device),
                "decision_taken": torch.stack(list_decision_taken),
                "oracle_parsing": torch.tensor(dynamic_oracle_decision).to(self.device),
                "label_score": torch.stack(list_label_decision_score).to(self.device),
                "oracle_label": torch.tensor(dynamic_oracle_label_decision).to(self.device),
                "mask_label": torch.tensor(list_mask_label).to(self.device) == 1}

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

    def _apply_decision(self, decision, config):
        '''
        Function
        :param decision_score:
        :param config:
        :return:
        '''

        for i, d in enumerate(decision):
            config[i] = self.transition.apply_decision(d, config[i])
        return config

    def _get_best_valid_decision(self, decision_score, config):
            # Batched config
            _, best_decision = torch.sort(decision_score, descending=True)
            decision = []
            for i, b in enumerate(best_decision):  # for each element of the batch
                best_decision = torch.tensor(-1, device=self.device)  # no decision valid
                for d in b:
                    if self.transition.is_decision_valid(d, config[i]):
                        best_decision = d
                        break
                decision.append(best_decision)
            return decision

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
        mask = [1 if (d == self.transition.RIGHT or d == self.transition.LEFT)
                else 0 for d in decision]
        batch_rep = []
        for c, d in zip(config, decision):
            batch_rep.append(self.label_policie.compute_representation(c, d, self.transition))
        label_score = self.label_neural_network(torch.stack(batch_rep))
        return label_score, mask
