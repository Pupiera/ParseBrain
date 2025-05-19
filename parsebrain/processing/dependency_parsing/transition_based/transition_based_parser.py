from typing import List

import speechbrain as sb
import torch

from .configuration import Configuration, GoldConfiguration
from .configuration_features_computer import ConfigurationFeaturesComputer
from .dynamic_oracle.dynamic_oracle import DynamicOracle
from .label.label_policie import LabelPolicie
from .static_oracle.static_oracle import StaticOracle
from .transition import Transition


# maybe inherit from nn.module ?
# ToDO Maybe the parser should also construct the tree -> how to store them ? (dict: words -> {head, label} )?, only apply on right arc or left arc
# ToDO : Heuristics to connect every words (if every words are shifted for example) or add condition for shift ? Not sure if possible...


class TransitionBasedParser:
    def __init__(
        self,
        neural_network: torch.nn.Module,
        transition: Transition,
        features_computer: ConfigurationFeaturesComputer,
        label_policie: LabelPolicie,
        label_neural_network: torch.nn.Module = None,
        decision_head: torch.nn.Module = None,
        label_head: torch.nn.Module = None,
        dynamic_oracle: DynamicOracle = None,
        exploration_rate: float = 0.5,
        static_oracle: StaticOracle = None,
        oracle_padding_value: int = -100,
    ):
        if dynamic_oracle is None and static_oracle is None:
            raise ValueError(
                "Dynamic alignment_oracle or static alignment_oracle need to be specified for transition based parser"
            )
        if label_neural_network is None and label_head is None:
            raise ValueError(
                "A neural network need to be specified for the label. Either use label_neural_network"
                "for a full disjoint network or use (label_head + decision_head) for a joint network using the same representation"
                "than the decision one"
            )
        self.parser_neural_network = neural_network
        self.label_neural_network = label_neural_network
        self.decision_head = decision_head
        self.label_head = label_head
        self.transition = transition
        self.features_computer = features_computer
        self.dynamic_oracle = dynamic_oracle
        self.label_policie = label_policie
        self.device = None
        self.static_oracle = static_oracle
        self.exploration_rate = exploration_rate
        # padding value of alignment_oracle parsing and alignment_oracle label,
        # -1 cause NaN with NLLL (numerical instability) and NaN * 0 = NaN so mask do not work...
        # -100 is the default ignore_value in torch.nn.functional.nll_loss
        self.oracle_padding_value = oracle_padding_value
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)

    def get_exploration_rate(self):
        return self.exploration_rate

    def _mask_by_valid(self, decisions_logits, config):
        """
        This function set the prob of invalid decision to 0 by modifying the logits.
        This allow better learning, because the system is focused on the decision it can take.
        decisions_logits : The logits of the decision before the softmax
        config : the current state of all the coniguration
        """
        for i in range(len(config)):
            d1 = decisions_logits[i]
            c = config[i]
            for key, item in self.transition.get_transition_dict().items():
                if not self.transition.is_decision_valid(item, c):
                    d1[item] = -(10**10)
            decisions_logits[i] = d1
        return decisions_logits

    def update_exploration_rate(self, rate):
        self.exploration_rate = rate

    def parse(
        self,
        config: Configuration,
        stage: sb.Stage,
        gold_config: GoldConfiguration = None,
        static: bool = False,
    ) -> dict():
        """
        Parse one sentence
        :param config: the configuration composed of the empty stack and the buffer with embedding
        :param stage: the current stage (TRAIN, DEV, TEST)
        :param gold_config: the true value of the tree, used for computing dynamic alignment_oracle and get supervision
        :param static: use dynamic alignment_oracle in a static way if needed. ie: for warming up the model in the first few epoch.

        :return: A dictionary with the following key:
        "decision_score": tensor log prob of each parsing decision (batch, seq, decision)
        "decision_taken": tensor of each decision taken by the system after verification of applicability (batch, seq)
        "oracle_parsing": tensor of best possible decision the model could have taken at this step
        "label_score": tensor log prob of each label decision (batch, seq, label)
        "oracle_label": tensor of best possible label the model should have taken at this step
        "mask_label" : mask the label taken when it is not possible to take a label with the current parsing decision.
        """
        self.stage = stage
        # toDo: make this device attribution cleaner.
        self.device = config[0].buffer.device
        self.static = static

        list_decision_taken = [[] for _ in range(len(config))]
        list_decision_score = [[] for _ in range(len(config))]
        oracle_decision = [[] for _ in range(len(config))]
        list_label_decision_score = [[] for _ in range(len(config))]
        dynamic_oracle_label_decision = [[] for _ in range(len(config))]
        list_mask_label = [[] for _ in range(len(config))]
        parsed_tree = [{} for _ in range(len(config))]
        oracle_len = [-1 for _ in range(len(config))]
        # we compute the supervision with static alignment_oracle
        step_seq = 0
        self.using_dynamic_as_static = (static and self.static_oracle is None) or not static

        if (
            sb.Stage.TRAIN == stage
            and static
            and not gold_config is None
            and self.static_oracle != None
        ):
            oracle_decision = self._get_static_supervision(static, gold_config)

        # PARSER LOOP
        while any([not self._is_terminal(conf) for conf in config]):
            # UAS
            decision_score, decision_taken, oracle_decision = self._step_UAS(config, gold_config, oracle_decision, oracle_len, step_seq)
            list_decision_score, list_decision_taken = self._update_UAS_list(decision_score, decision_taken, list_decision_score, list_decision_taken)
            #LAS
            label_score,  mask_label = self._step_LAS(config, decision_taken)            
            (list_label_decision_score, 
             dynamic_oracle_label_decision, 
             list_mask_label) = self._update_LAS_list(config, gold_config, label_score, 
                    decision_taken, list_label_decision_score, 
                    dynamic_oracle_label_decision, mask_label, list_mask_label)
            #UPDATE CONFIG
            config = self._apply_decision(decision_taken, config)
            parsed_tree = self._update_tree_v2(
                decision_taken, label_score, config, parsed_tree, mask_label
            )
            step_seq += 1

        # ----------------------------list to tensor -----------------------------------------
        #if not static or self.using_dynamic_as_static:
        if not torch.is_tensor(oracle_decision):
            oracle_decision = torch.tensor(oracle_decision)
        list_decision_score = [torch.stack(x) for x in list_decision_score]
        list_decision_taken = [torch.stack(x) for x in list_decision_taken]
        list_label_decision_score = [torch.stack(x) for x in list_label_decision_score]
        oracle_label_len = torch.tensor([len(d) for d in dynamic_oracle_label_decision])
        dynamic_oracle_label_decision = [
            torch.tensor(d) for d in dynamic_oracle_label_decision
        ]
        list_mask_label = [torch.tensor(m) for m in list_mask_label]
        oracle_len = torch.tensor(
            [
                oracle_len[i] if oracle_len[i] != -1 else step_seq
                for i in range(len(oracle_len))
            ]
        )

        return {
            "parse_log_prob": torch.stack(list_decision_score).to(self.device),
            "decision_taken": torch.stack(list_decision_taken),
            "oracle_parsing": oracle_decision.to(self.device),
            "label_log_prob": torch.nn.utils.rnn.pad_sequence(
                list_label_decision_score,
                batch_first=True,
            ).to(self.device),
            "oracle_label": torch.nn.utils.rnn.pad_sequence(
                dynamic_oracle_label_decision,
                batch_first=True,
                padding_value=self.oracle_padding_value,
            ).to(self.device),
            "mask_label": torch.nn.utils.rnn.pad_sequence(
                list_mask_label, batch_first=True, padding_value=-1
            ).to(self.device)
            == 1,
            "parsed_tree": parsed_tree,
            "oracle_parse_len": (oracle_len / torch.max(oracle_len)).to(self.device),
            "oracle_label_len": (oracle_label_len / torch.max(oracle_label_len)).to(
                self.device
            ),
        }



    def _step_UAS(self, config: Configuration, gold_config, oracle_decision, oracle_len, step_seq):
        # Batched decision score for UAS
        # todo: need to rename this
        # this function compute the embedding to use in input. Feature should be the encoding of these
        features = self._compute_features(config)
        # case for shared features uses
        # -----------------ACTIONS------------------------
        decision_score = self._decision_score(features)
        #decision_score = self._mask_by_valid(decision_score, config)
        decision_score = self.softmax(decision_score)

        if (
            (
                (sb.Stage.TRAIN == self.stage and self.using_dynamic_as_static)
                or sb.Stage.VALID == self.stage
            )
            and len(gold_config) >0
            and self.dynamic_oracle is not None
        ):
            oracle_decision = self._get_oracle_move_from_config_tree(
                config, gold_config, oracle_decision
            )

        # batched find best decision
        decision_taken = self.get_decision(
            decision_score, config, oracle_decision, oracle_len, step_seq, self.static
        )

        return decision_score, decision_taken, oracle_decision

    def _update_UAS_list(self, decision_score, decision_taken, list_decision_score, list_decision_taken):
        for i, d_score in enumerate(decision_score):
            list_decision_score[i].append(d_score)
        for i, d in enumerate(decision_taken):
            list_decision_taken[i].append(d)
        return list_decision_score, list_decision_taken


    def _step_LAS(self, config: Configuration, decision_taken):
        # Based on the decision taken, create feature then compute the label
        if self.label_neural_network is not None:
            label_score, mask_label = self._compute_label_not_shared(
                config, decision_taken
            )
        else:
            label_score, mask_label = self._compute_label_shared(
                self.shared_features, decision_taken
            )
        return label_score, mask_label

    def _update_LAS_list(self, config, gold_config, label_score, 
            decision_taken, list_label_decision_score, 
            dynamic_oracle_label_decision, mask_label, 
            list_mask_label):
        for i in range(len(config)):          
            if mask_label[i] == 1:
                list_label_decision_score[i].append(label_score[i])
                if len(gold_config) >0:
                    dynamic_oracle_label = self.dynamic_oracle.compute_label(
                        config[i], gold_config[i], decision_taken[i]
                    )
                    # if this a error in UAS (couple does not exist), reinforce current predictions.
                    if dynamic_oracle_label == self.oracle_padding_value:
                        dynamic_oracle_label = torch.argmax(label_score[i])
                    dynamic_oracle_label_decision[i].append(dynamic_oracle_label)
            list_mask_label[i].append(mask_label[i])
        return list_label_decision_score, dynamic_oracle_label_decision, list_mask_label


    def _decision_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        x :  features to score
        We take the last element of the sequence in case we use RNN.
        """
        if self.decision_head is None:
            x = self.parser_neural_network(x)
            # case with rnn (select last prediction of RNN)
            # we can't take hidden if it's not divided in two module
            if len(x.shape) >= 3:
                x = x[:, -1, :]
        else:
            x = self.parser_neural_network(x)
            #rnn case take hidden
            if len(x) == 2:
                hidden = x[1]
                if isinstance(hidden, tuple):
                    hidden = hidden[0]

                #shape [nb_hidden, batch, dim), mean by nb_hidden (nb_layers * bidirectionality)
                x = torch.mean(hidden, dim=0)#.unsqueeze(1)
            x = self.decision_head(x)
        return x

    def _update_tree(
        self,
        decision_taken: List[int],
        label_score: torch.Tensor,
        config: Configuration,
        parsed_tree: List[dict],
        mask_label: List[int],
    ) -> List[dict]:
        # toDo: maybe add with torch.no_grad()
        for i, d in enumerate(decision_taken):
            last_key = self.transition.update_tree(d, config[i], parsed_tree[i])
            if mask_label[i] == 1:
                parsed_tree[i][last_key]["label"] = torch.argmax(label_score[i]).item()
        return parsed_tree

    def _update_tree_v2(
        self,
        decision_taken: List[int],
        label_score: torch.Tensor,
        config: Configuration,
        parsed_tree: List[dict],
        mask_label: List[int],
    ) -> List[dict]:
        for i, d in enumerate(decision_taken):
            if mask_label[i] == 1:
                # take the last arc
                arc = config[i].arc[-1]
                key = arc.dependent.position
                parsed_tree[i][key] = {"head": arc.head.position}
                parsed_tree[i][key]["label"] = torch.argmax(label_score[i]).item()
        return parsed_tree

    def get_decision(
        self, decision_score, config, oracle_decision, oracle_len, step_seq, static
    ):
        # FOR LEN COMPUTATION ON TRAIN AND VALID, TO COMPUTE METRICS
        if sb.Stage.TRAIN == self.stage or sb.Stage.VALID == self.stage:
            for i, o_d in enumerate(oracle_decision):
                # if first time seing alignment_oracle padding
                if o_d[step_seq] == self.oracle_padding_value and oracle_len[i] == -1:
                    oracle_len[i] = step_seq
        


        if sb.Stage.TRAIN == self.stage and static:
            decision_taken = [x[step_seq] for x in oracle_decision]
        elif sb.Stage.TRAIN == self.stage:
            # exploration rate should be warmed up
            if torch.rand(1) <= self.exploration_rate:
                decision_taken = self._get_best_valid_decision(
                    decision_score, config
                )
            else:
                decision_taken = [x[step_seq] for x in oracle_decision]
        else:
            # if test or validation, full exploration.
            decision_taken = self._get_best_valid_decision(decision_score, config)
        return decision_taken

    def _get_oracle_move_from_config_tree(
        self,
        config: Configuration,
        gold_config: GoldConfiguration,
        dynamic_oracle_decision: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        for i in range(len(config)):
            decision = self.dynamic_oracle.get_oracle_move_from_config_tree(
                config[i], gold_config[i]
            )
            dynamic_oracle_decision[i].append(torch.tensor(decision).to(self.device))
        return dynamic_oracle_decision

    def _get_oracle_label_from_config_tree(
        self,
        config: Configuration,
        gold_config: GoldConfiguration,
        decision: int,
        dynamic_oracle_label: List[List[int]],
    ) -> List[List[int]]:
        for i, d, c, gc in enumerate(zip(decision, config, gold_config)):
            label = self.dynamic_oracle.compute_label(c, gc, d)
            dynamic_oracle_label[i].append(label)
        return dynamic_oracle_label

    def _compute_features(self, config: Configuration) -> torch.Tensor:
        """
        strategy to compute features
        e.g : concat embedding of first buffer word to the embedding of the first 3 element of the stack
        :param config:
        :return:
        """
        has_root = [c.has_root for c in config]
        return self.features_computer.compute_feature(
            [x.stack for x in config], [x.buffer for x in config], has_root, torch.stack([x.root for x in config]).squeeze(1), self.device
        )

    def _compute_features_2(self, x: torch.tensor) -> torch.tensor:
        features = self.features_extractor(x)
        # TODO: deal rnn case

    def _get_static_supervision(
        self, static: bool, gold_config: List[GoldConfiguration]
    ) -> torch.Tensor:
        return self.static_oracle.compute_sequence(gold_config)

    def _apply_decision(
        self, decision: int, config: List[Configuration]
    ) -> List[Configuration]:
        """
        Function
        :param decision_score:
        :param config:
        :return:
        """

        for i, d in enumerate(decision):
            config[i] = self.transition.apply_decision(d, config[i])
        return config

    def _get_best_valid_decision(
        self, decision_score: torch.Tensor, config: List[Configuration]
    ) -> List[torch.Tensor]:
        """
        This function should be deprecated with addition of masked probablities (ie : Probablity of non valid decision
        = 0
        We should be able to either directly take the greedy actions or to sample from the probablity
        @param decision_score:
        @param config:
        @return:
        """
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

    def _is_terminal(self, config: Configuration) -> bool:
        """
        Return whether the configuration is terminal or not.
        :param config:
        :return:
        """
        return self.transition.is_terminal(config)

    def _compute_label_not_shared(
        self, config: List[Configuration], decision: List[int]
    ) -> (torch.Tensor, torch.Tensor):
        """
        This function compute the label for each element of the batch.
        Maybe compute for all, but discard the prediction of not applicable one ?
        ToDo: Optimization, only compute rep when needed.
         Use masking to add to correct list
         RQ: Can't really if batchnorm need to work ...
        """
        # fill mask with 1 when decision taken can't create a label (ie not an arc)
        mask = [1 if self.transition.require_label(d) else 0 for d in decision]
        batch_rep = self.label_policie.compute_representation_batch(
            config, decision, self.transition
        )
        label_score = self.label_neural_network(batch_rep)
        return label_score, mask

    def _compute_label_shared(self, features: torch.Tensor, decision: List[int]):
        mask = [1 if self.transition.require_label(d) else 0 for d in decision]
        label_score = self.label_head(features)
        return label_score, mask
