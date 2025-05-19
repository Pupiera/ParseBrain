import torch
from functools import partial
import itertools
import copy
#todo: Deal with LM (only on ASR task)

#todo: Using Compatible hypothesis to start t+1 ? Reduce the risk of the beam of each task diverging and never being compatible again.

class TransducerBeamSearcherMultitask(torch.nn.Module):
    """
    This class implements the beam-search algorithm for the transducer model.

    Parameters
    ----------
    decode_network_task_dict : dict of list
        List of prediction network (PN) layers.
    tjoint: transducer_joint module
        This module perform the joint between TN and PN.
    classifier_network : list
        List of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob
    blank_id : int
        The blank symbol/index.
    beam : int
        The width of beam. Greedy Search is used when beam = 1.
    nbest : int
        Number of hypotheses to keep.
    lm_module : torch.nn.ModuleList
        Neural networks modules for LM.
    lm_weight : float
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y). (default: 0.3)
    state_beam : float
        The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (beam_hyps), if not, end the while loop.
        Reference: https://arxiv.org/pdf/1911.01629.pdf
    expand_beam : float
        The threshold coefficient to limit the number of expanded hypotheses
        that are added in A (process_hyp).
        Reference: https://arxiv.org/pdf/1911.01629.pdf
        Reference: https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/simple-decoder.cc (See PruneToks)

    Example
    -------
    searcher = TransducerBeamSearcher(
        decode_network_lst=[hparams["emb"], hparams["dec"]],
        tjoint=hparams["Tjoint"],
        classifier_network=[hparams["transducer_lin"]],
        blank_id=0,
        beam_size=hparams["beam_size"],
        nbest=hparams["nbest"],
        lm_module=hparams["lm_model"],
        lm_weight=hparams["lm_weight"],
        state_beam=2.3,
        expand_beam=2.3,
    )
    >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    >>> import speechbrain as sb
    >>> from parsebrain.speechbrain_custom.nnet.transducer.transducer_joint import TransducerJointMultitask
    >>> emb = sb.nnet.embedding.Embedding(
    ...     num_embeddings=35,
    ...     embedding_dim=3,
    ...     consider_as_one_hot=True,
    ...     blank_id=0
    ... )
    >>> dec = sb.nnet.RNN.GRU(
    ...     hidden_size=10, input_shape=(1, 40, 34), bidirectional=False
    ... )
    >>> lin = sb.nnet.linear.Linear(input_shape=(1, 40, 10), n_neurons=35)
    >>> emb_pos = sb.nnet.embedding.Embedding(
    ...     num_embeddings=35,
    ...     embedding_dim=3,
    ...     consider_as_one_hot=True,
    ...     blank_id=0
    ... )
    >>> dec_pos = sb.nnet.RNN.GRU(
    ...     hidden_size=10, input_shape=(1, 40, 34), bidirectional=False
    ... )
    >>> lin_pos = sb.nnet.linear.Linear(input_shape=(1, 40, 10), n_neurons=35)
    >>> joint_network= sb.nnet.linear.Linear(input_shape=(1, 1, 40, 35), n_neurons=35)
    >>> tjoint = Transducer_joint(joint_network, joint="sum")
    >>> joint_syntax = TransducerJointMultitask()
    >>> task = {"ASR" : [emb, dec], "POS" : [emb_pos, dec_pos]}
    >>> classifier_net_task = {"ASR" : [lin], "POS" : [lin_pos]}
    >>> blank_id_task = {"ASR" : 0, "POS": 0}
    >>> beam_size_task = {"ASR" : 2, "POS": 2}
    >>> n_best_task = {"ASR": 2, "POS": 2}
    >>> searcher = TransducerBeamSearcherMultitask(
    ...     decode_network_task_dict=task,
    ...     tjoint=tjoint,
    ...     transducer_joint_multitask=joint_syntax,
    ...     classifier_network_task_dict=classifier_net_task,
    ...     blank_id_task=blank_id_task,
    ...     beam_size_task=beam_size_task,
    ...     nbest_task=1,
    ...     lm_module=None,
    ...     lm_weight=0.0,
    ... )
    >>> dict_final = {'ASR' : [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10], 'POS' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
    >>> searcher.dict_final = dict_final
    >>> enc = torch.rand([2, 20, 10])
    >>> hyps, scores, _, _ = searcher(enc)
    """

    def __init__(
            self,
            decode_network_task_dict,
            tjoint,
            classifier_network_task_dict,
            transducer_joint_multitask,
            blank_id_task,
            beam_size_task,
            nbest_task=5,
            lm_module=None,
            lm_weight=0.0,
            state_beam=2.3,
            expand_beam=2.3,
    ):
        super(TransducerBeamSearcherMultitask, self).__init__()
        self.decode_network_task_dict = decode_network_task_dict
        self.tjoint = tjoint
        self.transducer_joint_multitask = transducer_joint_multitask
        self.classifier_network_task_dict = classifier_network_task_dict
        self.blank_id_task = blank_id_task
        self.beam_size_task = beam_size_task
        self.nbest_task = nbest_task
        self.lm = lm_module
        self.lm_weight = lm_weight
        # test with global beam
        self.nbest=1

        if lm_module is None and lm_weight > 0:
            raise ValueError("Language model is not provided.")

        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if all([v <=1 for k, v in self.beam_size_task.items() ]):
            self.searcher = self.transducer_greedy_decode
        else:
            self.searcher = self.transducer_beam_search_decode

    def forward(self, tn_output):
        """
        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        Topk hypotheses
        """

        hyps = self.searcher(tn_output)
        return hyps

    def transducer_greedy_decode(self, tn_output):
        """Transducer greedy decoder is a greedy decoder over batch which apply Transducer rules:
            1- for each time step in the Transcription Network (TN) output:
                -> Update the ith utterance only if
                    the previous target != the new one (we save the hiddens and the target)
                -> otherwise:
                ---> keep the previous target prediction from the decoder

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """
        hyp_all = {}
        for task in self.decode_network_dict_lst.keys():

            hyp_task = {
                "prediction": [[] for _ in range(tn_output.size(0))],
                "logp_scores": [0.0 for _ in range(tn_output.size(0))],
            }
            hyp_all[task] = hyp_task
        # prepare BOS = Blank for the Prediction Network (PN)
        hidden = None
        # First forward-pass on PN
        out_PN_all = {}
        hidden_PN_all = {}
        input_PN_all = {}
        for task in self.decode_network_dict_lst.keys():
            input_PN = (
                    torch.ones(
                        (tn_output.size(0), 1),
                        device=tn_output.device,
                        dtype=torch.int32,
                    )
                    * self.blank_id_task[task]
            )
            input_PN_all[task] = input_PN
            out_PN, hidden = self._forward_PN(input_PN, self.decode_network_task_dict[task])
            out_PN_all[task] = out_PN
            hidden_PN_all[task] = hidden

        # For each time step
        for t_step in range(tn_output.size(1)):
            '''
            Do each task in parrallel. Need all the PN to be computed when joining with transcription network.
            Also get a joint PN representation of all the PN network.
            '''
            joint_PN_rep = self.compute_joint_PN(out_PN_all)
            for task in self.decode_network_task_dict.keys():
                # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                # for each task, predictions on the joint_PN_rep.
                log_probs = self._joint_forward_step(
                    tn_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
                    joint_PN_rep.unsqueeze(1),
                    self.classifier_network_task_dict[task]
                )

                # Sort outputs at time
                logp_targets, positions = torch.max(
                    self.softmax(log_probs).squeeze(1).squeeze(1), dim=1
                )
                # Batch hidden update
                have_update_hyp = []
                for i in range(positions.size(0)):
                    # Update hiddens only if
                    # 1- current prediction is non blank
                    if positions[i].item() != self.blank_id_task[task]:
                        hyp_all[task]["prediction"][i].append(positions[i].item())
                        hyp_all[task]["logp_scores"][i] += logp_targets[i]
                        input_PN_all[task][i][0] = positions[i]
                        have_update_hyp.append(i)
                if len(have_update_hyp) > 0:
                    # Select sentence to update
                    # And do a forward steps + generated hidden
                    (
                        selected_input_PN,
                        selected_hidden,
                    ) = self._get_sentence_to_update(
                        have_update_hyp, input_PN_all[task], hidden_PN_all[task]
                    )
                    selected_out_PN, selected_hidden = self._forward_PN(
                        selected_input_PN, self.decode_network_task_dict[task], selected_hidden
                    )
                    # update hiddens and out_PN
                    out_PN_all[task][have_update_hyp] = selected_out_PN
                    hidden_PN_all[task] = self._update_hiddens(
                        have_update_hyp, selected_hidden, hidden_PN_all[task]
                    )

        all_hyp = [h['predictions'] for h in hyp_all]
        all_log_prob = [h['logp_scores'].exp().mean() for h in hyp_all]
        return (
            all_hyp,
            all_log_prob,
            None,
            None,
        )


    def compute_joint_PN(self, output_PN):
        """
        this function take all the output_PN and combine them with the desired function (sum or concat + network)
        @param output_PN: dict with a keys for each task.
        @return:
        """
        list = [v for k, v in  output_PN.items() ]
        return self.transducer_joint_multitask(list)


    def transducer_beam_search_decode(self, tn_output):
        """Transducer beam search decoder is a beam search decoder over batch which apply Transducer rules:
            1- for each utterance:
                2- for each time steps in the Transcription Network (TN) output:
                    -> Do forward on PN and Joint network
                    -> Select topK <= beam
                    -> Do a while loop extending the hyps until we reach blank
                        -> otherwise:
                        --> extend hyp by the new token

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """

        # min between beam and max_target_lent
        n_best = []

        nbest_batch_score = []
        n_best_batch = []
        beam_global = []

        nbest_batch_all = {}
        nbest_batch_score_all = {}

        for task in self.decode_network_task_dict.keys():
            nbest_batch_all[task] = []
            nbest_batch_score_all[task] = []

        hyp_all = {}
        beam_hyps_all = {}
        # for each element of batch (sentence, audio)
        for i_batch in range(tn_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            out_PN_all={}
            hidden_all={}
            input_PN_all = {}
            blank_all = {}
            hyp_global = {"beams": {},
                        "logp_score": 0.0}
            for task in self.decode_network_task_dict.keys():
                blank = (
                        torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                        * self.blank_id_task[task]
                )
                input_PN = (
                        torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                        * self.blank_id_task[task]
                )
                input_PN_all[task] = input_PN
                blank_all[task] = blank
                out_PN, hidden = self._forward_PN(input_PN, self.decode_network_task_dict[task])
                out_PN_all[task] = out_PN
            # First forward-pass on PN
                hyp = {
                    "prediction": [self.blank_id_task[task]],
                    "logp_score": 0.0,
                    "hidden_dec": None,
                    "out_pn": None
                }
                hyp_all[task] = hyp
                hyp_global["beams"][task] = hyp
                beam_hyps_all[task] = [hyp]

            beam_global.append(hyp_global)


            if self.lm_weight > 0:
                # only for ASR task.
                # todo : make this clean to get ASR
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)

            #beam_hyps = [hyp]

            # For each time step
            for t_step in range(tn_output.size(1)):
                # beam local store at each timestep the best combination of beam for each task. {}
                beam_local = []
                for task in self.decode_network_task_dict.keys():
                    # get hyps for extension
                    #Use global beam here  to setup process hyp
                    process_hyps =  [beam["beams"][task] for beam in beam_global]
                    #process_hyps = beam_hyps_all[task]
                    beam_hyps = []
                    while True:
                        # Add norm score
                        a_best_hyp = max(
                            process_hyps, key=partial(get_transducer_key),
                        )

                        # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                        if len(beam_hyps) > 0:
                            b_best_hyp = max(
                                beam_hyps, key=partial(get_transducer_key),
                            )
                            a_best_prob = a_best_hyp["logp_score"]
                            b_best_prob = b_best_hyp["logp_score"]
                            if b_best_prob >= self.state_beam + a_best_prob:
                                break
                            if len([b for b in beam_hyps if b["logp_score"] >= a_best_prob]) >= self.beam_size_task[task]:
                                break
                        # remove best hyp from process_hyps
                        process_hyps.remove(a_best_hyp)

                        # forward PN
                        # why the [0,0] ?
                        input_PN_all[task][0, 0] = a_best_hyp["prediction"][-1]
                        out_PN, hidden = self._forward_PN(
                            input_PN_all[task],
                            self.decode_network_task_dict[task],
                            a_best_hyp["hidden_dec"],
                        )
                        out_PN_all[task] = out_PN
                        hidden_all[task] = hidden

                        joint_out_PN = {}
                        joint_out_PN[task] = out_PN
                        for task_ in self.decode_network_task_dict.keys():
                            if task_ == task:
                                continue
                            task_out_pn = max(beam_hyps_all[task], key=partial(get_transducer_key),)["out_pn"]
                            joint_out_PN[task_] = task_out_pn

                        joint_PN_rep = self.compute_joint_PN(joint_out_PN)
                        # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                        # do with joint rep so every task as information about the other dec
                        log_probs = self._joint_forward_step(
                            tn_output[i_batch, t_step, :]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .unsqueeze(0),
                            joint_PN_rep.unsqueeze(0),
                            self.classifier_network_task_dict[task]
                        )

                        if self.lm_weight > 0:
                            log_probs_lm, hidden_lm = self._lm_forward_step(
                                input_PN_all[task], a_best_hyp["hidden_lm"]
                            )

                        log_prob_view = log_probs.view(-1)

                        # Sort outputs at time
                        logp_targets, positions = torch.topk(
                            log_prob_view, k=self.beam_size_task[task], dim=-1
                        )

                        # take the best continuation except for the blank index
                        best_logp = (
                            logp_targets[0]
                            if positions[0] != blank_all[task]
                            else logp_targets[1]
                        )

                        # add y* + blank to B
                        hyp_blank = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"]
                                          + log_prob_view[self.blank_id_task[task]],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                            "out_pn": out_PN,
                        }
                        beam_hyps.append(hyp_blank)
                        # not sure about this part for the LM
                        if self.lm_weight > 0:
                            hyp_blank["hidden_lm"] = a_best_hyp["hidden_lm"]

                        # Extend hyp by  selection
                        for j in range(logp_targets.size(0)):
                            if positions[j] == self.blank_id_task[task]:
                                continue

                            topk_hyp = {
                                "prediction": a_best_hyp["prediction"][:],
                                "logp_score": a_best_hyp["logp_score"]
                                              + logp_targets[j],
                                "hidden_dec": a_best_hyp["hidden_dec"],
                                "out_pn": out_PN,
                            }

                            beam_hyps_all[task] = beam_hyps
                            if logp_targets[j] >= best_logp - self.expand_beam:
                                topk_hyp["prediction"].append(positions[j].item())
                                topk_hyp["hidden_dec"] = hidden
                                if self.lm_weight > 0:
                                    topk_hyp["hidden_lm"] = hidden_lm
                                    topk_hyp["logp_score"] += (
                                            self.lm_weight
                                            * log_probs_lm[0, 0, positions[j]]
                                    )
                                process_hyps.append(topk_hyp)

                #update each beam in beam_global as if we added blank (ie : do nothing)
                # Need to recompute the different out for each task (since we are going from somewhere else in the beam)
                for i in range(len(beam_global)):
                    # For each hypothesis (beam in beam_global)
                    input_PN_global = {}
                    out_PN_global = {}
                    for task in self.decode_network_task_dict.keys():
                        input_PN_global[task] = (
                            torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                            * self.blank_id_task[task]
                        )
                        # Need to compute this once per global beam and store it.
                        #todo: PB probably with the input not having the right shape
                        input_PN_global[task][0, 0] = beam_global[i]["beams"][task]["prediction"][-1]
                        hidden_dec = beam_global[i]["beams"][task]["hidden_dec"]
                        out_PN, hidden = self._forward_PN(
                            input_PN_global[task],
                            self.decode_network_task_dict[task],
                            hidden_dec,
                        )
                        out_PN_global[task] = out_PN
                        beam_global[i]["beams"][task]["hidden_dec"] = hidden
                    #compute joint probs
                    joint_PN_global = self.compute_joint_PN(out_PN_global)
                    for task in self.decode_network_task_dict.keys():
                        # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                        # do with joint rep so every task as information about the other dec
                        log_probs = self._joint_forward_step(
                                tn_output[i_batch, t_step, :]
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .unsqueeze(0),
                                joint_PN_global.unsqueeze(0),
                                self.classifier_network_task_dict[task]
                            )
                        log_prob_view = log_probs.view(-1)
                        beam_global[i]["beams"][task]["logp_score"]+= log_prob_view[self.blank_id_task[task]].item()
                        beam_global[i]["logp_score"]+= log_prob_view[self.blank_id_task[task]].item()

                #compute the best legal combination of beam.
                prob_combined_beam = self.compute_compatible_matrix_prob(beam_hyps_all)
                #remove masked value (-0) to not be choosen by topk
                prob_combined_beam[prob_combined_beam == -0] = -1e9

                # get the beam indice from the topk (flattened)
                #todo : define value of the topk for beam local.
                prob_combined_beam_flatten = prob_combined_beam.flatten()
                k = min(5, prob_combined_beam_flatten.shape[0])
                top_values, top_indices = torch.topk(prob_combined_beam_flatten, k=k)
                mask = torch.isin(prob_combined_beam, top_values)
                top_coordinates = torch.stack(torch.where(mask), dim=1)
                # do i need to add the hidden_dec ?
                for coordinate in top_coordinates:
                    beam_task = {}
                    for i, task in enumerate(beam_hyps_all.keys()):
                        local_coordinate = coordinate[i]
                        beam_task[task] = beam_hyps_all[task][local_coordinate]
                    hyp_local = {'beams' : beam_task,
                                 'logp_score' : prob_combined_beam[tuple(coordinate)]}
                    beam_local.append(hyp_local)
                beam_global.extend(beam_local)

                #todo: define beam_size value for beam_global, and maybe a less dumb way than top hyp ?
                beam_global=sorted(beam_global, key=partial(get_transducer_key_multitask), reverse=True)[:5]

            # WITH BEAM GLOBAL
            n_best_hyps = sorted(beam_global, key=partial(get_transducer_key_multitask), reverse = True)[: self.nbest]
            all_predictions = []
            all_scores = []
            for hyp in n_best_hyps:
                pred = {}
                for task in self.decode_network_task_dict.keys():
                    # remove the first default blank
                    pred[task] = hyp["beams"][task]["prediction"][1:]
                all_predictions.append(pred)
                all_scores.append(hyp["logp_score"])
            n_best_batch.append(all_predictions)
            nbest_batch_score.append(all_scores)

        n_best = (
            [n_best_utt[0] for n_best_utt in n_best_batch],
            torch.Tensor([nbest_utt_score[0] for nbest_utt_score in nbest_batch_score]).exp().mean(),
            n_best_batch,
            nbest_batch_score
        )
        return n_best

    def _joint_forward_step(self, h_i, out_PN, classifier_network):
        """Join predictions (TN & all PN). """

        with torch.no_grad():
            # the output would be a tensor of [B,T,U, oneof[sum,concat](Hidden_TN,Hidden_PN)]
            out = self.tjoint(h_i, out_PN, )
            # forward the output layers + activation + save logits
            out = self._forward_after_joint(out, classifier_network)
            log_probs = self.softmax(out)
        return log_probs

    def _lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        hs : No limit
            The memory variables are generated in this timestep.
            (e.g., RNN hidden states).
        """
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits)
        return log_probs, hs

    def _get_sentence_to_update(self, selected_sentences, output_PN, hidden):
        """Select and return the updated hiddens and output
        from the Prediction Network.

        Arguments
        ----------
        selected_sentences : list
            List of updated sentences (indexes).
        output_PN: torch.tensor
            Output tensor from prediction network (PN).
        hidden : torch.tensor
            Optional: None, hidden tensor to be used for
            recurrent layers in the prediction network.

        Returns
        -------
        selected_output_PN: torch.tensor
            Outputs a logits tensor [B_selected,U, hiddens].
        hidden_update_hyp: torch.tensor
            Selected hiddens tensor.
        """

        selected_output_PN = output_PN[selected_sentences, :]
        # for LSTM hiddens (hn, hc)
        if isinstance(hidden, tuple):
            hidden0_update_hyp = hidden[0][:, selected_sentences, :]
            hidden1_update_hyp = hidden[1][:, selected_sentences, :]
            hidden_update_hyp = (hidden0_update_hyp, hidden1_update_hyp)
        else:
            hidden_update_hyp = hidden[:, selected_sentences, :]
        return selected_output_PN, hidden_update_hyp

    def _update_hiddens(self, selected_sentences, updated_hidden, hidden):
        """Update hidden tensor by a subset of hidden tensor (updated ones).

        Arguments
        ----------
        selected_sentences : list
            List of index to be updated.
        updated_hidden : torch.tensor
            Hidden tensor of the selected sentences for update.
        hidden : torch.tensor
            Hidden tensor to be updated.

        Returns
        -------
        torch.tensor
            Updated hidden tensor.
        """

        if isinstance(hidden, tuple):
            hidden[0][:, selected_sentences, :] = updated_hidden[0]
            hidden[1][:, selected_sentences, :] = updated_hidden[1]
        else:
            hidden[:, selected_sentences, :] = updated_hidden
        return hidden

    def _forward_PN(self, out_PN, decode_network_lst, hidden=None):
        """Compute forward-pass through a list of prediction network (PN) layers.

        Arguments
        ----------
        out_PN : torch.tensor
            Input sequence from prediction network with shape
            [batch, target_seq_lens].
        decode_network_lst: list
            List of prediction network (PN) layers.
        hinne : torch.tensor
            Optional: None, hidden tensor to be used for
                recurrent layers in the prediction network

        Returns
        -------
        out_PN : torch.tensor
            Outputs a logits tensor [B,U, hiddens].
        hidden : torch.tensor
            Hidden tensor to be used for the next step
            by recurrent layers in prediction network.
        """
        for layer in decode_network_lst:
            if layer.__class__.__name__ in [
                "RNN",
                "LSTM",
                "GRU",
                "LiGRU",
                "LiGRU_Layer",
            ]:
                out_PN, hidden = layer(out_PN, hidden)
            else:
                out_PN = layer(out_PN)
        return out_PN, hidden

    def _forward_after_joint(self, out, classifier_network):
        """Compute forward-pass through a list of classifier neural network.

        Arguments
        ----------
        out : torch.tensor
            Output from joint network with shape
            [batch, target_len, time_len, hiddens]
        classifier_network : list
            List of output layers (after performing joint between TN and PN)
            exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B, U,T, Output_Dim];
        """

        for layer in classifier_network:
            out = layer(out)
        return out

    def compute_compatible_matrix_prob(self, beams):
        '''

        @param beams:
        @return:
        '''
        pred = {}
        logp = {}
        for task, task_beam in beams.items():
            pred[task] = []
            logp[task] = []
            for b in task_beam:
                pred[task].append(b['prediction'])
                logp[task].append(b['logp_score'])
        beams_len = self._get_beam_len(pred)
        matrix_mask = self._compute_matrix_compatibility(beams_len)
        matrix_prob = self._compute_matrix_prob(logp)
        return matrix_mask * matrix_prob


    def _compute_matrix_prob(self, beams):
        '''
        Compute the combination of probability in each element of the beams.
        @param beams:
        @return:
        >>> searcher = TransducerBeamSearcherMultitask(
        ...     decode_network_task_dict={},
        ...     tjoint=None,
        ...     transducer_joint_multitask={},
        ...     classifier_network_task_dict={},
        ...     blank_id_task={},
        ...     beam_size_task={},
        ...     nbest_task=1,
        ...     lm_module=None,
        ...     lm_weight=0.0,
        ... )
        >>> beams = {'ASR': [0.5, 0.25], 'POS': [0.4, 0.8], 'GOV': [0.2, 0.8, 0.3]}
        >>> searcher._compute_matrix_prob(beams)
        tensor([[[1.1000, 1.7000, 1.2000],
                 [1.5000, 2.1000, 1.6000]],
        <BLANKLINE>
                [[0.8500, 1.4500, 0.9500],
                 [1.2500, 1.8500, 1.3500]]])
        '''


        dims = [len(b) for k, b in beams.items()]
        beam_prob = [b for k, b in beams.items()]
        indices = torch.cartesian_prod(*[torch.arange(dim) for dim in dims])
        result = torch.zeros(dims)
        for i, b in enumerate(beam_prob):
            b_indices = indices[:, i]
            b_values = torch.tensor(b)[b_indices]
            result += b_values.reshape(dims)
        return result

    def _compute_matrix_compatibility(self, beams_len):
        '''
        This function compute the compatibility between beam.
        For a pair of beam to be compatible, they need to have the same length as defined in beams_len
        ie : the same number of word and part of speech for example.
        @param beams_len: list of list of shape [nb_task, beam_size]
        @return: a boolean matrice of shape [beams_size[0], beams_size[1], ..., beams_size[n]]

        >>> searcher = TransducerBeamSearcherMultitask(
        ...     decode_network_task_dict={},
        ...     tjoint=None,
        ...     transducer_joint_multitask={},
        ...     classifier_network_task_dict={},
        ...     blank_id_task={},
        ...     beam_size_task={},
        ...     nbest_task=1,
        ...     lm_module=None,
        ...     lm_weight=0.0,
        ... )
        >>> beams_len = [[1,2,3], [1,2], [2,9,9]]
        >>> x = searcher._compute_matrix_compatibility(beams_len)
        >>> print(x[1][1][0])
        tensor(True)
        >>> print(x[0][1][0])
        tensor(False)
        '''

        num_tasks = len(beams_len)
        beam_sizes = [len(beams_len[i]) for i in range(num_tasks)]

        compatibility_matrix = torch.zeros(size=beam_sizes, dtype=torch.bool)

        for beam_indices in itertools.product(*[range(size) for size in beam_sizes]):
            beam_lengths = []
            for task in range(num_tasks):
                beam_lengths.append(beams_len[task][beam_indices[task]])
            if all(beam_lengths[i] == beam_lengths[0] for i in range(num_tasks)):
                compatibility_matrix[beam_indices] = True

        return compatibility_matrix


    def _get_beam_len(self, beams):
        '''
        Compute the len of each beam (Only count final element, ie take into account the subword.)
        @param beams:
        @return:

        >>> searcher = TransducerBeamSearcherMultitask(
        ...     decode_network_task_dict={},
        ...     tjoint=None,
        ...     transducer_joint_multitask={},
        ...     classifier_network_task_dict={},
        ...     blank_id_task={},
        ...     beam_size_task={},
        ...     nbest_task=1,
        ...     lm_module=None,
        ...     lm_weight=0.0,
        ... )
        >>> searcher.dict_final = {'ASR': [1, 5, 9],'POS': [1, 2, 3]}
        >>> beams = {'ASR': [[1, 0, 2, 3, 4, 5]],'POS': [[1, 0, 3, 2]]}
        >>> searcher._get_beam_len(beams)
        [[2], [3]]
        '''
        dict_final = self.dict_final
        beams_len = []
        for task, task_beam in beams.items():
            local_beams_len = []
            for beam in task_beam:
                len_b = 0
                for b in beam:
                    if b in dict_final[task]:
                        len_b+=1
                local_beams_len.append(len_b)
            beams_len.append(local_beams_len)
        return beams_len

class TransducerBeamSearcherMultitaskSimple(TransducerBeamSearcherMultitask):
    '''
    >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    >>> import speechbrain as sb
    >>> from parsebrain.speechbrain_custom.nnet.transducer.transducer_joint import TransducerJointMultitask
    >>> emb = sb.nnet.embedding.Embedding(
    ...     num_embeddings=35,
    ...     embedding_dim=3,
    ...     consider_as_one_hot=True,
    ...     blank_id=0
    ... )
    >>> dec = sb.nnet.RNN.GRU(
    ...     hidden_size=10, input_shape=(1, 40, 34), bidirectional=False
    ... )
    >>> lin = sb.nnet.linear.Linear(input_shape=(1, 40, 10), n_neurons=35)
    >>> emb_pos = sb.nnet.embedding.Embedding(
    ...     num_embeddings=35,
    ...     embedding_dim=3,
    ...     consider_as_one_hot=True,
    ...     blank_id=0
    ... )
    >>> dec_pos = sb.nnet.RNN.GRU(
    ...     hidden_size=10, input_shape=(1, 40, 34), bidirectional=False
    ... )
    >>> lin_pos = sb.nnet.linear.Linear(input_shape=(1, 40, 10), n_neurons=35)
    >>> joint_network= sb.nnet.linear.Linear(input_shape=(1, 1, 40, 35), n_neurons=35)
    >>> tjoint = Transducer_joint(joint_network, joint="sum")
    >>> joint_syntax = TransducerJointMultitask()
    >>> task = {"ASR" : [emb, dec], "POS" : [emb_pos, dec_pos]}
    >>> classifier_net_task = {"ASR" : [lin], "POS" : [lin_pos]}
    >>> blank_id_task = {"ASR" : 0, "POS": 0}
    >>> beam_size_task = {"ASR" : 2, "POS": 2}
    >>> n_best_task = {"ASR": 1, "POS": 1}
    >>> searcher = TransducerBeamSearcherMultitaskSimple(
    ...     decode_network_task_dict=task,
    ...     tjoint=tjoint,
    ...     transducer_joint_multitask=joint_syntax,
    ...     classifier_network_task_dict=classifier_net_task,
    ...     blank_id_task=blank_id_task,
    ...     beam_size_task=beam_size_task,
    ...     nbest_task=n_best_task,
    ...     lm_module=None,
    ...     lm_weight=0.0,
    ... )
    >>> dict_final = {'ASR' : [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10], 'POS' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
    >>> searcher.dict_final = dict_final
    >>> enc = torch.rand([2, 20, 10])
    >>> hyps = searcher(enc)
    '''
    def transducer_beam_search_decode(self, tn_output):
        '''
        Return a dictionary x[task] : ()
        '''
        # min between beam and max_target_lent
        n_best_all = []

        nbest_batch_score = []
        n_best_batch = []

        nbest_batch_all = {}
        nbest_batch_score_all = {}

        for task in self.decode_network_task_dict.keys():
            nbest_batch_all[task] = []
            nbest_batch_score_all[task] = []

        hyp_all = {}
        beam_hyps_all = {}
        # for each element of batch (sentence, audio)
        for i_batch in range(tn_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            out_PN_all={}
            hidden_all={}
            input_PN_all = {}
            blank_all = {}
            for task in self.decode_network_task_dict.keys():
                blank = (
                        torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                        * self.blank_id_task[task]
                )
                input_PN = (
                        torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                        * self.blank_id_task[task]
                )
                input_PN_all[task] = input_PN
                blank_all[task] = blank
                out_PN, hidden = self._forward_PN(input_PN, self.decode_network_task_dict[task])
                out_PN_all[task] = out_PN
            # First forward-pass on PN
                hyp = {
                    "prediction": [self.blank_id_task[task]],
                    "logp_score": 0.0,
                    "hidden_dec": None,
                    "out_pn": None,
                }
                hyp_all[task] = hyp
                beam_hyps_all[task] = [hyp]
            if self.lm_weight > 0:
                # only for ASR task.
                # todo : make this clean to get ASR
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)
                beam_hyps_all[task] = [hyp]


            # For each time step
            for t_step in range(tn_output.size(1)):
                for task in self.decode_network_task_dict.keys():
                    # get hyps for extension
                    #Use global beam here  to setup process hyp
                    process_hyps = copy.copy(beam_hyps_all[task])
                    beam_hyps = []
                    count=0
                    while True:
                        # Add norm score
                        a_best_hyp = max(
                            process_hyps, key=partial(get_transducer_key),
                        )

                        # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                        if len(beam_hyps) > 0:
                            b_best_hyp = max(
                                beam_hyps, key=partial(get_transducer_key),
                            )
                            a_best_prob = a_best_hyp["logp_score"]
                            b_best_prob = b_best_hyp["logp_score"]
                            if b_best_prob >= self.state_beam + a_best_prob:
                                break
                            if len([b for b in beam_hyps if b["logp_score"] >= a_best_prob]) >= self.beam_size_task[task]:
                                break
                            #hard break to stop beam if more than X step has been done for time t
                            if count >100:
                                break
                        # remove best hyp from process_hyps
                        process_hyps.remove(a_best_hyp)
                        count+=1
                        # forward PN
                        # why the [0,0] ?
                        input_PN_all[task][0, 0] = a_best_hyp["prediction"][-1]
                        out_PN, hidden = self._forward_PN(
                            input_PN_all[task],
                            self.decode_network_task_dict[task],
                            a_best_hyp["hidden_dec"],
                        )
                        #todo: Find better way to store out_PN linked to the beam ? 
                        # combine with the best prob of the other beam ?
                        a_best_hyp["out_pn"] = out_PN
                        #out_PN_all[task] = out_PN
                        hidden_all[task] = hidden
                        # Compute joint_rep based on best t-1 beam out_PN for other task and on current out_PN for current task
                        joint_out_PN = {}
                        joint_out_PN[task] = out_PN
                        for task_ in self.decode_network_task_dict.keys():
                            if task_ == task:
                                continue
                            task_out_pn = max(beam_hyps_all[task], key=partial(get_transducer_key),)["out_pn"]
                            joint_out_PN[task_] = task_out_pn

                        joint_PN_rep = self.compute_joint_PN(joint_out_PN)

                        # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                        # do with joint rep so every task as information about the other dec
                        log_probs = self._joint_forward_step(
                            tn_output[i_batch, t_step, :]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .unsqueeze(0),
                            joint_PN_rep.unsqueeze(0),
                            self.classifier_network_task_dict[task]
                        )

                        if self.lm_weight > 0:
                            log_probs_lm, hidden_lm = self._lm_forward_step(
                                input_PN_all[task], a_best_hyp["hidden_lm"]
                            )

                        log_prob_view = log_probs.view(-1)

                        # Sort outputs at time
                        logp_targets, positions = torch.topk(
                            log_prob_view, k=self.beam_size_task[task], dim=-1
                        )

                        # take the best continuation except for the blank index
                        best_logp = (
                            logp_targets[0]
                            if positions[0] != blank_all[task]
                            else logp_targets[1]
                        )

                        # add y* + blank to B
                        hyp_blank = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"]
                                          + log_prob_view[self.blank_id_task[task]],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                            "out_pn" : out_PN
                        }
                        beam_hyps.append(hyp_blank)
                        # not sure about this part for the LM
                        if self.lm_weight > 0:
                            hyp_blank["hidden_lm"] = a_best_hyp["hidden_lm"]

                        # Extend hyp by  selection
                        for j in range(logp_targets.size(0)):
                            if positions[j] == self.blank_id_task[task]:
                                continue

                            topk_hyp = {
                                "prediction": a_best_hyp["prediction"][:],
                                "logp_score": a_best_hyp["logp_score"]
                                              + logp_targets[j],
                                "hidden_dec": a_best_hyp["hidden_dec"],
                                "out_PN": out_PN
                            }

                            beam_hyps_all[task] = beam_hyps
                            if logp_targets[j] >= best_logp - self.expand_beam:
                                topk_hyp["prediction"].append(positions[j].item())
                                topk_hyp["hidden_dec"] = hidden
                                if self.lm_weight > 0:
                                    topk_hyp["hidden_lm"] = hidden_lm
                                    topk_hyp["logp_score"] += (
                                            self.lm_weight
                                            * log_probs_lm[0, 0, positions[j]]
                                    )
                                process_hyps.append(topk_hyp)

            for task in self.decode_network_task_dict.keys():
                # Add norm score
                nbest_hyps = sorted(
                    beam_hyps_all[task], key=partial(get_transducer_key), reverse=True,
                )[: self.nbest_task[task]]
                all_predictions = []
                all_scores = []

                for hyp in nbest_hyps:
                    all_predictions.append(hyp["prediction"][1:])
                    all_scores.append(hyp["logp_score"] / len(hyp["prediction"]))
                nbest_batch_all[task].append(all_predictions)
                nbest_batch_score_all[task].append(all_scores)
        for i_batch in range(tn_output.size(0)):
            x = {}
            for task in self.decode_network_task_dict.keys():
                x[task] = [
                nbest_batch_all[task][i_batch][0],
                    torch.Tensor(
                        nbest_batch_score_all[task][i_batch][0]
                    )
                    .exp()
                    .mean(),
                    nbest_batch_all[task][i_batch],
                    nbest_batch_score_all[task][i_batch],
                ]
            n_best_all.append(x)
        return n_best_all


def get_transducer_key(x):
    """Argument function to customize the sort order (in sorted & max).
    To be used as `key=partial(get_transducer_key)`.

    Arguments
    ----------
    x : dict
        one of the items under comparison

    Returns
    -------
    float
        Normalized log-score.
    """
    logp_key = x["logp_score"] / len(x["prediction"])

    return logp_key



def get_transducer_key_multitask(x):
    logp_key = x["logp_score"]
    return logp_key


if __name__ == "__main__":
    import doctest
    doctest.testmod()
