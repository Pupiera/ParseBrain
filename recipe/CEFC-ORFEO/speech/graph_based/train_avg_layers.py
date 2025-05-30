#!/usr/bin/env python3
import sys
import torch
from tqdm import tqdm
import time
import types
from typing import NamedTuple


import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main


from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader

from parsebrain.dataio.pred_to_file.pred_to_conllu import write_token_dict_conllu
from parsebrain.speechbrain_custom.decoders.ctc import ctc_greedy_decode
from torch.nn.utils.rnn import pad_sequence
import logging

# ---------------------Scoring import----------------------#
from parsebrain.processing.dependency_parsing.graph_based import score_to_tree


# --------------------------------------- Alignment import --------------------------------#
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.dataio.wer import print_alignments
from natsort import natsorted
from operator import itemgetter

# ----------------------------------------- Import profiling -------------------------------------------#
import wandb

"""Recipe for training a sequence-to-sequence ASR system with Orfeo.
The system employs a wav2vec2 encoder and a ASRDep2Label decoder.
Decoding is performed with greedy decoding (will be extended to beam search with language model).

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml

With the default hyperparameters, the system employs a pretrained wav2vec2 encoder.
The wav2vec2 model is pretrained following the model given in the hprams file.
It may be dependent on the language.

Authors
 * Titouan Parcollet 2021 for ASR template
 * PUPIER Adrien 2022 for adapting template to dependency parsing.
"""

INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"

# -----------------------------------------------------------BRAIN ASR---------------------------------------------#
# Define training procedure
class ASR(sb.core.Brain):
    """
    This is a subclass of Brain in speechbrain.
    Contains the definition/ behaviour of the model (forward + loss computation)
    """

    def compute_forward(self, batch, stage):
        """
        This function compute the forward pass of batch.
        It uses the Dep2Label paradigm for dependency parsing

        Parameters
        ----------
        batch : the corresponding row of the CSV file. ( contains the value defined in the pipeline)
        ["id", "sig", "tokens_bos", "tokens_eos", "tokens", "wrd", "pos_tensor", "govDep2label", "depDep2Label"]
        stage : TRAIN, DEV, TEST. Allow for specific computation based on the stage.

        Returns
        p_ctc : The probability of a given character in this frame [batch, time, char_alphabet]
        wav_lens : the lentgh of the wav file
        p_depLabel : the probablity of the syntactic functions [batch, max(seq_len), dep_alphabet]
        p_govLabel : the probablity of the head [batch, max(seq_len), gov_alphabet]
        p_posLabel : the probablity of the part of speech : [batch, max(seq_len), POS_alphabet]
        seq_len : the length of each dependency parsing element of the batch (audio word embedding lentgh)
        -------

        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass ASR
        batch_size = wavs.shape[0]
        feats = self.modules.wav2vec2(wavs)
        # ASR
        feats_asr = feats[-1]
        x_asr = self.modules.enc(feats_asr)  # [batch,time,1024]
        logits = self.modules.ctc_lin(x_asr)  # [batch,time,76]

        # [batch,time,76] (76 in output neurons corresponding to BPE size)
        p_ctc = self.hparams.log_softmax(logits)
        # Forward pass dependency parsing
        result = {"p_ctc": p_ctc, "wav_lens": wav_lens}
        if self.is_training_syntax:
            feats_syntax = feats[int(self.hparams.begin_layer):-1]
            feats_syntax = torch.mean(feats_syntax, dim=-1)
            x_parsing = self.modules.enc(feats_syntax)
            sequence, mapFrameToWord = ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            input_dep, seq_len = self._create_inputDep(x_parsing, mapFrameToWord)
            input_dep = input_dep.to(self.device)
            # Add a dummy root embedding. Important for graph based computation.
            embedding_root = self.modules.embedding_root(
                torch.zeros((batch_size, 1)).to(self.device)
            ).to(self.device)
            input_dep = torch.cat((embedding_root, input_dep), dim=1)
            # Need to add one to the seq len since we added the dummy root.

            x = torch.ones((batch_size)).to(self.device)
            seq_len = torch.add(seq_len.to(self.device), x).to(
                dtype=torch.long, device=self.device
            )

            batch_len = input_dep.shape[0]
            # init hidden to zeros for each sentence
            features, _ = self.modules.rnn(input_dep)
            pos_scores, arc_scores, dep_scores = self.modules.graph_parser(features)
            arc_scores = self.hparams.log_softmax(arc_scores)
            dep_scores = self.hparams.log_softmax(dep_scores)

            result_parsing = {
                "pos_scores": pos_scores,
                "dep_scores": dep_scores,
                "arc_scores": arc_scores,
                "seq_len": seq_len,
            }
            result = {**result, **result_parsing}
        return result

    def _create_inputDep(self, x, mapFrameToWord):
        """
        Compute the word audio embedding
        Parameters
        ----------
        x : The encoder representation output ( 1 frame per 20ms with wav2vec )
        mapFrameToWord : The mapping of frame to word from the CTC module.

        Returns
        batch : The padded batch of word audio embedding [batch, max(seq_len)]
        seq_len : the length of each element of the batch
        -------

        """
        batch = []
        hidden_size = self.hparams.repFusionHidden
        is_bidirectional = self.hparams.repFusionBidirectional
        n_layers = self.hparams.repFusionLayers
        nb_hidden = n_layers * (1 + is_bidirectional)
        for i, (rep, map) in enumerate(
            zip(x, mapFrameToWord)
        ):  # for 1 element on the batch do :
            map = torch.Tensor(map)
            uniq = torch.unique(map)
            fusionedRep = []
            # init hidden to zeros for each sentence
            hidden = torch.zeros(nb_hidden, 1, hidden_size, device=self.device)
            # init hidden to zeros for each sentence
            cell = torch.zeros(nb_hidden, 1, hidden_size, device=self.device)
            # For each word find the limit of the relevant sequence of frames
            for e in uniq:
                # ignore 0, if empty tensor, try with everything (i.e correspond to transition of words)
                if e.item() == 0 and len(uniq) > 1:
                    continue
                relevant_column = (map == e).nonzero(as_tuple=False)
                min = torch.min(relevant_column)
                max = torch.max(relevant_column)
                # should not break autograd https://discuss.pytorch.org/t/select-columns-from-a-batch-of-matrices-by-index/85321/3
                frames = rep[min : max + 1, :].unsqueeze(0)
                # extract feature from all the relevant audio frame representation
                _, (hidden, cell) = self.modules.RepFusionModel(frames, (hidden, cell))
                if is_bidirectional:
                    fusionedRep.append(torch.cat((hidden[-2], hidden[-1]), dim=1))
                else:
                    fusionedRep.append(hidden[-1])
            batch.append(torch.stack(fusionedRep))
            # print(f"Last tensor shape : {batch[-1].shape}")
        seq_len = [len(e) for e in batch]
        batch = torch.reshape(
            pad_sequence(batch, batch_first=True),
            (len(mapFrameToWord), -1, hidden_size * (1 + is_bidirectional)),
        )
        return batch, torch.Tensor(seq_len)

    def compute_objectives_syntax(
        self, predicted_words, p_depLabel, p_govLabel, p_posLabel, seq_len, batch
    ):
        ids = batch.id
        batch_size = len(ids)
        gold_dep = batch.dep_tokens[0]  # get tensor from padded data class
        gold_head = batch.head[0]
        gold_pos = batch.pos_tokens[0]
        tokens, tokens_lens = batch.tokens
        target_words = undo_padding(tokens, tokens_lens)
        target_words = self.tokenizer(target_words, task="decode_from_list")
        try:
            # No len mismatch beetwen pred and target. Using speechrbain alignment
            allowed = 0
            # and dynamic alignment_oracle based on the segmentation of the ASR predictions
            # 1 is allowed in case of inerstion of 1 empty string at the end (happen rarely)

            wer_details_alig = wer_details_for_batch(
                ids=ids,
                refs=target_words,
                hyps=predicted_words,
                compute_alignments=True,
            )
            # format  [('=', 0, 0), ('D', 1, None), ('=', 2, 1), ('S', 3, 2), ('I', None, 3)]
            # ie : (type of token, gold_position, system_position)
            wer_alig = [a["alignment"] for a in wer_details_alig]
            # [1:] because of root inserted in the begining, do not take into account for alignment_oracle.
            (
                gold_head,
                gold_dep,
                gold_pos,
            ) = self.hparams.oracle.find_best_tree_from_alignment(
                wer_alig,
                [x[1:] for x in gold_head.tolist()],
                [x[1:] for x in gold_dep.tolist()],
                [x[1:] for x in gold_pos.tolist()],
            )
            gold_dep = pad_sequence(gold_dep, batch_first=True).to(self.device)
            gold_head = pad_sequence(gold_head, batch_first=True).to(self.device)
            gold_pos = pad_sequence(gold_pos, batch_first=True).to(self.device)

            root_dep = torch.full(
                (batch_size, 1), fill_value=dep_label_dict.get("ROOT")
            ).to(self.device)
            gold_dep = torch.cat((root_dep, gold_dep), dim=1)
            root_head = torch.full((batch_size, 1), fill_value=0).to(self.device)
            gold_head = torch.cat((root_head, gold_head), dim=1)
            root_pos = torch.full((batch_size, 1), fill_value=pos_dict.get("ROOT")).to(
                self.device
            )
            gold_pos = torch.cat((root_pos, gold_pos), dim=1)

            # add value for root

            try:
                num_padded_deps = gold_head.shape[1]
                num_deprels = p_depLabel.shape[-1]
                # Need to add one to sequence len because of root.

                loss_pos = self.hparams.posLabel_cost(
                    p_posLabel, gold_pos, length=seq_len, allowed_len_diff=allowed
                )
                loss_arc = self.hparams.govLabel_cost(
                    p_govLabel,
                    gold_head,
                    length=seq_len,
                    allowed_len_diff=allowed,
                )

                # label loss, must not take into account the non existing arc
                # we replace the value of padding with 0
                positive_heads = gold_head.masked_fill(
                    gold_head.eq(self.hparams.LABEL_PADDING), 0
                )
                heads_selection = positive_heads.view(batch_size, num_padded_deps, 1, 1)
                # [batch, n_dependents, 1, n_labels]
                heads_selection = heads_selection.expand(
                    batch_size, num_padded_deps, 1, num_deprels
                )
                # [batch, n_dependents, 1, n_labels]
                # squeeze dim is given to preserve the batch size in case of end of epoch
                # where batc might be equal to 1
                predicted_labels_scores = torch.gather(
                    p_depLabel, -2, heads_selection
                ).squeeze(dim=2)
                loss_deprel = self.hparams.depLabel_cost(
                    predicted_labels_scores,
                    gold_dep,
                    length=seq_len,
                    allowed_len_diff=allowed,
                )
            except (ValueError, RuntimeError) as e:
                print(gold_head)
                print(gold_dep)
                print(gold_pos)
                raise ValueError() from e
        except ValueError as e:
            print(f" True shape : {gold_dep.shape}, pred shape: {p_depLabel.shape}")
            raise ValueError() from e
        return {
            "loss_gov": loss_arc,
            "loss_dep": loss_deprel,
            "loss_pos": loss_pos,
        }

    def compute_objectives(self, predictions, batch, stage):
        """
        if self.is_training_syntax:
            p_ctc, wav_lens, p_depLabel, p_govLabel, p_posLabel, seq_len = predictions
        else:
            p_ctc, wav_lens = predictions
        """
        ids = batch.id

        tokens, tokens_lens = batch.tokens
        loss = self.hparams.ctc_cost(
            predictions["p_ctc"], tokens, predictions["wav_lens"], tokens_lens
        )
        if self.is_training_syntax:
            sequence = sb.decoders.ctc_greedy_decode(
                predictions["p_ctc"],
                predictions["wav_lens"],
                blank_id=self.hparams.blank_index,
            )
            predicted_words = self.tokenizer(sequence, task="decode_from_list")
            for i, sent in enumerate(predicted_words):
                for w in sent:
                    if w == "":
                        predicted_words[i] = [w for w in sent if w != ""]
                        if len(predicted_words[i]) == 0:
                            predicted_words[i].append("EMPTY_ASR")
            loss_dict = self.compute_objectives_syntax(
                predicted_words,
                predictions["dep_scores"],
                predictions["arc_scores"],
                predictions["pos_scores"],
                predictions["seq_len"],
                batch,
            )
            loss = (
                loss * 0.4
                + loss_dict["loss_gov"] * 0.2
                + loss_dict["loss_dep"] * 0.2
                + loss_dict["loss_pos"] * 0.2
            )
        if sb.Stage.TRAIN != stage:
            # need to get predicted words
            if not self.is_training_syntax:
                sequence = sb.decoders.ctc_greedy_decode(
                    predictions["p_ctc"],
                    predictions["wav_lens"],
                    blank_id=self.hparams.blank_index,
                )
                predicted_words = self.tokenizer(sequence, task="decode_from_list")
                for i, sent in enumerate(predicted_words):
                    for w in sent:
                        if w == "":
                            predicted_words[i] = [w for w in sent if w != ""]
                            if len(predicted_words[i]) == 0:
                                predicted_words[i].append("EMPTY_ASR")

            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

            wer_details = wer_details_for_batch(
                ids=ids,
                refs=target_words,
                hyps=predicted_words,
                compute_alignments=True,
            )
            self.stage_wer_details.extend(wer_details)
            if self.is_training_syntax:
                for a_scores, d_scores, p_scores, seq_len, form, sent_id in zip(
                    predictions["arc_scores"],
                    predictions["dep_scores"],
                    predictions["pos_scores"],
                    predictions["seq_len"],
                    predicted_words,
                    batch.id,
                ):
                    tree = score_to_tree(
                        a_scores[:seq_len, :seq_len],
                        d_scores[:seq_len, :seq_len, :],
                        p_scores[:seq_len, :],
                        sent_id=sent_id,
                        greedy=False,
                        root_in_form=False,
                        form=form,
                        reverse_pos_dict=reverse_pos_dict,
                        reverse_dep_label_dict=reverse_dep_label_dict,
                    )
                    self.result_trees.append(tree)
        return loss

    # --------------------------------------------------------------------------------------#

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def count_param_module(self):
        for key, value in self.modules.items():
            print(key)
            print(sum(p.numel() for p in value.parameters()))

    @property
    def _optimizer_step_limit_exceeded(self):
        return (
            self.optimizer_step_limit is not None
            and self.optimizer_step >= self.optimizer_step_limit
        )

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.
        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:
        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``
        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.
        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not (
            isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:
            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    if self._optimizer_step_limit_exceeded:
                        logger.info("Train iteration limit exceeded")
                        break
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Do validation stage only 1/freq_valid times
            try:
                freq_valid = self.hparams.freq_valid
            except AttributeError:
                freq_valid = 1
            if freq_valid != 1 and epoch % self.hparams.freq_valid != 0:
                print(
                    f" skipping valid stage for epoch {epoch} (only doing 1/{freq_valid} epochs)"
                )
                continue
            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(loss, avg_valid_loss)

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[sb.Stage.VALID, avg_valid_loss, epoch],
                    )
            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.
        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            ckpt = self.checkpointer.recover_if_possible(
                # Only modification is : load which module are activated from checkpoint
                device=torch.device(self.device)
            )
            if ckpt is not None:
                self.is_training_syntax = ckpt.meta["is_training_syntax"]

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``
        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.
        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """
        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            ckpt = self.checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )
            if ckpt is not None:
                self.is_training_syntax = ckpt.meta["is_training_syntax"]

    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        self.checkpointer.save_and_keep_only(
            end_of_epoch=False,
            num_to_keep=1,
            ckpt_predicate=lambda c: INTRA_EPOCH_CKPT_FLAG in c.meta,
            meta={
                INTRA_EPOCH_CKPT_FLAG: True,
                "is_training_syntax": self.is_training_syntax,
            },
            verbosity=logging.DEBUG,
        )

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.model_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()
            if self.check_gradients(loss):
                self.wav2vec_optimizer.step()
                self.model_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.dep2Label_metrics = self.hparams.dep2Label_computer()
            self.gov2Label_metrics = self.hparams.gov2Label_computer()
            self.stage_wer_details = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            # Activate syntax module for training once ASR is good enough
            if self.is_training_syntax:
                if stage == sb.Stage.VALID:
                    st = self.hparams.dev_output_conllu
                    goldPath_CoNLLU = self.hparams.dev_gold_conllu
                    alig_file = self.hparams.alig_path + "_valid"
                    order = self.dev_order
                else:
                    st = self.hparams.test_output_conllu
                    goldPath_CoNLLU = self.hparams.test_gold_conllu
                    alig_file = self.hparams.alig_path + "_test"
                    order = self.test_order

                # write accumulated wer_details.
                self.stage_wer_details = natsorted(
                    self.stage_wer_details, key=itemgetter(*["key"])
                )
                with open(alig_file, "w", encoding="utf-8") as f_out:
                    print_alignments(self.stage_wer_details, file=f_out)
                self._convert_tree_to_conllu()
                with open(st, "w", encoding="utf-8") as f_v:
                    write_token_dict_conllu(self.data_valid, f_v, order=order)
                self.result_trees = []
                self.data_valid = {}

                d = types.SimpleNamespace()
                d.system_file = st
                d.gold_file = goldPath_CoNLLU
                d.alignment_file = alig_file
                metrics_dict = self.hparams.evaluator.evaluate_wrapper(d)

                stage_stats["LAS"] = metrics_dict["LAS"].f1 * 100
                stage_stats["UAS"] = metrics_dict["UAS"].f1 * 100
                stage_stats["SER"] = metrics_dict["seg_error_rate"].precision * 100
                stage_stats["SENTENCES"] = metrics_dict["Sentences"].precision * 100
                stage_stats["Tokens"] = metrics_dict["Tokens"].precision * 100
                stage_stats["UPOS"] = metrics_dict["UPOS"].f1 * 100
            try:
                start_syntax_wer = self.hparams.start_syntax_WER
            except:
                start_syntax_wer = 50
            if stage_stats["WER"] < start_syntax_wer and not self.is_training_syntax:
                print("activating training syntax")
                self.is_training_syntax = True
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr_model)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}  # fuse dict
            wandb.log(wandb_stats)
            if self.is_training_syntax and "LAS" in stage_stats.keys():
                self.checkpointer.save_and_keep_only(
                    meta={
                        "WER": stage_stats["WER"],
                        "is_training_syntax": self.is_training_syntax,
                        "LAS": stage_stats["LAS"],
                    },
                    max_keys=["LAS"],
                )
            else:
                self.checkpointer.save_and_keep_only(
                    meta={
                        "WER": stage_stats["WER"],
                        "is_training_syntax": self.is_training_syntax,
                        "LAS": 0,
                    },
                    min_keys=["WER"],
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    # to debug
    def transcribe_dataset(
        self,
        dataset,  # Must be obtained from the dataio_function
        min_key,  # We load the model with the lowest WER
        loader_kwargs,  # opts for the dataloading
        max_key=None,
    ):
        # If dataset isn't a Dataloader, we create it.
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(dataset, sb.Stage.TEST, **loader_kwargs)
        # loading best model
        self.on_evaluate_start(min_key=min_key, max_key=max_key)
        self.wer_metric = self.hparams.error_rate_computer()
        # eval mode (remove dropout etc)
        self.modules.eval()

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            WER = []
            for batch in tqdm(dataset, dynamic_ncols=True):

                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied
                # in compute_forward().
                ids = batch.id
                predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                (
                    p_ctc,
                    wav_lens,
                    p_depLabel,
                    p_govLabel,
                    p_posLabel,
                    seq_len,
                ) = predictions

                sequence = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index
                )
                # We go from tokens to words.
                predicted_words = self.tokenizer(sequence, task="decode_from_list")
                for i, sent in enumerate(predicted_words):
                    for w in sent:
                        if w == "":
                            predicted_words[i] = [w for w in sent if w != ""]
                            if len(predicted_words[i]) == 0:
                                predicted_words[i].append("EMPTY_ASR")

                tokens, tokens_lens = batch.tokens
                target_words = undo_padding(tokens, tokens_lens)
                target_words = self.tokenizer(target_words, task="decode_from_list")
                wer_details = wer_details_for_batch(
                    ids=ids,
                    refs=target_words,
                    hyps=predicted_words,
                    compute_alignments=True,
                )
                WER.extend(wer_details)
                self.wer_metric.append(ids, predicted_words, target_words)
            st = self.hparams.test_output_conllu

            order = self.test_order

            WER = natsorted(WER, key=itemgetter(*["key"]))
            with open(self.hparams.alig_path + "_test", "w", encoding="utf-8") as f_out:
                print_alignments(WER, file=f_out)

            wer_metric = self.wer_metric.summarize()
            print(f"WER : {wer_metric}")

    def _convert_tree_to_conllu(self):
        for tree in self.result_trees:
            s_id = tree["sent_id"]
            self.data_valid[s_id] = {"sent_id": s_id, "sentence": []}
            for i, (edge, pos, f) in enumerate(
                zip(tree["edges"], tree["pos_tags"], tree["form"]), start=1
            ):
                self.data_valid[s_id]["sentence"].append(
                    {
                        "ID": i,
                        "FORM": f,
                        "UPOS": pos,
                        "HEAD": edge[0],
                        "DEPREL": edge[1],
                    }
                )


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """
        Audio Pipeline
        Parameters
        ----------
        wav : the wav file path

        Returns
        resampled : return the raw signal from the file with the right sampling ( 16Khz)
        -------

        """
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        """
        ASR pipeline
        Parameters
        ----------
        wrd : The word contained in the CSV file

        Returns
        tokens_list : the tokenized word list
        token_bos : the tokenized word begining with BOS tag (thus sentence shifted to the right)
        token_eos : the tokenized word ending with EOS tag
        tokens :  the tokenized word tensor
        -------

        """
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Define dependency pipeline

    @sb.utils.data_pipeline.takes("pos", "gov", "dep")
    @sb.utils.data_pipeline.provides("pos_tokens", "head", "dep_tokens")
    def syntax_pipeline(pos, head, dep):
        poss = pos.split(" ")
        poss.insert(0, "ROOT")
        pos_tokens = torch.tensor([pos_dict.get(p) for p in poss])
        yield pos_tokens
        he = head.split(" ")
        head = [int(h) for h in he]
        head.insert(0, 0)
        yield torch.tensor(head)
        de = dep.split(" ")
        de.insert(0, "ROOT_GRAPH")
        try:
            dep_token = torch.tensor([dep_label_dict.get(d) for d in de])
            yield dep_token
        except RuntimeError as e:
            print(f"{[d for d in de]}")
            print(f"{[dep_label_dict.get(d) for d in de]}")
            raise e

    sb.dataio.dataset.add_dynamic_item(datasets, syntax_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "wrd",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "pos_tokens",
            "head",
            "dep_tokens",
        ],
    )
    return train_data, valid_data, test_data


def get_id_from_CoNLLfile(path):
    """
    Get the sentence id from the conll file in the order
    Will be used to write in the same order for comparaison sakes.
    """
    sent_id = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.startswith("# sent_id"):
                field = line.split("=")
                sent_id.append(field[1].replace(" ", "").replace("\n", ""))
    return sent_id


dep_label_dict = {
    "PERIPH": 0,
    "SUBJ": 1,
    "ROOT": 2,
    "DEP": 3,
    "DM": 4,
    "SPE": 5,
    "MARK": 6,
    "PARA": 7,
    "AUX": 8,
    "DISFLINK": 9,
    "MORPH": 10,
    "PARENTH": 11,
    "AFF": 12,
    "ROOT_GRAPH": 13,
    "__JOKER__": 14,
    "DELETION": 15,
    "INSERTION": 16,
}
reverse_dep_label_dict = {v: k for k, v in dep_label_dict.items()}


pos_dict = {
    "PADDING": 0,
    "ADV": 1,
    "CLS": 2,
    "VRB": 3,
    "PRE": 4,
    "INT": 5,
    "DET": 6,
    "NOM": 7,
    "COO": 8,
    "CLI": 9,
    "ADJ": 10,
    "VNF": 11,
    "CSU": 12,
    "ADN": 13,
    "PRQ": 14,
    "VPP": 15,
    "PRO": 16,
    "NUM": 17,
    "X": 18,
    "CLN": 19,
    "VPR": 20,
    "ROOT": 21,
    "DELETION": 22,
    "INSERTION": 23,
}

reverse_pos_dict = {v: k for k, v in pos_dict.items()}

label_alphabet = [dep_label_dict, pos_dict]
reverse_label_alphabet = [reverse_dep_label_dict, reverse_pos_dict]

if __name__ == "__main__":
    wandb.init(group="Audio")
    print(f"wandb run name : {wandb.run.name}")
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    goldPath_CoNLLU_dev = hparams["dev_gold_conllu"]
    goldPath_CoNLLU_test = hparams["test_gold_conllu"]

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from cefcOrfeo_prepare import prepare_cefcOrfeo  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_cefcOrfeo,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.count_param_module()

    # Adding objects to trainer.
    # doing this to avoid overwriting the class constructor
    asr_brain.tokenizer = tokenizer
    asr_brain.result_trees = []
    asr_brain.data_valid = {}
    asr_brain.hparams.oracle.set_alphabet(label_alphabet, reverse_label_alphabet)
    asr_brain.is_training_syntax = False
    # Diverse information on the data such as PATH and order of sentences.
    asr_brain.dev_order = get_id_from_CoNLLfile(hparams["dev_gold_conllu"])
    asr_brain.test_order = get_id_from_CoNLLfile(hparams["test_gold_conllu"])
    encoding = asr_brain.hparams.encoding
    asr_brain.optimizer_step_limit = None
    # Training
    try:
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["test_dataloader_options"],
        )
    except RuntimeError as e:  # Memory Leak
        import gc

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    pass
                    # print(type(obj), obj.size())
            except:
                pass
        raise RuntimeError() from e
    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

    # transcribe
    """
    run_on_main(
        asr_brain.transcribe_dataset,
        kwargs={
            "dataset": test_data,
            "min_key": "WER",
            "loader_kwargs": hparams["test_dataloader_options"],
        },
    )
    """
