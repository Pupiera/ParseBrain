#!/usr/bin/env/python3
"""Recipe for training a Transducer ASR system with librispeech.
The system employs an encoder, a decoder, and an joint network
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
language model is used on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
"""

import sys
import torch
import logging
import speechbrain as sb
import torchaudio
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import undo_padding
from speechbrain.tokenizers.SentencePiece import SentencePiece
from hyperpyyaml import load_hyperpyyaml

from speechbrain.dataio.wer import print_alignments
from natsort import natsorted
from operator import itemgetter

from torch.nn.utils.rnn import pad_sequence

# --------------------------------------- Alignment import --------------------------------#
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.dataio.wer import print_alignments
from natsort import natsorted
from operator import itemgetter


logger = logging.getLogger(__name__)
INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"



# Define training procedure


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos
        pos_eos, _ = batch.pos_eos
        gov_eos, _ = batch.gov_eos
        dep_eos, _ = batch.dep_eos

        pos_bos, _ = batch.pos_bos
        gov_bos, _ = batch.gov_bos
        dep_bos, _ = batch.dep_bos

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        feats = self.modules.wav2vec2(wavs)

        batch_size = feats.size(0)

        # Add augmentation if specified

        x = self.modules.enc(feats)
        seq_emb = self.hparams.emb_asr(tokens_with_bos)
        dec_asr, _ = self.modules.dec(seq_emb)
        jointASR = self.modules.TjointASR(x.unsqueeze(2), dec_asr.unsqueeze(1))
        logits_transducer_asr = self.modules.transducer_lin(jointASR)
        result_trans = { "logits_transducer": logits_transducer_asr,
                         "wav_lens": wav_lens}
        # teacher forcing in train time.
        if self.is_training_syntax:
            pos_emb = self.hparams.emb_pos(pos_bos)
            #gov_emb = self.hparams.emb_gov(gov_bos)
            #dep_emb = self.hparams.emb_dep(dep_bos)

            # max_word_len = synt_emb.size(1)
            # max_seq_len = seq_emb.size(1)

            # keep only the hidden from the rnns
            dec_pos, _ = self.modules.dec_pos(pos_emb)
            #dec_gov, _ = self.modules.dec_gov(gov_emb)
            #dec_dep, _ = self.modules.dec_dep(dep_emb)

            # not this easy for the teacher forcing, because shape synt_emb != seq_emb
            # need to give same syntaxic emb to the same word even if it's divised in subword.
            # the joint syntax module crash.
            # through maybe needed to combine the decoder...

            # Joint network
            # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
            # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]

            jointPOS = self.modules.TjointPOS(x.unsqueeze(2), dec_pos.unsqueeze(1))
            #jointGOV = self.modules.TjointGOV(x.unsqueeze(2), dec_gov.unsqueeze(1))
            #jointDEP = self.modules.TjointDEP(x.unsqueeze(2), dec_dep.unsqueeze(1))

            # joint ASR and syntax predictions influenced from previous word and syntax.
            # joint shape [B, U, T, H]

            # Output layer for transducer log-probabilities
            logits_transducer_pos = self.modules.transducer_pos(jointPOS)
            #logits_transducer_gov = self.modules.transducer_gov(jointGOV)
            #logits_transducer_dep = self.modules.transducer_dep(jointDEP)

            if hasattr(self.hparams, "dec_pos_lin"):
                #auxiliary task
                logits_dec_pos = self.modules.dec_pos_lin(dec_pos)
                p_dec_pos = self.hparams.log_softmax(logits_dec_pos)
                #logits_dec_gov = self.modules.dec_pos_lin(dec_gov)
                #p_dec_gov = self.hparams.log_softmax(logits_dec_gov)
                #logits_dec_dep = self.modules.dec_pos_lin(dec_dep)
                #p_dec_dep = self.hparams.log_softmax(logits_dec_dep)
                result_parsing_dec = {"p_dec_pos": p_dec_pos,
                                      #"p_dec_gov": p_dec_gov,
                                      #"p_dec_dep": p_dec_dep
                                      }
                result_trans = {**result_trans, **result_parsing_dec}

            # shape of logits_transducer is [batch, time_audio, seq_len, output_task]
            # not sure if the softmaxed one are needed ? For
            p_posLabel = self.hparams.log_softmax(logits_transducer_pos)
            #p_govLabel = self.hparams.log_softmax(logits_transducer_gov)
            #p_depLabel = self.hparams.log_softmax(logits_transducer_dep)

            result_parsing = {
                "logits_transducer_pos": logits_transducer_pos,
             #   "logits_transducer_gov": logits_transducer_gov,
             #   "logits_transducer_dep": logits_transducer_dep,
                "p_posLabel": p_posLabel,
             #   "p_depLabel": p_depLabel,
             #   "p_govLabel": p_govLabel,
                # "seq_len": seq_len,
            }
            result_trans = {**result_trans, **result_parsing}

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            # Probably needed for oracle. Need to make custom multitask tranduscer_beamsearch
            if self.is_training_syntax:
                pass
                # hyps = self.hparams.greedy_beam_searcher(x)
                # result_trans["predicted_tokens"] = hyps
            return_CTC = False
            return_CE = False
            current_epoch = self.hparams.epoch_counter.current
            if (
                    hasattr(self.hparams, "ctc_cost")
                    and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                return_CTC = True
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.enc_lin(x)
                p_ctc = self.hparams.log_softmax(out_ctc)
            if (
                    hasattr(self.hparams, "ce_cost")
                    and current_epoch <= self.hparams.number_of_ce_epochs
            ):
                return_CE = True
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(dec_asr)
                p_ce = self.hparams.log_softmax(p_ce)
            if return_CE and return_CTC:
                result_asr = {"p_ctc": p_ctc, "p_ce": p_ce}
                result = {**result_trans, **result_asr}
            elif return_CTC:
                result_asr = {"p_ctc": p_ctc}
                result = {**result_trans, **result_asr}
            elif return_CE:
                result_asr = {"p_ce": p_ce}
                result = {**result_trans, **result_asr}
            else:
                result = result_trans
            return result

        elif stage != sb.Stage.TRAIN:
            best_hyps_ASR = self.hparams.beam_searcherASR(x)
            if self.is_training_syntax:
                best_hyps_POS = self.hparams.beam_searcherPOS(x)
                #best_hyps_GOV = self.hparams.beam_searcherGOV(x)
                #best_hyps_DEP = self.hparams.beam_searcherDEP(x)

                best_hyps = []
                print(f"best_hyps_ASR {best_hyps_ASR}")

                for i in range(batch_size):
                    hyp = {}
                    hyp['-asr'] = best_hyps_ASR[0][i]
                    hyp['-pos'] = best_hyps_POS[0][i]
                    #hyp['-gov'] = best_hyps_GOV[0][i]
                    #hyp['-dep'] = best_hyps_DEP[0][i]
                    best_hyps.append(hyp)
                print(f"best_hyps {best_hyps}")
            else:
                best_hyps = []
                for i in range(batch_size):
                    hyp = {}
                    hyp['-asr'] = best_hyps_ASR[0][i]
                    best_hyps.append(hyp)
            result_beam = {"predicted_tokens": best_hyps}
            result = {**result_trans, **result_beam}

            return result

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
            if ckpt is not None and "is_training_syntax" in ckpt.meta.keys():
                self.is_training_syntax = ckpt.meta["is_training_syntax"]
                print(f"loading ckpt with syntax training: {self.is_training_syntax}")


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


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""

        ids = batch.id
        current_epoch = self.hparams.epoch_counter.current
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos
        batch_size = tokens.size(0)

        if stage == sb.Stage.TRAIN:
            if "p_ctc" in predictions.keys() and "p_ce" in predictions.keys():
                CTC_loss = self.hparams.ctc_cost(predictions["p_ctc"], tokens, predictions["wav_lens"], token_lens)
                CE_loss = self.hparams.ce_cost(predictions["p_ce"], tokens_eos, length=token_eos_lens)
                loss_transducer = self.hparams.transducer_cost(
                    predictions["logits_transducer"], tokens, predictions["wav_lens"], token_lens
                )
                loss = (
                        self.hparams.ctc_weight * CTC_loss
                        + self.hparams.ce_weight * CE_loss
                        + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                        * loss_transducer
                )
            elif "p_ctc" in predictions.keys() or "p_ce" in predictions.keys():
                # one of the 2 heads (CTC or CE) is still computed
                # CTC alive
                if current_epoch <= self.hparams.number_of_ctc_epochs:
                    CTC_loss = self.hparams.ctc_cost(
                        predictions["p_ctc"], tokens, predictions["wav_lens"], token_lens
                    )
                    loss_transducer = self.hparams.transducer_cost(
                        predictions["logits_transducer"], tokens, predictions["wav_lens"], token_lens
                    )
                    loss = (
                            self.hparams.ctc_weight * CTC_loss
                            + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
                # CE for decoder alive
                else:
                    CE_loss = self.hparams.ce_cost(
                        predictions["p_ce"], tokens_eos, length=token_eos_lens
                    )
                    # Transducer loss use logits from RNN-T model.
                    loss_transducer = self.hparams.transducer_cost(
                        predictions["logits_transducer"], tokens, predictions["wav_lens"], token_lens
                    )
                    loss = (
                            self.hparams.ce_weight * CE_loss
                            + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
            else:
                # Transducer loss use logits from RNN-T model.
                loss = self.hparams.transducer_cost(
                    predictions["logits_transducer"], tokens, predictions["wav_lens"], token_lens
                )
        else:
            # Transducer loss use logits from RNN-T model.
            loss = self.hparams.transducer_cost(
                predictions["logits_transducer"], tokens, predictions["wav_lens"], token_lens
            )

        if stage != sb.Stage.TRAIN:

            # Decode token terms to words
            print(f'predictions["predicted_tokens"] {predictions["predicted_tokens"]}')
            pred_asr = [x['-asr'] for x in predictions["predicted_tokens"]]
            predicted_words = self.tokenizer(pred_asr, task="decode_from_list")
            for i in range(batch_size):
                # split_words = predicted_words[i].split(" ")
                len_pred_word = len(predicted_words[i])
                if len_pred_word == 1 and predicted_words[i][0] == '':
                    len_pred_word = 0
                if self.is_training_syntax:
                    len_pred = min([len_pred_word, len(predictions["predicted_tokens"][i]['-pos']),
                                    #len(predictions["predicted_tokens"][i]['-gov']),
                                    #len(predictions["predicted_tokens"][i]['-dep'])
                                    ])
                else:
                    len_pred = len_pred_word
                # predicted_words[i] = " ".join(split_words[:len_pred])
                predicted_words[i] = predicted_words[i][:len_pred]
                if self.is_training_syntax:
                    predictions["predicted_tokens"][i]['-pos'] = predictions["predicted_tokens"][i]['-pos'][:len_pred]
                    #predictions["predicted_tokens"][i]['-gov'] = predictions["predicted_tokens"][i]['-gov'][:len_pred]
                    #predictions["predicted_tokens"][i]['-dep'] = predictions["predicted_tokens"][i]['-dep'][:len_pred]
            # Convert indices to words
            for i, p_w in enumerate(predicted_words):
                if len(p_w) == 0:
                    predicted_words[i].append('EMPTY_ASR')
            target_words = undo_padding(tokens, token_lens)
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
            if stage != sb.Stage.TRAIN:        
                loss_parsing = self.compute_objectives_parsing(predictions, batch, stage, predicted_words)
            else:
                loss_parsing = self.compute_objectives_parsing(predictions, batch, stage)
            loss = self.hparams.loss_weight_asr* loss + self.hparams.loss_weight_parsing* loss_parsing

        return loss

    def truncate_logits(self, logits, subword):
        '''
        This function truncate the U from the ASR part of the model.
        So that there is one representation corresponding to one supervision for word level task
        this should always be of the same length as the word level supervision. 
        When using teacher forcing.
        '''
        truncated_logits_transducer = torch.clone(logits)
        result = []
        padding_begining = []
        for batch_element, sub in zip(truncated_logits_transducer, subword):
            indices_to_filter = []
            i = 0
            for s in sub:
                s = s.item()
                print(i)
                print(f"s {s}")
                if s > 1:
                    #delete this many row, excluding one.
                    indices_to_filter.extend(list(range(i, i+s-1)))
                    #may need to check the value of i
                    i = i+s-1
                #remove the padding
                if s == 0:
                    indices_to_filter.extend(list(range(i, batch_element.shape[1])))
                    break
                i+=1
            if len(indices_to_filter) >=1:
                print(sub)
                print(indices_to_filter)
                mask = torch.ones_like(batch_element, dtype=torch.bool)
                mask[:, indices_to_filter, :] = False
                print(mask)
                print(mask.shape)
                t = batch_element.shape[0]
                dim_out = batch_element.shape[-1]
                batch_element = batch_element[mask].view(t, -1, dim_out)
                #batch_element = batch_element.view(t, -1, dim_out)
                print(batch_element.shape)
            result.append(batch_element)
        print([x.shape for x in result])
        result = pad_sequence([tensor.transpose(0,1) for tensor in result ], batch_first=True, padding_value=0)
        return result.transpose(1,2)




    def compute_objectives_parsing(self, predictions, batch, stage, predicted_words=None):

        pos, pos_lens = batch.pos_tensor
        print(f"gold pos {pos}")
        print(torch.argmax(predictions["logits_transducer_pos"]))
        gov, gov_lens = batch.govDep2Label
        dep, dep_lens = batch.depDep2Label

        # not sure if right value ?
        # maybe need to adapt the supervision to be of same shape.
        tokens, token_lens = batch.tokens
        # removing the bos of the number of subdword count
        subword_count, _ = batch.subword_count_bos
        subword_count = subword_count[:, 1:]

        batch_size = pos.size(0)
        max_word_len = tokens.size(1)

        if hasattr(self.hparams, "dec_pos_lin"):
            #auxiliary loss
            loss_dec_pos = self.hparams.ce_cost(predictions['p_dec_pos'], pos, length=token_lens)
            #loss_dec_gov = self.hparams.ce_cost(predictions['p_dec_gov'], gov, length=token_lens)
            #loss_dec_dep = self.hparams.ce_cost(predictions['p_dec_dep'], dep, length=token_lens)
            #loss_dec = (loss_dec_pos + loss_dec_gov + loss_dec_dep)/3
            loss_dec = loss_dec_pos
        
        if hasattr(self.hparams, "loss_align"):
            # loss align between word piece and other task
            # 1 combined matrice per task, (word, task)
            print(predictions["logits_transducer"].shape)
            print(predictions["logits_transducer_pos"].shape)
            print(subword_count)
            # the length should be correct because of the teacher forcing.
            # so need to deal with subword. (duplicate correct POS  for subword ? OR have a special POS for subword ?) Or only keep 1 subword.
            truncated_logits_transducer = self.truncate_logits(predictions["logits_transducer"], subword_count)
            
            try:
                combined_pos_lattice = truncated_logits_transducer.unsqueeze(-1) * predictions["logits_transducer_pos"].unsqueeze(-2)
            # shoud be of shape [B, U, T, voc_w * voc_pos]
                combined_pos_lattice = combined_pos_lattice.view(batch_size, combined_pos_lattice.shape[1], combined_pos_lattice.shape[2], -1).contiguous()
            except RuntimeError as e:
                print("TRUNC ASR")
                print(truncated_logits_transducer.shape)
                print(truncated_logits_transducer)
                print("POS")
                print(predictions["logits_transducer_pos"].shape)
                print(predictions["logits_transducer_pos"])

                raise e

            # the supervision should be voc_w_int * voc_s + voc_s_int ?
            # Need to deal with case where one word has multiple subword.
            # Need to have the information on wheter the word is over or not. 
            # to be able to go to the next part of speech.
            align_pos_supervision = []
            for sentence, poss in zip(batch.tokens.data, pos):
                pos_indice = 0
                a_p_s = []
                for w in sentence:
                    if w in list_final_subtokens['-asr']:
                        a_p_s.append(w * self.hparams.output_neurons_pos + poss[pos_indice])
                        pos_indice+=1
                align_pos_supervision.append(torch.tensor(a_p_s))
            align_pos_supervision = pad_sequence(align_pos_supervision, batch_first=True)
            align_pos_supervision = align_pos_supervision.to(self.device)
            print("ALIGN SHAPE")
            print(combined_pos_lattice.shape)
            print(align_pos_supervision.shape)
            loss_align_pos = self.hparams.transducer_cost_pos(
                    combined_pos_lattice, align_pos_supervision, predictions["wav_lens"], pos_lens)
        
        loss_transducer_pos = self.hparams.transducer_cost_pos(
            predictions["logits_transducer_pos"], pos, predictions["wav_lens"], pos_lens
        )
        '''
        loss_transducer_gov = self.hparams.transducer_cost_gov(
            predictions["logits_transducer_gov"], gov, predictions["wav_lens"], gov_lens
        )
        loss_transducer_dep = self.hparams.transducer_cost_dep(
            predictions["logits_transducer_dep"], dep, predictions["wav_lens"], dep_lens
        )
        '''

        if stage != sb.Stage.TRAIN:
            ids = batch.id
            #pred_gov = [x['-gov'] for x in predictions['predicted_tokens']]
            pred_gov = []
            #pred_dep = [x['-dep'] for x in predictions['predicted_tokens']]
            pred_dep = []
            pred_pos = [x['-pos'] for x in predictions['predicted_tokens']]

            for x in predictions['predicted_tokens']:
                length = len(x['-pos'])
                pred_gov.append([1 for _ in range(length)])
                pred_dep.append([1 for _ in range(length)])


            #we need at least one element for the conllu eval
            for i, p_w in enumerate(predicted_words):
                if len(p_w) == 0:
                    predicted_words[i].append('EMPTY_ASR')

            for i, p_gov in enumerate(pred_gov):
                if len(p_gov) == 0:
                    pred_gov[i].append(1)

            for i, p_dep in enumerate(pred_dep):
                if len(p_dep) == 0:
                    pred_dep[i].append(1)

            for i, p_pos in enumerate(pred_pos):
                if len(p_pos) == 0:
                    pred_pos[i].append(1)

            print(predicted_words)
            for i, pred in enumerate(predicted_words):
                for y, p in enumerate(pred):
                    if p =='':
                        pred[y] = 'EMPTY_FORM'
                predicted_words[i] = pred

            self.hparams.evaluator.decode(
                [
                    pred_gov,
                    pred_dep,
                    pred_pos,
                ],
                predicted_words,
                ids,
                prob=False,
            )

        loss = self.hparams.pos_weight * loss_transducer_pos 
        #+ self.hparams.gov_weight *loss_transducer_gov 
        #+ self.hparams.dep_weight * loss_transducer_dep
        if hasattr(self.hparams, "dec_pos_lin"):
            loss += self.hparams.auxiliary_weight * loss_dec
        return loss

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
            self.stage_wer_details = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
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

                self.hparams.evaluator.writeToCoNLLU(st, order)

                # write accumulated wer_details.
                self.stage_wer_details = natsorted(
                    self.stage_wer_details, key=itemgetter(*["key"])
                )
                with open(alig_file, "w", encoding="utf-8") as f_out:
                    print_alignments(self.stage_wer_details, file=f_out)

                metrics_dict = self.hparams.evaluator.evaluateCoNLLU(
                    goldPath_CoNLLU, st, alig_file
                )
                stage_stats["LAS"] = metrics_dict["LAS"].f1 * 100
                stage_stats["UAS"] = metrics_dict["UAS"].f1 * 100
                stage_stats["SER"] = metrics_dict["seg_error_rate"].precision * 100
                stage_stats["SENTENCES"] = metrics_dict["Sentences"].precision * 100
                stage_stats["Tokens"] = metrics_dict["Tokens"].precision * 100
                stage_stats["UPOS"] = metrics_dict["UPOS"].f1 * 100
            if not self.is_training_syntax and stage_stats["WER"] <50:
                print("activating syntax learning")
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
            if "LAS" in stage_stats.keys():
                self.checkpointer.save_and_keep_only(
                    meta={"WER": stage_stats["WER"],
                         "LAS": stage_stats["LAS"],
                         "is_training_syntax": self.is_training_syntax
                         },
                    max_keys=["LAS"],
                )
            else:
                self.checkpointer.save_and_keep_only(
                    meta={"WER": stage_stats["WER"],
                         "is_training_syntax": self.is_training_syntax
                        },
                    min_keys=["WER"],
                )
            with open(self.hparams.wer_file_valid, "w") as w:
                self.wer_metric.write_stats(w)
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


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
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_shorter_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

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

    # We also sort the test data so it is faster to test
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
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
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "subword_count_bos"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["blank_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["blank_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        subword_bos_len = [1] + [
            len(tokenizer.sp.encode_as_ids(w)) for w in wrd.split(" ")
        ]
        yield torch.LongTensor(subword_bos_len)

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Define dependency pipeline

    @sb.utils.data_pipeline.takes("wrd", "pos", "gov", "dep")
    @sb.utils.data_pipeline.provides(
        "wrd",
        "pos_tensor",
        "pos_bos",
        "pos_eos",
        "govDep2Label",
        "gov_bos",
        "gov_eos",
        "depDep2Label",
        "dep_bos",
        "dep_eos",
    )
    def dep2label_pipeline(wrd, poss, gov, dep):
        """
        The dependecy parsing pipeline.
        Parameters
        ----------
        wrd : the raw word
        poss: the part of speech in the csv file
        gov : the gov/head label in the csv file
        dep : the syntactic function in the csv file

        Returns
        wrd: the raw word
        pos_list : the tensor of labeled part of speech
        govDep2label: the relative encoding of head/gov labeled
        depDep2Label: the labeled syntactic function
        -------

        """
        yield wrd
        # 3 task is POS tagging
        try:
            pos_list = [label_alphabet[2].get(p) for p in poss.split(" ")]
        except TypeError:
            print(wrd)
            print(poss)
            print([label_alphabet[2].get(p) for p in poss.split(" ")])
        yield torch.LongTensor(pos_list)
        yield torch.LongTensor([label_alphabet[2]["-BOS-"]] + pos_list)
        # yield torch.LongTensor([hparams["pos_bos"]] + pos_list)
        yield torch.LongTensor(pos_list + [hparams["pos_eos"]])

        fullLabel = encoding.encodeFromList(
            [w for w in wrd.split(" ")],
            [p for p in poss.split(" ")],
            [g for g in gov.split(" ")],
            [d for d in dep.split(" ")],
        )
        # first task is gov pos and relative position
        try:
            gov_list = [
                label_alphabet[0].get(fl.split("\t")[-1].split("{}")[0])
                for fl in fullLabel
            ]
            yield torch.LongTensor(gov_list)
            yield torch.LongTensor([label_alphabet[0]["-BOS-@-BOS-"]] + gov_list)
            # yield torch.LongTensor([hparams["gov_bos"]] + gov_list)
            yield torch.LongTensor(gov_list + [hparams["gov_eos"]])

        except TypeError as e:
            print(wrd)
            print([fl.split("\t")[-1].split("{}")[0] for fl in fullLabel])
            print(
                [
                    label_alphabet[0].get(fl.split("\t")[-1].split("{}")[0])
                    for fl in fullLabel
                ]
            )
            raise TypeError() from e
        # second task is dependency type
        dep_list = [label_alphabet[1].get(fl.split("{}")[1]) for fl in fullLabel]
        yield torch.LongTensor(dep_list)
        yield torch.LongTensor([label_alphabet[1]["-BOS-"]] + dep_list)
        # yield torch.LongTensor([hparams["dep_bos"]] + dep_list)
        yield torch.LongTensor(dep_list + [hparams["dep_eos"]])

    sb.dataio.dataset.add_dynamic_item(datasets, dep2label_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "subword_count_bos",
            "wrd",
            "pos_tensor",
            "pos_bos",
            "pos_eos",
            "govDep2Label",
            "gov_bos",
            "gov_eos",
            "depDep2Label",
            "dep_bos",
            "dep_eos",
        ],
    )
    return train_data, valid_data, test_data


def build_label_alphabet(path_encoded_train):
    label_gov = {"<unk>": 0}
    label_dep = {"<unk>": 0}
    label_pos = {"<unk>": 0}
    with open(path_encoded_train, "r", encoding="utf-8") as inputFile:
        for line in inputFile:
            field = line.split("\t")
            if len(field) > 1:
                fullLabel = field[-1]
                labelSplit = fullLabel.split("{}")
                govLabel = labelSplit[0]
                if govLabel not in label_gov:
                    label_gov[govLabel] = len(label_gov)
                depLabel = labelSplit[-1].replace("\n", "")
                if depLabel not in label_dep:
                    label_dep[depLabel] = len(label_dep)
                pos = field[1]
                if pos not in label_pos:
                    label_pos[pos] = len(label_pos)
    for pos_key in label_pos:
        for i in range(1, 20):
            key = f"{i}@{pos_key}"
            if "+" + key not in label_gov.keys():
                label_gov["+" + key] = len(label_gov)
            if "-" + key not in label_gov.keys():
                label_gov["-" + key] = len(label_gov)
    label_gov["-1@INSERTION"] = len(label_gov)
    label_gov["-1@DELETION"] = len(label_gov)
    label_dep["INSERTION"] = len(label_dep)
    label_dep["DELETION"] = len(label_dep)
    label_pos["INSERTION"] = len(label_pos)
    return [label_gov, label_dep, label_pos]


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


def build_reverse_alphabet(alphabet):
    reverse = []
    for alpha in alphabet:
        reverse.append({item: key for (key, item) in alpha.items()})
    return reverse


def get_final_subtokens(tokenizer):
    num_subtokens = tokenizer.sp.get_piece_size()
    final_subtokens = []
    for i in range(num_subtokens):
        sub_tok = tokenizer.sp.id_to_piece(i)
        print(sub_tok)
        if "▁" in sub_tok:
            print(f"added {i}")
            final_subtokens.append(i)
    return final_subtokens


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    path_encoded_train = "all.seq"  # For alphabet generation
    label_alphabet = build_label_alphabet(path_encoded_train)
    reverse_label_alphabet = build_reverse_alphabet(label_alphabet)

    # If --distributed_launch then
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

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    encoding = asr_brain.hparams.encoding
    # asr_brain.hparams.oracle.set_alphabet(label_alphabet, reverse_label_alphabet)
    asr_brain.is_training_syntax = False
    asr_brain.hparams.evaluator.set_alphabet(label_alphabet)
    # Diverse information on the data such as PATH and order of sentences.
    asr_brain.dev_order = get_id_from_CoNLLfile(hparams["dev_gold_conllu"])
    asr_brain.test_order = get_id_from_CoNLLfile(hparams["test_gold_conllu"])
    asr_brain.tokenizer = tokenizer

    list_final_subtokens = {}
    list_final_subtokens['-asr'] = get_final_subtokens(tokenizer)
    list_final_subtokens['-pos'] = range(1, len(label_alphabet[2]))
    list_final_subtokens['-gov'] = range(1, len(label_alphabet[0]))
    list_final_subtokens['-dep'] = range(1, len(label_alphabet[1]))


    # Training
    with torch.autograd.set_detect_anomaly(True):
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
