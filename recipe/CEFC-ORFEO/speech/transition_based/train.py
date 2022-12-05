"""
Recipe for transition based parsing on the CEFC-ORFEO dataset.
authors : Adrien PUPIER
"""
import sys
import types
import logging
import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.data_utils import undo_padding
from speechbrain.tokenizers.SentencePiece import SentencePiece
import torchaudio

import parsebrain as pb  # extension of speechbrain
import wandb
from debug_utils import plot_grad_flow
from parsebrain.dataio.pred_to_file.pred_to_conllu import write_token_dict_conllu

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
    GoldConfigurationASR,
    Word,
)

from parsebrain.speechbrain_custom.decoders.ctc import ctc_greedy_decode

from torch.nn.utils.rnn import pad_sequence

INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"

# @export
# @profile_optimiser
class Parser(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """
        Do ASR (CTC)
        Create Audio word Embedding from ASR and wav2vec2 rep
        From the ASR and the WER information (sub/ins/del) create new gold_config
        Create the config using the word audio embedding.
        Use the parser as usual.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        sent_ids = batch.id
        tokens_bos, _ = batch.tokens_bos
        tokens, tokens_lens = batch.tokens

        result = {"wav_lens": wav_lens}

        # ASR
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass ASR
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)  # [batch,time,1024]
        logits = self.modules.ctc_lin(x)  # [batch,time,76]
        p_ctc = self.hparams.log_softmax(
            logits
        )  # [batch,time,76] (76 in output neurons corresponding to BPE size)
        result["p_ctc"] = p_ctc
        # Syntax
        if self.is_training_syntax:
            sequence, mapFrameToWord = ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            input_dep, seq_len = self._create_inputDep(x, mapFrameToWord)
            result["seq_len"] = seq_len
            input_dep = input_dep.to(self.device)

            batch_len = input_dep.shape[0]
            hidden = torch.zeros(
                4, batch_len, self.hparams.repFusionHidden, device=self.device
            )  # init hidden to zeros for each sentence
            cell = torch.zeros(
                4, batch_len, self.hparams.repFusionHidden, device=self.device
            )
            lstm_out, _ = self.modules.feat(
                input_dep, (hidden, cell)
            )  # Contextualize the rep.
            pos_log_prob = self.hparams.neural_network_POS(lstm_out)
            result["pos_log_prob"] = pos_log_prob

            config = []
            # get ASR words.
            predicted_words = self.tokenizer(sequence, task="decode_from_list")
            result["predicted_words"] = predicted_words
            for ba_lstm, p_words in zip(lstm_out, predicted_words):
                config.append(Configuration(ba_lstm, self.create_words_list(p_words)))

            # if train compute gold_config based on ASR output and gold parse.
            gold_config = []
            # Get WER alignment
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")
            result["target_words"] = target_words
            wer_details = wer_details_for_batch(
                ids=sent_ids,
                hyps=predicted_words,
                refs=target_words,
                compute_alignments=True,
            )
            head_tokens, head_length = batch.head_tokens
            print(batch.dep_tokens)
            dep_tokens = batch.dep_tokens
            for w_d, gov, label in zip(wer_details, head_tokens, dep_tokens):
                print(gov)
                print(label)
                print(w_d["alignment"])
                gold_config.append(GoldConfigurationASR(w_d["alignment"], gov, label))

            if stage == sb.Stage.TRAIN:
                static = (
                    self.hparams.number_of_epochs_static
                    >= self.hparams.epoch_counter.current
                )
                parsing_dict = self.hparams.parser.parse(
                    config, stage, gold_config, static=static
                )
            else:
                parsing_dict = self.hparams.parser.parse(
                    config, stage, None, static=False
                )
            result["parsing_dict"] = parsing_dict
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
        # for 1 element on the batch do :
        for i, (rep, map) in enumerate(zip(x, mapFrameToWord)):
            map = torch.Tensor(map)
            uniq = torch.unique(map)
            fusionedRep = []
            # init hidden to zeros for each sentence
            hidden = torch.zeros(nb_hidden, 1, hidden_size, device=self.device)
            # init cell to zeros for each sentence
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

    def compute_objectives(self, predictions, batch, stage):

        # compute loss : Need to compute predictions (list of gold transitions)
        result = predictions
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        # ASR loss
        loss = self.hparams.ctc_cost(
            result["p_ctc"], tokens, result["wav_lens"], tokens_lens
        )
        if sb.Stage.TRAIN != stage:
            ids = batch.id
            if result.get("predicted_words") is None:
                sequence = sb.decoders.ctc_greedy_decode(
                    result["p_ctc"],
                    result["wav_lens"],
                    blank_id=self.hparams.blank_index,
                )
                predicted_words = self.tokenizer(sequence, task="decode_from_list")
                result["predicted_words"] = predicted_words

                # Convert indices to words
                target_words = undo_padding(tokens, tokens_lens)
                target_words = self.tokenizer(target_words, task="decode_from_list")
                result["target_words"] = target_words

            self.wer_metric.append(
                ids, result["predicted_words"], result["target_words"]
            )
            self.cer_metric.append(
                ids, result["predicted_words"], result["target_words"]
            )
        if self.is_training_syntax:
            # gold config contains adapted supervision computed from ASR and gold parse (only in train)
            loss = loss * self.hparams.ctc_cost + self.compute_objective_parsing(
                predictions, batch, stage
            )
        return loss

    def compute_objective_parsing(self, predictions, batch, stage):
        result = predictions
        words = batch.words
        pos = batch.pos_tokens.data.to(self.device)
        sent_ids = batch.sent_id
        parsing_dict = result["parsing_dict"]

        loss = self.hparams.pos_cost_weight * self.hparams.pos_cost(
            result["pos_log_prob"], pos, length=result["seq_len"]
        )
        self.acc_dyna.append(
            parsing_dict["parse_log_prob"],
            parsing_dict["oracle_parsing"],
            parsing_dict["oracle_parse_len"],
        )

        loss += self.hparams.parse_cost_weight * self.hparams.parse_cost(
            parsing_dict["parse_log_prob"],
            parsing_dict["oracle_parsing"],
            parsing_dict["oracle_parse_len"],
        )
        # Compute the loss for each element based on decision and only keep relevant one.
        # Allow to compute label in a batch way.
        loss += self.hparams.label_cost_weight * self.hparams.label_cost(
            parsing_dict["label_log_prob"],
            parsing_dict["oracle_label"],
            parsing_dict["oracle_label_len"],
        )
        # Populate the list that will be written at the end of the stage.
        if sb.Stage.VALID == stage:
            predicted_pos = [
                [reverse_pos_dict.get(p.item()) for p in poss]
                for poss in torch.argmax(result["pos_log_prob"], dim=-1)
            ]
            self._create_data_from_parsed_tree(
                parsing_dict["parsed_tree"], sent_ids, words, predicted_pos
            )
        return loss

    def _create_data_from_parsed_tree(self, parsed_tree, sent_ids, words, pos):
        for p_t, sent, po, sent_id in zip(parsed_tree, words, pos, sent_ids):
            self.data_valid.append({"sent_id": sent_id, "sentence": []})
            root = None
            for i in range(len(sent)):
                if i + 1 in p_t.keys():
                    self.data_valid[-1]["sentence"].append(
                        {
                            "ID": i + 1,
                            "FORM": sent[i],
                            "UPOS": po[i],
                            "HEAD": p_t[i + 1]["head"],
                            "DEPREL": reverse_dep_label_dict.get(
                                p_t[i + 1]["label"], "UNDEFINED"
                            ),
                        }
                    )
                elif root is None:
                    # If no head, this is the root
                    self.data_valid[-1]["sentence"].append(
                        {
                            "ID": i + 1,
                            "FORM": sent[i],
                            "UPOS": po[i],
                            "HEAD": 0,
                            "DEPREL": "root",
                        }
                    )
                    root = i + 1
                else:
                    self.data_valid[-1]["sentence"].append(
                        {
                            "ID": i + 1,
                            "FORM": sent[i],
                            "UPOS": po[i],
                            "HEAD": root,
                            "DEPREL": "DEFAULT_ROOT",
                        }
                    )

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.acc_dyna = self.hparams.acc_dyna
            self.acc_dyna.correct = 0
            self.acc_dyna.total = 0

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        print(f"loss: {stage_loss}")
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        if stage == sb.Stage.VALID:  # metrics value
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            if self.is_training_syntax:
                with open(self.hparams.file_valid, "w", encoding="utf-8") as f_v:
                    write_token_dict_conllu(self.data_valid, f_v)
                self.data_valid = []
                d = types.SimpleNamespace()
                d.system_file = self.hparams.file_valid
                d.gold_file = self.hparams.valid_conllu
                # add file for WER insertion/sub/del
                metrics = self.hparams.eval_conll(d)
                stage_stats["LAS"] = metrics["LAS"].f1 * 100
                stage_stats["UAS"] = metrics["UAS"].f1 * 100
                stage_stats["UPOS"] = metrics["UPOS"].f1 * 100
                stage_stats["Acc_dyna"] = self.acc_dyna.summarize()
                print(
                    f"WER {stage_stats['WER']}, CER {stage_stats['CER']} : "
                    f"accd_dyna : {stage['Acc_dyna']}"
                    f"UPOS : {stage_stats['UPOS']} , UAS : {stage_stats['UAS']} LAS : {stage_stats['LAS']}"
                )
            if stage_stats["WER"] < 50:
                self.is_training_syntax = True

        # Optimization of learning rate, logging, checkpointing
        if stage == sb.Stage.VALID:
            # Annealing/Schedulers
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
            # logging
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}
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

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

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
            # plot_grad_flow(self.hparams.modules["parser"].named_parameters())

            if self.check_gradients(loss):
                self.wav2vec_optimizer.step()
                self.model_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
        return loss.detach()

    def create_words_list(self, words):
        words_list = []
        for i, w in enumerate(words):
            words_list.append(Word(w, i + 1))
        return words_list


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder}
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder}
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define wav pipeline
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

    # 2. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "words",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
    )
    def text_pipeline(wrd):
        wrds = " ".join(wrd)
        yield wrds
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    @sb.utils.data_pipeline.takes("pos", "gov", "dep")
    @sb.utils.data_pipeline.provides("pos_tokens", "head_tokens", "dep_tokens")
    def syntax_pipeline(pos, gov, dep):
        """
        compute gold configuration here
        """
        pos_tokens = torch.tensor([pos_dict.get(p) for p in pos.split(" ")])
        yield pos_tokens
        head_tokens = torch.tensor([int(h) for h in gov.split(" ")])
        yield head_tokens
        dep_token = [dep_label_dict.get(d) for d in dep.split(" ")]
        yield dep_token

    sb.dataio.dataset.add_dynamic_item(datasets, syntax_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "words",
            "tokens_list",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "pos_tokens",
            "head_tokens",
            "dep_tokens",
        ],
    )
    return train_data, valid_data, test_data


dep_label_dict = {
    "periph": 0,
    "subj": 1,
    "root": 2,
    "dep": 3,
    "dm": 4,
    "spe": 5,
    "mark": 6,
    "para": 7,
    "aux": 8,
    "disflink": 9,
    "morph": 10,
    "parenth": 11,
    "aff": 12,
    "ROOT": 13,
    "__JOKER__": 14,
    "INSERTION": 15,
    "DELETION": 16,
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
    "INSERTION": 21,
}

reverse_pos_dict = {v: k for k, v in pos_dict.items()}


def main():
    wandb.init()

    # end test debug
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # run_on_main(hparams["pretrainer"].collect_files)
    # hparams["pretrainer"].load_collected(device=run_opts["device"])

    # test debug
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

    brain = Parser(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    brain.tokenizer = tokenizer
    brain.is_training_syntax = False
    brain.data_valid = []
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # print(brain.profiler.key_averages().table(sort_by="self_cpu_time_total"))


if __name__ == "__main__":
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats()
