#!/usr/bin/env python3
import sys
import torch
from tqdm import tqdm
import time

import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main


from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader


from parsebrain.speechbrain_custom.decoders.ctc import ctc_greedy_decode
from torch.nn.utils.rnn import pad_sequence
import logging


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
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)  # [batch,time,1024]
        logits = self.modules.ctc_lin(x)  # [batch,time,76]
        # [batch,time,76] (76 in output neurons corresponding to BPE size)
        p_ctc = self.hparams.log_softmax(logits)
        # Use BOS syntactic information if we condition on previous word, use without bos if we condition on the predictions of
        # the current word. Pronbably more pertinent with current word, so no pos but <eos> instead.
        # Forward pass dependency parsing
        # [ batch, subwords, dim]
        seq_emb = self.hparams.emb_asr(tokens_bos)
        logits_seq, attn = self.modules.seqdec(seq_emb, x, wav_lens)
        p_seq_tf = self.hparams.log_softmax(logits_seq)
        result = {"p_ctc": p_ctc, "wav_lens": wav_lens, "p_seq_tf": p_seq_tf}
        # if in dev or test, do beam search.
        if stage != sb.Stage.TRAIN:
            # beam search
            # If training syntax, use the predictions else use teacher forcing (on syntax)
            # synt emb is either the teacher forcing one, or the same than for the greedy search.
            p_tokens, score = self.hparams.beam_searcher(x, wav_lens)
            # Override the greedy search
            result["p_tokens"] = p_tokens
        return result

    def compute_objectives(self, predictions, batch, stage):
        ids = batch.id

        tokens, tokens_lens = batch.tokens
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        loss = self.hparams.ctc_cost(
            predictions["p_ctc"], tokens, predictions["wav_lens"], tokens_lens
        )
        loss_seq = self.hparams.seq_cost(
            predictions["p_seq_tf"], tokens_eos, length=tokens_eos_lens
        )
        loss += loss_seq
        if sb.Stage.TRAIN != stage:
            # need to get predicted words

            """
            sequence = sb.decoders.ctc_greedy_decode(
                predictions["p_ctc"],
                predictions["wav_lens"],
                blank_id=self.hparams.blank_index,
            )
            """
            sequence = predictions["p_tokens"]
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
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
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
            self.checkpointer.save_and_keep_only(
                meta={
                    "WER": stage_stats["WER"],
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
        "tokens_list", "tokens_bos", "tokens_eos", "tokens", "subword_count_bos"
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
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        subword_bos_len = [1] + [
            len(tokenizer.sp.encode_as_ids(w)) for w in wrd.split(" ")
        ]
        yield torch.LongTensor(subword_bos_len)

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

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
        ],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":
    wandb.init(group="Audio")
    print(f"wandb run name : {wandb.run.name}")
    path_encoded_train = "all.seq"  # For alphabet generation
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
    # Diverse information on the data such as PATH and order of sentences.
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
    run_on_main(
        asr_brain.transcribe_dataset,
        kwargs={
            "dataset": test_data,
            "min_key": "WER",
            "loader_kwargs": hparams["test_dataloader_options"],
        },
    )
    """
    asr_brain.transcribe_dataset(
        dataset=test_data,  # Must be obtained from the dataio_function
        min_key="WER",
        loader_kwargs=hparams["test_dataloader_options"],
    )
    """
