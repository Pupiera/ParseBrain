#!/usr/bin/env python3
import sys
import torch
import logging
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------------------- Import profiling -------------------------------------------#
import wandb

# ------------------------------------------ Transcribe import -----------------------------------------#
from speechbrain.utils.edit_distance import wer_details_for_batch
from parsebrain.speechbrain_custom.decoders.ctc import ctc_greedy_decode


"""Recipe for training a sequence-to-sequence ASR system with CommonVoice.
The system employs a wav2vec2 encoder and a ASRHOPS decoder.
Decoding is performed with greedy decoding (will be extended to beam search).

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml

With the default hyperparameters, the system employs a pretrained wav2vec2 encoder.
The wav2vec2 model is pretrained following the model given in the hprams file.
It may be dependent on the language.

The neural network is trained with ASR on sub-word units estimated with
Byte Pairwise Encoding (BPE).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training languages (all CommonVoice languages), and many
other possible variations.

Authors
 * Titouan Parcollet 2021 for ASR template
 * PUPIER Adrien 2022 for adapting template to orfeo dataset
"""


# -----------------------------------------------------------BRAIN ASR---------------------------------------------#
# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
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
        p_ctc = self.hparams.log_softmax(
            logits
        )  # [batch,time,76] (76 in output neurons corresponding to BPE size)
        # VAE with representation on the semantic of the whole sentence

        rep_sent = self.modules.rnn_frames(x)[:, -1, :]
        rep_sent_encoded = self.modules.encoder_vae(rep_sent)
        mean = self.modules.mean_nn(rep_sent_encoded)
        std = self.modules.std_nn(rep_sent_encoded)
        latent = self.sample_VAE(mean, std)
        decoded = self.modules.decoder_sent(latent)

        # seq2seq decoding of this semantic latent variable
        e_in = self.emb(tokens_bos)
        h, _ = self.modules.seq_decoder(e_in, x, hidden_states=decoded)
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)
        result = {"p_ctc": p_ctc, "wav_lens": wav_lens, "p_seq": p_seq}
        if stage != sb.stage.TRAIN:
            p_tokens, scores = self.hparams.beam_search(x, hidden_states=decoded)
            result["p_tokens"] = p_tokens
        return result

    def sample_VAE(self, mu: torch.tensor, logvar: torch.tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :@param mu : (Tensor) Mean of the latent Gaussian [B, L D]
        :@param logvar : (Tensor) Standard deviation of latent Gaussian
        Equivalent to the diagonal of the covariance matrix
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (ASRHOPS) given predictions and targets."""
        if stage != sb.Stage.TRAIN:
            p_ctc, wav_lens, sequence = predictions
        else:
            p_ctc, wav_lens = predictions
        ids = batch.id

        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        if stage != sb.Stage.TRAIN:
            predicted_words = self.tokenizer(sequence, task="decode_from_list")
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def transcribe_dataset(
        self,
        dataset,  # Must be obtained from the dataio_function
        min_key,  # We load the model with the lowest WER
        loader_kwargs,  # opts for the dataloading
    ):

        # If dataset isn't a Dataloader, we create it.
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(dataset, sb.Stage.TEST, **loader_kwargs)

        self.on_evaluate_start(
            min_key=min_key
        )  # We call the on_evaluate_start that will load the best model
        self.modules.eval()  # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            transcripts = []
            sent_ids = []
            WER = []
            for batch in tqdm(dataset, dynamic_ncols=True):

                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied
                # in compute_forward().
                out = self.compute_forward(batch, stage=sb.Stage.TEST)
                p_ctc, wav_lens, predicted_tokens = out
                # We go from tokens to words.
                predicted_words = self.tokenizer(
                    predicted_tokens, task="decode_from_list"
                )
                transcripts.append(predicted_words)
                print(predicted_words)
                tokens, tokens_lens = batch.tokens
                target_words = undo_padding(tokens, tokens_lens)
                target_words = self.tokenizer(target_words, task="decode_from_list")
                WER.append(
                    wer_details_for_batch(batch.id, target_words, predicted_words)
                )
                sent_ids.append(batch.id)

        transcripts = [item for sublist in transcripts for item in sublist]
        sent_ids = [item for sublist in sent_ids for item in sublist]
        WER = [item for sublist in WER for item in sublist]
        return transcripts, sent_ids, WER

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
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

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
    # Make dataset with train but not filtered by duration
    transcribe_train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

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

    datasets = [train_data, transcribe_train_data, valid_data, test_data]

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
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, transcribe_train_data, valid_data, test_data


if __name__ == "__main__":
    wandb.init(group="Audio")
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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
    train_data, transcribe_train_data, valid_data, test_data = dataio_prepare(
        hparams, tokenizer
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    # doing this to avoid overwriting the class constructor
    asr_brain.tokenizer = tokenizer
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
                    print(type(obj), obj.size())
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

    # train
    transcripts, sent_ids, WER = asr_brain.transcribe_dataset(
        dataset=transcribe_train_data,  # Must be obtained from the dataio_function
        min_key="WER",  # We load the model with the lowest WER
        loader_kwargs=hparams["test_dataloader_options"],  # opts for the dataloading
    )

    with open(hparams["transcript_train"], "w") as out:
        for t, sent_id, wer in zip(transcripts, sent_ids, WER):
            txt = " ".join(t)
            if txt == "":
                txt = "EMPTY_ASR"
            out.write(f"{txt}\t({sent_id})\t{wer['WER']}\n")
    # dev
    transcripts, sent_ids, WER = asr_brain.transcribe_dataset(
        dataset=valid_data,  # Must be obtained from the dataio_function
        min_key="WER",  # We load the model with the lowest WER
        loader_kwargs=hparams["test_dataloader_options"],  # opts for the dataloading
    )

    with open(hparams["transcript_dev"], "w") as out:
        for t, sent_id, wer in zip(transcripts, sent_ids, WER):
            txt = " ".join(t)
            if txt == "":
                txt = "EMPTY_ASR"
            out.write(f"{txt}\t({sent_id})\t{wer['WER']}\n")

    #  test
    transcripts, sent_ids, WER = asr_brain.transcribe_dataset(
        dataset=test_data,  # Must be obtained from the dataio_function
        min_key="WER",  # We load the model with the lowest WER
        loader_kwargs=hparams["test_dataloader_options"],  # opts for the dataloading
    )

    with open(hparams["transcript_test"], "w") as out:
        for t, sent_id, wer in zip(transcripts, sent_ids, WER):
            txt = " ".join(t)
            if txt == "":
                txt = "EMPTY_ASR"
            out.write(f"{txt}\t({sent_id})\t{wer['WER']}\n")
