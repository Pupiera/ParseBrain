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

logger = logging.getLogger(__name__)

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

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        feats = self.modules.wav2vec2(wavs)

        # Add augmentation if specified

        x = self.modules.enc(feats)
        # teacher forcing in train time.
        seq_emb = self.hparams.emb_asr(tokens_with_bos)
        pos_emb = self.hparams.pos_emb(pos_eos)
        gov_emb = self.hparams.gov_emb(gov_eos)
        dep_emb = self.hparams.dep_emb(dep_eos)

        # not sure of the shape expected. Should H_enc and H_dec be of same size ?
        synt_emb = torch.cat((pos_emb, gov_emb, dep_emb), dim=-1)
        batch_size = feats.size(0)
        max_word_len = synt_emb.size(1)
        max_seq_len = seq_emb.size(1)

        # not this easy for the teacher forcing, because shape synt_emb != seq_emb
        # need to give same syntaxic emb to the same word even if it's divised in subword.

        aligned_emb = torch.zeros(
            (batch_size, max_seq_len, synt_emb.size(-1)), device=self.device
        )
        # if bos take full subword_count, else take it without the first element. [1:]
        subword_count, _ = batch.subword_count_bos
        for i in range(batch_size):
            for j in range(max_word_len):
                if subword_count[i][j].item() == 0:
                    continue
                start_index = sum(subword_count[i][:j])
                end_index = start_index + subword_count[i][j]
                # synt_emb [1,1,dim]
                word_emb = synt_emb[i, j, :]
                # [subword(j), dim]
                repeated_word_emb = word_emb.repeat(subword_count[i][j], 1)
                aligned_emb[i, start_index:end_index, :] = repeated_word_emb
        syntax_emb = torch.cat((seq_emb, aligned_emb), dim=-1)

        h, _ = self.modules.dec(syntax_emb)
        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))
        print(joint.shape)
        # joint shape [B, U, T, H]

        # Output layer for transducer log-probabilities
        logits_transducer = self.modules.transducer_lin(joint)

        result_trans = {"logits_transducer": logits_transducer, "wav_lens": wav_lens}

        if self.is_training_syntax:
            input_dep, seq_len = self.hparams.word_speech_rep(x)
            input_dep = input_dep.to(self.device)
            seq_len = seq_len.to(self.device)
            batch_len = input_dep.shape[0]
            # init hidden to zeros for each sentence
            hidden = torch.zeros(4, batch_len, 800, device=self.device)
            cell = torch.zeros(4, batch_len, 800, device=self.device)
            lstm_out, _ = self.modules.dep2LabelFeat(input_dep, (hidden, cell))
            logits_posLabel = self.modules.posDep2Label(lstm_out)
            p_posLabel = self.hparams.log_softmax(logits_posLabel)
            logits_depLabel = self.modules.depDep2Label(lstm_out)
            p_depLabel = self.hparams.log_softmax(logits_depLabel)
            logits_govLabel = self.modules.govDep2Label(lstm_out)
            p_govLabel = self.hparams.log_softmax(logits_govLabel)
            # Greedy search for seq2seq to get ASR for syntax oracle.

            pred_gov = torch.argmax(p_govLabel)
            pred_dep = torch.argmax(p_depLabel)
            pred_pos = torch.argmax(p_posLabel)
            # Add eos to each element
            pred_gov = torch.cat(
                (pred_gov, torch.ones((batch, 1) * hparams["gov_eos"])), dim=1
            )
            pred_dep = torch.cat(
                (pred_dep, torch.ones((batch, 1) * hparams["dep_eos"])), dim=1
            )
            pred_pos = torch.cat(
                (pred_pos, torch.ones((batch, 1) * hparams["pos_eos"])), dim=1
            )
            # Get embedding from predictions
            pos_emb = self.hparams.pos_emb(pred_pos)
            gov_emb = self.hparams.gov_emb(pred_gov)
            dep_emb = self.hparams.dep_emb(pred_dep)
            result_parsing = {
                "p_posLabel": p_posLabel,
                "p_depLabel": p_depLabel,
                "p_govLabel": p_govLabel,
                "seq_len": seq_len,
            }
            result_trans = {**result_trans, **result_parsing}
            synt_emb = torch.cat((pos_emb, gov_emb, dep_emb), dim=-1)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            if self.is_training_syntax:
                hyps = self.hparams.greedy_beam_searcher(x, synt_emb)
                result_trans["predicted_tokens"] = hyps
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
                p_ce = self.modules.dec_lin(h)
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

        elif stage == sb.Stage.VALID:
            best_hyps, scores, _, _ = self.hparams.beam_searcher(x)
            result_beam = {"predicted_tokens": best_hyps}
            result = {**result_trans, **result_beam}
            return result
        else:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.beam_searcher(x)
            result_beam = {"predicted_tokens": best_hyps}
            result = {**result_trans, **result_beam}
            return result

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""

        ids = batch.id
        current_epoch = self.hparams.epoch_counter.current
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos

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
            elif "p_ctc" in predictions.keys() or "p_ce" in predictions.keys()
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
            predicted_words = self.tokenizer(predictions["predicted_tokens"], task="decode_from_list")

            # Convert indices to words
            target_words = undo_padding(tokens, token_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        if self.is_training_syntax:
            loss_parsing = self.compute_objectives_parsing(predictions, batch, stage)
            loss+= loss_parsing
        return loss

    def compute_objectives_parsing(self, predictions, batch, stage):
        raise NotImplementedError


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

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
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
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
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
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
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

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Define dependency pipeline

    @sb.utils.data_pipeline.takes("wrd", "pos", "gov", "dep")
    @sb.utils.data_pipeline.provides(
        "wrd",
        "pos_tensor",
        "pos_bos",
        "pos_eos",
        "govDep2label",
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
        yield torch.LongTensor([hparams["pos_bos"]] + pos_list)
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
            yield torch.LongTensor([hparams["gov_bos"]] + gov_list)
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
        yield torch.LongTensor([hparams["dep_bos"]] + dep_list)
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
            "wrd",
            "pos_tensor",
            "pos_bos",
            "pos_eos",
            "govDep2label",
            "gov_bos",
            "gov_eos",
            "depDep2Label",
            "dep_bos",
            "dep_eos",
        ],
    )
    return train_data, valid_data, test_data


def build_label_alphabet(path_encoded_train):
    label_gov = dict()
    label_dep = dict()
    label_pos = dict()
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
    asr_brain.hparams.oracle.set_alphabet(label_alphabet, reverse_label_alphabet)
    asr_brain.is_training_syntax = False
    asr_brain.hparams.evaluator.set_alphabet(label_alphabet)
    # Diverse information on the data such as PATH and order of sentences.
    asr_brain.dev_order = get_id_from_CoNLLfile(hparams["dev_gold_conllu"])
    asr_brain.test_order = get_id_from_CoNLLfile(hparams["test_gold_conllu"])
    asr_brain.tokenizer = tokenizer

    list_final_subtokens = get_final_subtokens(tokenizer)
    asr_brain.hparams.beam_searcher.set_list_final_subtoken(list_final_subtokens)
    asr_brain.hparams.greedy_search.set_list_final_subtoken(list_final_subtokens)

    # Training
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
