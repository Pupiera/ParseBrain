"""
Recipe for transition based parsing on the CEFC-ORFEO dataset.
authors : Adrien PUPIER
"""
import sys

import speechbrain as sb
import torch
import wandb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import parsebrain as pb  # extension of speechbrain
from parsebrain.dataio.pred_to_file.pred_to_conllu import write_token_dict_conllu
from parsebrain.processing.dependency_parsing.eval.conll18_ud_eval import (
    evaluate_wrapper,
)
from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
    GoldConfiguration,
    Word,
)


# ToDo: num_workers back to 6 (0 for debugging)
class Parser(sb.core.Brain):
    def compute_forward(self, batch, stage):
        tokens = batch.tokens.data
        tokens = tokens.to(self.device)
        tokens_conllu = batch.tokens_conllu.data.to(self.device)
        features = self.extract_features(tokens, tokens_conllu)
        config = []
        gold_config = []
        for wrds, feat, head, dep in zip(
            batch.words, features, batch.head, batch.dep_tokens
        ):
            # words_list.append(self.create_words_list(wrds))
            config.append(Configuration(feat, self.create_words_list(wrds)))
            gold_config.append(GoldConfiguration(head, dep))
        if sb.Stage.TRAIN == stage:
            parsing_dict = self.hparams.parser.parse(config, stage, gold_config)
        else:
            parsing_dict = self.hparams.parser.parse(config, stage)
        return (
            parsing_dict["decision_score"],
            parsing_dict["decision_taken"],
            parsing_dict["oracle_parsing"],
            parsing_dict["label_score"],
            parsing_dict["oracle_label"],
            parsing_dict["mask_label"],
            parsing_dict["parsed_tree"],
        )

    def compute_objectives(self, predictions, batch, stage):
        # compute loss : Need to compute predictions (list of gold transitions)
        (
            parse_log_prob,
            parse,
            dynamic_oracle_decision,
            label_log_prob,
            dynamic_oracle_label,
            mask_label_decision,
            parsed_tree,
        ) = predictions
        words = batch.words
        pos = batch.pos_tokens
        sent_ids = batch.sent_id
        # Padded decision for structure of tree is marked with -1
        mask_parse = parse != -1
        # We compute the loss for each value, and we only keep the case where the decision was valid. (not batch padding)
        # loss need log prob in form (Batch, class, seq)
        loss = (
            self.hparams.parse_cost(
                torch.transpose(parse_log_prob, 1, -1), dynamic_oracle_decision
            )
            .masked_select(mask_parse)
            .mean()
        )
        # Compute the loss for each element based on decision and only keep relevant one.
        # Allow to compute label in a batch way.
        loss += (
            self.hparams.label_cost(
                torch.transpose(label_log_prob, 1, -1), dynamic_oracle_label
            )
            .masked_select(mask_label_decision)
            .mean()
        )
        # Populate the list that will be written at the end of the stage.
        if sb.Stage.VALID == stage or True:
            self._create_data_from_parsed_tree(parsed_tree, sent_ids, words, pos)
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

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.VALID:  # metrics value
            with open(self.hparams.file_valid, "w", encoding="utf-8") as f_v:
                write_token_dict_conllu(self.data_valid, f_v)
            self.data_valid = []
            metrics = evaluate_wrapper(
                {
                    "system_file": self.hparams.file_valid,
                    "gold_file": self.hparams.valid_conllu,
                }
            )
            stage_stats["LAS"] = metrics["LAS"].f1 * 100
            stage_stats["UAS"] = metrics["UAS"].f1 * 100

        if (
            stage == sb.Stage.VALID
        ):  # Optimization of learning rate, logging, checkpointing
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}
            wandb.log(wandb_stats)
            self.checkpointer.save_and_keep_only(
                meta={"LAS": stage_stats["LAS"]}, max_keys=["LAS"]
            )

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:

            self.model_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.model_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()
            if self.check_gradients(loss):
                self.model_optimizer.step()

            self.model_optimizer.zero_grad()

        return loss.detach()

    def get_last_subword_emb(self, emb, words_end_position):
        newEmb = []
        for b_e, b_w_end in zip(emb, words_end_position):
            newEmb.append(b_e[b_w_end].to(self.device))
        return newEmb
        # return pad_sequence(newEmb, batch_first=True)

    def extract_features(self, tokens, words_end_position):
        features = self.hparams.lm_model.get_embeddings(tokens)
        return self.get_last_subword_emb(features, words_end_position)

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

    train_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["train_conllu"],
        keys=hparams["conllu_keys"],
    )

    valid_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["valid_conllu"],
        keys=hparams["conllu_keys"],
    )

    test_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["test_conllu"],
        keys=hparams["conllu_keys"],
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define text pipeline:
    @sb.utils.data_pipeline.takes("sent_id", "words")
    @sb.utils.data_pipeline.provides(
        "sent_id",
        "words",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
        "tokens_conllu",
    )
    def text_pipeline(sent_id, words):
        yield sent_id
        wrd = " ".join(words)
        yield words
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        tokens_conllu = []
        for i, str in enumerate(words):
            tokens_conllu.extend([i + 1] * len(tokenizer.encode_as_ids(str)))
        x = []
        y = 0
        for t_c in reversed(tokens_conllu):
            if t_c != y:
                x.append(True)
                y = t_c
            else:
                x.append(False)
        x.reverse()
        tokens_conllu = torch.BoolTensor(x)
        yield tokens_conllu

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    @sb.utils.data_pipeline.takes("POS", "HEAD", "DEP")
    @sb.utils.data_pipeline.provides("pos_tokens", "head", "dep_tokens")
    def syntax_pipeline(pos, head, dep):
        """
        compute gold configuration here
        """
        pos_tokens = pos
        yield pos_tokens
        yield head
        dep_token = [dep_label_dict.get(d) for d in dep]
        yield dep_token
        # gold_config = GoldConfiguration(HEAD)
        # yield gold_config

    sb.dataio.dataset.add_dynamic_item(datasets, syntax_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(
    datasets,
    [
        "sent_id",
        "words",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
        "tokens_conllu",
        "pos_tokens",
        "head",
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

if __name__ == "__main__":
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
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    tokenizer = hparams["tokenizer"]

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    brain = Parser(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    brain.data_valid = []
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )
