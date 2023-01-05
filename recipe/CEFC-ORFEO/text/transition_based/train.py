"""
Recipe for transition based parsing on the CEFC-ORFEO dataset.
authors : Adrien PUPIER
"""
import math
import sys
import types

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import parsebrain as pb  # extension of speechbrain
import wandb
from debug_utils import plot_grad_flow
from parsebrain.dataio.pred_to_file.pred_to_conllu import write_token_dict_conllu

from parsebrain.processing.dependency_parsing.transition_based.configuration import (
    Configuration,
    GoldConfiguration,
    Word,
)


from transformers import CamembertModel, CamembertTokenizerFast


# @export
# @profile_optimiser
class Parser(sb.core.Brain):
    def compute_forward(self, batch, stage):
        tokens = batch.tokens.data
        tokens = tokens.to(self.device)
        tokens_conllu = batch.tokens_conllu.data.to(self.device)
        features = self.hparams.features_extractor.extract_features(
            tokens, tokens_conllu
        )
        seq_len = torch.tensor([len(w) for w in batch.words]).to(self.device)
        config = []
        gold_config = []
        static = (
            self.hparams.number_of_epochs_static >= self.hparams.epoch_counter.current
        )
        if stage != sb.Stage.TEST:
            for id, wrds, feat, head, dep in zip(
                batch.sent_id, batch.words, features, batch.head, batch.dep_tokens
            ):
                # words_list.append(self.create_words_list(wrds))
                config.append(Configuration(feat, self.create_words_list(wrds)))
                gold_config.append(GoldConfiguration(head, dep, id))
        else:
            for wrds, feat, head, dep in zip(
                batch.words, features, batch.head, batch.dep_tokens
            ):
                # words_list.append(self.create_words_list(wrds))
                config.append(Configuration(feat, self.create_words_list(wrds)))
        pos_log_prob = self.hparams.neural_network_POS(features)
        if sb.Stage.TRAIN == stage:
            parsing_dict = self.hparams.parser.parse(
                config, stage, gold_config, static=static
            )
        else:
            parsing_dict = self.hparams.parser.parse(config, stage, gold_config)
        # print(parsing_dict["parsed_tree"])
        return parsing_dict, seq_len, pos_log_prob

    def compute_objectives(self, predictions, batch, stage):
        # compute loss : Need to compute predictions (list of gold transitions)
        parsing_dict, seq_len, pos_log_prob = predictions
        words = batch.words
        pos = batch.pos_tokens.data.to(self.device)
        sent_ids = batch.sent_id

        loss_pos = self.hparams.pos_cost(pos_log_prob, pos, length=seq_len)
        self.hparams.acc_dyna.append(
            parsing_dict["parse_log_prob"],
            parsing_dict["oracle_parsing"],
            parsing_dict["oracle_parse_len"],
        )
        loss_parse = self.hparams.parse_cost(
            parsing_dict["parse_log_prob"],
            parsing_dict["oracle_parsing"],
            parsing_dict["oracle_parse_len"],
        )
        # Compute the loss for each element based on decision and only keep relevant one.
        # Allow to compute label in a batch way.
        loss_label = self.hparams.label_cost(
            parsing_dict["label_log_prob"],
            parsing_dict["oracle_label"],
            parsing_dict["oracle_label_len"],
        )
        loss = 0.3 * loss_pos + 0.4 * loss_parse + 0.3 * loss_label
        # Populate the list that will be written at the end of the stage.
        if sb.Stage.VALID == stage:
            predicted_pos = [
                [reverse_pos_dict.get(p.item()) for p in poss]
                for poss in torch.argmax(pos_log_prob, dim=-1)
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

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        print(
            f"loss: {stage_loss}, acc_to_oracle : {self.hparams.acc_dyna.summarize()}"
        )
        if stage == sb.Stage.VALID:  # metrics value
            with open(self.hparams.file_valid, "w", encoding="utf-8") as f_v:
                write_token_dict_conllu(self.data_valid, f_v)
            self.data_valid = []
            d = types.SimpleNamespace()
            d.system_file = self.hparams.file_valid
            d.gold_file = self.hparams.valid_conllu
            metrics = self.hparams.eval_conll(d)
            stage_stats["LAS"] = metrics["LAS"].f1 * 100
            stage_stats["UAS"] = metrics["UAS"].f1 * 100
            stage_stats["UPOS"] = metrics["UPOS"].f1 * 100
            print(
                f"UPOS : {stage_stats['UPOS']} , UAS : {stage_stats['UAS']} LAS : {stage_stats['LAS']}"
            )
        if (
            stage == sb.Stage.VALID
        ):  # Optimization of learning rate, logging, checkpointing
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}
            wandb.log(wandb_stats)
            # self.checkpointer.save_and_keep_only(
            #    meta={"LAS": stage_stats["LAS"]}, max_keys=["LAS"]
            # )

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
            # plot_grad_flow(self.hparams.modules["parser"].named_parameters())
            # exit()
            if self.check_gradients(loss):
                self.model_optimizer.step()

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

    train_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["train_conllu"],
        keys=hparams["conllu_keys"],
    )
    train_data = train_data.filtered_sorted(sort_key="sent_len")

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
        # tokens_list = tokenizer.encode_as_ids(wrd)
        tokens_list = tokenizer.encode_plus(wrd)["input_ids"]
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        tokens_conllu = []
        for i, str in enumerate(words):
            # tokens_conllu.extend([i + 1] * len(tokenizer.encode_as_ids(str)))
            tokens_conllu.extend(
                [i + 1] * len(tokenizer.encode_plus(str)["input_ids"][1:-1])
            )
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
        pos_tokens = torch.tensor([pos_dict.get(p) for p in pos])
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
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
    camembert = CamembertModel.from_pretrained("camembert-base").to(run_opts["device"])
    for param in camembert.parameters():
        param.require_grad = False
    # tokenizer = hparams["tokenizer"]

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    brain = Parser(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    brain.hparams.features_extractor.set_model(camembert)
    brain.tokenizer = tokenizer
    for param in brain.hparams.modules["lm_model"].parameters():
        param.require_grad = False
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
