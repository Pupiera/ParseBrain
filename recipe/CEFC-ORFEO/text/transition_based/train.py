"""
Recipe for transition based parsing on the CEFC-ORFEO dataset.
authors : Adrien PUPIER
"""
import math
import sys
import types

import logging

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


# todo: update transition based arc eager with unshift + disallow reduce if the root element has no deps yet
from transformers import CamembertModel, CamembertTokenizerFast

INTRA_EPOCH_CKPT_FLAG = "brain_intra_epoch_ckpt"


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

        if hasattr(self.hparams, "encoder_rnn"):
            features, hidden = self.modules.encoder_rnn(features)
        batch_size = features.shape[0]
        seq_len = torch.tensor([len(w) for w in batch.words]).to(self.device)
        config = []
        gold_config = []
        static = (
            self.hparams.number_of_epochs_static >= self.hparams.epoch_counter.current
        )
        root_value = self.hparams.special_embedding(
            torch.zeros((batch_size, 1)).to(self.device)
        )
        if stage != sb.Stage.TEST:
            for id, wrds, feat, head, dep, root in zip(
                batch.sent_id,
                batch.words,
                features,
                batch.head,
                batch.dep_tokens,
                root_value,
            ):
                config.append(
                    Configuration(
                        feat, self.create_words_list(wrds), root_embedding=root
                    )
                )
                gold_config.append(GoldConfiguration(head, dep, id))
        else:
            for wrds, feat, head, dep, root in zip(
                batch.words, features, batch.head, batch.dep_tokens, root_value
            ):
                # words_list.append(self.create_words_list(wrds))
                config.append(
                    Configuration(
                        feat, self.create_words_list(wrds), root_embedding=root
                    )
                )
        pos_log_prob = self.hparams.neural_network_POS(features)
        if sb.Stage.TRAIN == stage:
            parsing_dict = self.hparams.parser.parse(
                config, stage, gold_config, static=static
            )
        else:
            #gold config for valid to compute stat but none for test
            #import pudb; pudb.set_trace()
            parsing_dict = self.hparams.parser.parse(config, stage, gold_config)
        return parsing_dict, seq_len, pos_log_prob

    def compute_objectives(self, predictions, batch, stage):
        # compute loss : Need to compute predictions (list of gold transitions)
        parsing_dict, seq_len, pos_log_prob = predictions
        words = batch.words
        pos = batch.pos_tokens.data.to(self.device)
        sent_ids = batch.sent_id
        loss_pos = self.hparams.pos_cost(pos_log_prob, pos, length=seq_len)
        # NO ORACLE FOR TEST so no loss
        if sb.Stage.TEST != stage:
            self.hparams.acc_dyna.append(
                parsing_dict["parse_log_prob"],
                parsing_dict["oracle_parsing"],
                parsing_dict["oracle_parse_len"],
            )
            #need to encode decision_taken as one-hot for the metric
            #import pudb; pudb.set_trace()
            decision_taken = parsing_dict["decision_taken"]
            # Replace padding by another value in range of transitions, will be ignored by the metrics because of oracle_parse_len
            decision_taken[decision_taken==-100] = 0
            decision_taken_one_hot = torch.eye(self.hparams.number_transitions,device = self.device)[decision_taken]
            self.hparams.acc_dyna_with_oracle.append(
                decision_taken_one_hot,
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
            loss = 0.2 * loss_pos + 0.6 * loss_parse + 0.2 * loss_label
        else: 
            loss = torch.tensor(1)

        # Populate the list that will be written at the end of the stage.
        if sb.Stage.TRAIN != stage:
            predicted_pos = [
                [reverse_pos_dict.get(p.item()) for p in poss]
                for poss in torch.argmax(pos_log_prob, dim=-1)
            ]
            self._create_data_from_parsed_tree(
                parsing_dict["parsed_tree"], sent_ids, words, predicted_pos
            )
        return loss

    def _create_data_from_parsed_tree(self, parsed_tree, sent_ids, words, pos):
        # Bug in the case of word not having an head.
        for p_t, sent, po, sent_id in zip(parsed_tree, words, pos, sent_ids):
            self.data_valid[sent_id] = {"sent_id": sent_id, "sentence": []}
            r = [w["head"] == 0 for k, w in p_t.items()]
            has_root = any(r)
            if has_root:
                root_position = r.index(True) + 1
            for i in range(len(sent)):
                if i + 1 in p_t.keys():
                    if p_t[i + 1]["head"] == 0:
                        p_t[i + 1]["label"] = dep_label_dict["root"]
                    self.data_valid[sent_id]["sentence"].append(
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
                elif not has_root:
                    # If no head, this is the root
                    self.data_valid[sent_id]["sentence"].append(
                        {
                            "ID": i + 1,
                            "FORM": sent[i],
                            "UPOS": po[i],
                            "HEAD": 0,
                            "DEPREL": "root",
                        }
                    )
                    root_position = i + 1
                else:
                    self.data_valid[sent_id]["sentence"].append(
                        {
                            "ID": i + 1,
                            "FORM": sent[i],
                            "UPOS": po[i],
                            "HEAD": root_position,
                            "DEPREL": "DEFAULT_ROOT",
                        }
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
                self.hparams.parser.exploration_rate = ckpt.meta["exploration_rate"]


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
                self.hparams.parser.exploration_rate = ckpt.meta["exploration_rate"]


    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        self.checkpointer.save_and_keep_only(
            end_of_epoch=False,
            num_to_keep=1,
            ckpt_predicate=lambda c: INTRA_EPOCH_CKPT_FLAG in c.meta,
            meta={
                INTRA_EPOCH_CKPT_FLAG: True,
                "exploration_rate" : self.hparams.parser.exploration_rate
            },
            verbosity=logging.DEBUG,
        )



    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.hparams.acc_dyna.correct = 0
        self.hparams.acc_dyna.total = 0
        self.hparams.acc_dyna_with_oracle.correct = 0
        self.hparams.acc_dyna_with_oracle.total = 0

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            print(
                    f"loss: {stage_loss}, acc_to_oracle : {self.hparams.acc_dyna.summarize()}, acc_with_oracle_help : {self.hparams.acc_dyna_with_oracle.summarize()}, exploration_rate: {self.hparams.parser.get_exploration_rate()}"
            )

        if stage == sb.Stage.VALID:
            print(
                    f"loss: {stage_loss}, acc_to_oracle : {self.hparams.acc_dyna.summarize()}"
            )
        if stage != sb.Stage.TRAIN:  # metrics value
            # ADD changing path for writting and evaluation for test here
            with open(self.hparams.file_valid, "w", encoding="utf-8") as f_v:
                write_token_dict_conllu(self.data_valid, f_v)
            self.data_valid = {}
            d = types.SimpleNamespace()
            d.system_file = self.hparams.file_valid
            d.gold_file = self.hparams.valid_conllu
            metrics = self.hparams.eval_conll.evaluate_wrapper(d)
            stage_stats["LAS"] = metrics["LAS"].f1 * 100
            stage_stats["UAS"] = metrics["UAS"].f1 * 100
            stage_stats["UPOS"] = metrics["UPOS"].f1 * 100
            print(
                f"UPOS : {stage_stats['UPOS']} , UAS : {stage_stats['UAS']} LAS : {stage_stats['LAS']}"
            )
        if (
            stage == sb.Stage.VALID
        ):  # Optimization of learning rate, logging, checkpointing

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}
            wandb.log(wandb_stats)
            self.checkpointer.save_and_keep_only(
                meta={"LAS": stage_stats["LAS"],
                    "exploration_rate": self.hparams.parser.exploration_rate},
                max_keys=["LAS"]
            )
        # update the exploration rate
        if stage == sb.Stage.VALID and epoch > self.hparams.number_of_epochs_static:
            self.hparams.parser.update_exploration_rate(
                self.hparams.scheduler.update_rate(self.hparams.parser.exploration_rate)
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

    def count_param_module(self):
        for key, value in self.modules.items():
            print(key)
            print(sum(p.numel() for p in value.parameters()))


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
    print(brain.device)
    brain.hparams.features_extractor.set_model(camembert)
    brain.tokenizer = tokenizer
    brain.data_valid = {}
    brain.count_param_module()
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
        )

    #import pudb; pudb.set_trace()
    brain.evaluate(
        test_data,
        max_key='LAS',
        test_loader_kwargs=hparams["test_dataloader_options"]
    )

    # print(brain.profiler.key_averages().table(sort_by="self_cpu_time_total"))


if __name__ == "__main__":
    import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()
