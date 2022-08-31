'''
Recipe for transition based parsing on the CEFC-ORFEO dataset.
authors : Adrien PUPIER
'''
import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import parsebrain as pb  # extension of speechbrain
import torch


class Parser(sb.core.Brain):

    def compute_forward(self, batch, stage):
        print(batch)
        print(batch.tokens)
        print(batch.tokens_conllu)
        tokens = batch.tokens.data
        print(tokens)
        print(tokens.shape)
        features = self.extract_features(batch)
        # Need to construct config for the current batch,
        config = None
        parse, decision_score_history = self.parser.parse(features, config)
        return parse, decision_score_history

    def compute_objectives(self, predictions, batch, stage):
        # compute loss : Need to compute predictions (list of gold transitions)
        parse, decision_score_history = predictions
        raise NotImplementedError

    # if stage == sb.Stage.TRAIN:

    def extract_features(self, batch):
        raise NotImplementedError


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["train_conllu"], keys=hparams['conllu_keys'],
    )

    valid_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["valid_conllu"], keys=hparams['conllu_keys'],
    )

    test_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["test_conllu"], keys=hparams['conllu_keys'],
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define text pipeline:
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens", "tokens_conllu"
    )
    def text_pipeline(words):
        wrd = " ".join(words)
        print(wrd)
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
            tokens_conllu.extend([i+1]*len(tokenizer.encode_as_ids(str)))
        tokens_conllu = torch.LongTensor(tokens_conllu)
        yield tokens_conllu

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["tokens_bos", "tokens_eos", "tokens", "tokens_conllu"],
    )
    return train_data, valid_data, test_data


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

    tokenizer = hparams['tokenizer']

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    brain = Parser(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams['dataloader_options'],
            valid_loader_kwargs=hparams['test_dataloader_options'],
            )
