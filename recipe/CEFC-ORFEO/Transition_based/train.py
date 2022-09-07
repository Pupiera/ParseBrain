'''
Recipe for transition based parsing on the CEFC-ORFEO dataset.
authors : Adrien PUPIER
'''
import sys
import torch
import speechbrain as sb
import parsebrain as pb  # extension of speechbrain
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from parsebrain.processing.dependency_parsing.transition_based.configuration\
    import Configuration, GoldConfiguration, Word
from torch.nn.utils.rnn import pad_sequence

class Parser(sb.core.Brain):

    def compute_forward(self, batch, stage):
        tokens = batch.tokens.data
        tokens = tokens.to(self.device)
        tokens_conllu = batch.tokens_conllu.data.to(self.device)
        features = self.extract_features(tokens, tokens_conllu).to(self.device)
        words_list = self.create_words_list(batch.words[0]) # batch of size 1
        config = Configuration(features, words_list)
        gold_config = GoldConfiguration(batch.HEAD[0])
        if sb.Stage.TRAIN == stage:
            parse_log_prob, parse,  dynamic_oracle_decision =\
                self.hparams.parser.parse(config, stage, gold_config)
        else:
            parse_log_prob, parse, dynamic_oracle_decision =\
                self.hparams.parser.parse(config, stage)
        return parse_log_prob, dynamic_oracle_decision

    def compute_objectives(self, predictions, batch, stage):
        # compute loss : Need to compute predictions (list of gold transitions)
        parse, dynamic_oracle_decision = predictions
        loss = self.hparams.parse_cost(parse, dynamic_oracle_decision)
        return loss

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
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
            newEmb.append(b_e[b_w_end])
        return pad_sequence(newEmb, batch_first=True)


    def extract_features(self, tokens, words_end_position):
        features = self.hparams.lm_model.get_embeddings(tokens)
        return self.get_last_subword_emb(features, words_end_position)
    
    def create_words_list(self, words):
        words_list = []
        for i, w in enumerate(words):
            words_list.append(Word(w, i+1))
        return words_list


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
        "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "tokens_conllu"
    )
    def text_pipeline(words):
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
            tokens_conllu.extend([i+1]*len(tokenizer.encode_as_ids(str)))
        x = []
        y = 0
        for t_c in reversed(tokens_conllu):
            if t_c != y :
                x.append(True)
                y = t_c
            else:
                x.append(False)
        x.reverse()
        tokens_conllu = torch.BoolTensor(x)
        yield tokens_conllu

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    @sb.utils.data_pipeline.takes("POS", "HEAD", "DEP")
    @sb.utils.data_pipeline.provides("POS", "HEAD", "DEP")
    def syntax_pipeline(POS, HEAD, DEP):
        '''
        compute gold configuration here
        '''
        yield POS
        yield HEAD
        yield DEP
        #gold_config = GoldConfiguration(HEAD)
        #yield gold_config

    sb.dataio.dataset.add_dynamic_item(datasets, syntax_pipeline)


    # 3. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["words","tokens_list","tokens_bos", "tokens_eos", "tokens", "tokens_conllu", "POS", "HEAD", "DEP"],
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
