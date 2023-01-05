import torch


class EmbeddingStrategy:
    def set_model(self, model):
        self.model = model
        self.device = model.device

    def extract_features(self, tokens, words_end_position):
        raise NotImplementedError("")


class LastSubWordEmbedding(EmbeddingStrategy):
    def get_last_subword_emb(self, emb, words_end_position):
        newEmb = []
        for b_e, b_w_end in zip(emb, words_end_position):
            newEmb.append(b_e[b_w_end].to(self.device))
        return torch.nn.utils.rnn.pad_sequence(newEmb, batch_first=True)
        # return pad_sequence(newEmb, batch_first=True)

    def extract_features(self, tokens, words_end_position):
        # features = self.hparams.lm_model.get_embeddings(tokens)
        # print(tokens)
        features = self.model(tokens)["last_hidden_state"]
        features = features[:, 1:-1, :]  # Remove <bos> and <eos>
        # print(features.shape)
        # print(words_end_position)
        # print(self.tokenizer.convert_ids_to_tokens(tokens[0]))
        return self.get_last_subword_emb(features, words_end_position)
