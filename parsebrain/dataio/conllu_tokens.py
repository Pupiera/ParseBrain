class ConlluDict(dict):
    def extend_by_keys(self, keys, values):
        for k, v in zip(keys, values):
            if k in self.keys():
                self[k].append(v)
            else:
                self[k] = [v]

    def set_sent_id(self, sent_id):
        self['sent_id'] = sent_id
