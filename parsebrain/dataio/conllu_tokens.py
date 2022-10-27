class ConlluDict(dict):
    def extend_by_keys(self, keys, values):
        for k, v in zip(keys, values):
            if k in self.keys():
                self[k].append(v)
            else:
                self[k] = [v]

    def set_sent_id(self, sent_id):
        self["sent_id"] = sent_id

    def is_empty(self, keys):
        """
        Test the first key in the list of key.
        If key not in the dict return True
        """
        try:
            return len(self[keys[0]]) == 0
        except KeyError:
            return True
