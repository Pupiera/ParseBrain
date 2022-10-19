ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)  # same as eval
label = [
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
]


def write_token_dict_conllu(data, file):
    """
    Write a dict into a file in the conllu format

    >>> data_ = [{"sent_id": "test1", "sentence" : [{"ID": 1, "FORM": "XXX","UPOS": "YYY", "HEAD":2, "DEPREL" :"deps"},
    ...                                            {"ID": 2, "FORM": "XYY", "UPOS": "YXX", "HEAD":0, "DEPREL" :"root"}]},
    ...            {"sent_id": "test2", "sentence" : [{"ID": 1, "FORM": "XXX","UPOS": "YYY", "HEAD":2, "DEPREL" :"deps"},
    ...                                          {"ID": 2, "FORM": "XYY", "UPOS": "YXX", "HEAD":0, "DEPREL" :"root"}]}]
    >>> with open("test_write.conllu","w",encoding="utf-8") as tmp_file:
    ...     write_token_dict_conllu(data_, tmp_file)
    """
    for d in data:
        sent = ""
        if "sent_id" in d:
            sent = f"# sent_id = {d['sent_id']}\n"
        sent += "# text = "
        sentence = d["sentence"]
        for token in sentence:
            sent += f"{token['FORM']} "
        sent += "\n"
        for token in sentence:
            for l in label:
                if l in token:
                    sent += f"{token[l]}\t"
                else:
                    sent += "_\t"
            sent = sent[0:-1]
            sent += "\n"  # replace last \t of each line by \n
        sent += "\n"
        file.write(sent)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
