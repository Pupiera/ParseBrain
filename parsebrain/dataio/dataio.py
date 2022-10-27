import re

from parsebrain.dataio.conllu_tokens import ConlluDict


def load_data_conllu(conllu_path, keys, replacement={}):
    """
    Load conllu and format string values
    file must be formated with # sent_id = XXX before the tokens
    :param conllu_path:
    :param replacement:
    :return:

    --------
    Exemple with one sentence extracted from CFPB corpus :
    >>> conllu_ex = ''' # sent_id = cefc-cfpb-1000-5-260
    ... # text = c'est pas possible parce que j'ai des choses
    ... 1	c'  ce	CLS	CLS	_	2	subj	_	_	663.030029	663.400024	Marcel_Gabbio
    ... 2	est	être	VRB	VRB	_	0	root	_	_	663.409973	663.460022	Marcel_Gabbio
    ... 3	pas	pas	ADN	ADN	_	2	dep	_	_	663.469971	663.590027	Marcel_Gabbio
    ... 4	possible	possible	ADJ	ADJ	_	2	dep	_	_	663.599976	663.969971	Marcel_Gabbio
    ... 5	parce que	parce que	CSU	CSU	_	2	dep	_	_	663.979980	664.159973	Marcel_Gabbio
    ... 6	j'	je	CLS	CLS	_	7	subj	_	_	664.169983	664.260010	Marcel_Gabbio
    ... 7	ai	avoir	VRB	VRB	_	5	dep	_	_	664.270020	664.330017	Marcel_Gabbio
    ... 8	des	de	PRE	PRE	_	7	dep	_	_	664.340027	664.419983	Marcel_Gabbio
    ... 9	choses	chose	NOM	NOM	_	8	dep	_	_	664.429993	664.820007	Marcel_Gabbio'''
    >>> tmp_file = 'test.conllu'
    >>> with open(tmp_file , "w", encoding="utf-8") as fo:
    ...     _ = fo.write(conllu_ex)
    >>>
    >>> data = load_data_conllu(tmp_file, ['lineNumber', 'words', 'lemmas','POS','UPOS','tags','HEAD','DEP','tags2','tags3','timestamp_begin','timestamp_end','speaker'])
    >>> data['cefc-cfpb-1000-5-260']['words'], data['cefc-cfpb-1000-5-260']['UPOS']
    (["c'", 'est', 'pas', 'possible', 'parce que', "j'", 'ai', 'des', 'choses'], ['CLS', 'VRB', 'ADN', 'ADJ', 'CSU', 'CLS', 'VRB', 'PRE', 'NOM'])
    >>>
    """
    with open(conllu_path, "r", encoding="utf-8") as conllu_file:
        result = {}
        current_sent = ConlluDict()
        for line in conllu_file:
            line = line.strip()
            if line.startswith("#"):  # comment in conllu file
                if line.startswith("# sent_id"):
                    sent_id = line.split()[-1]
                    current_sent = ConlluDict()
                    current_sent.set_sent_id(sent_id)
                continue
            if not line:  # end of sentence
                try:
                    result[sent_id] = current_sent
                except KeyError:
                    raise KeyError(
                        "conllu sentence has to have an 'sent_id' comment, with unique ids"
                        " for all data points"
                    )
                current_sent = ConlluDict()
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                fields = re.split(r"\s{2,}", line)
            current_sent.extend_by_keys(keys, fields)
        if not current_sent.is_empty(keys):
    result[sent_id] = current_sent
        return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
