import re
from typing import List

from parsebrain.dataio.conllu_tokens import ConlluDict


def load_data_conllu(conllu_path: str, keys: List, replacement: dict = {}, UD: bool = False, blackListed_value: dict = None):
    """
    Load conllu and format string values
    file must be formated with # sent_id = XXX before the tokens
    :param conllu_path:
    :param replacement:
    :return:

    --------
    Exemple with one sentence extracted from CFPB corpus :
    (Note, the punct "," is not in the original sentence
    >>> conllu_ex = ''' # sent_id = cefc-cfpb-1000-5-260
    ... # text = c'est pas possible parce que j'ai des choses
    ... 1	c'  ce	CLS	CLS	_	2	subj	_	_	663.030029	663.400024	Marcel_Gabbio
    ... 2	est	être	VRB	VRB	_	0	root	_	_	663.409973	663.460022	Marcel_Gabbio
    ... 3	pas	pas	ADN	ADN	_	2	dep	_	_	663.469971	663.590027	Marcel_Gabbio
    ... 4	possible	possible	ADJ	ADJ	_	2	dep	_	_	663.599976	663.969971	Marcel_Gabbio
    ... 5	parce que	parce que	CSU	CSU	_	2	dep	_	_	663.979980	664.159973	Marcel_Gabbio
    ... 6       ,       ,       PUNCT   PUNCT   _       5       punct   _       _       664.150000      665.156990      Marcel_Gabbio 
    ... 7	j'	je	CLS	CLS	_	8	subj	_	_	664.169983	664.260010	Marcel_Gabbio
    ... 8	ai	avoir	VRB	VRB	_	5	dep	_	_	664.270020	664.330017	Marcel_Gabbio
    ... 9	des	de	PRE	PRE	_	8	dep	_	_	664.340027	664.419983	Marcel_Gabbio
    ... 10	choses	chose	NOM	NOM	_	9	dep	_	_	664.429993	664.820007	Marcel_Gabbio'''
    >>> tmp_file = 'test.conllu'
    >>> with open(tmp_file , "w", encoding="utf-8") as fo:
    ...     _ = fo.write(conllu_ex)
    >>>
    >>> data = load_data_conllu(tmp_file, ['lineNumber', 'words', 'lemmas','POS','UPOS','tags','HEAD','DEP','tags2','tags3','timestamp_begin','timestamp_end','speaker'], blackListed_value = {"POS": ["PUNCT"]})
    >>> data['cefc-cfpb-1000-5-260']['words'], data['cefc-cfpb-1000-5-260']['UPOS'], data['cefc-cfpb-1000-5-260']['HEAD'], data['cefc-cfpb-1000-5-260']['lineNumber']
    (["c'", 'est', 'pas', 'possible', 'parce que', "j'", 'ai', 'des', 'choses'], ['CLS', 'VRB', 'ADN', 'ADJ', 'CSU', 'CLS', 'VRB', 'PRE', 'NOM'], ['2', '0', '2', '2', '2', '7', '5', '7', '8'], ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    >>>
    """
    with open(conllu_path, "r", encoding="utf-8") as conllu_file:
        result = {}
        current_sent = ConlluDict()
        for line in conllu_file:

            line = line.strip()
            if line.startswith("#"):  # comment in conllu file
                if line.startswith("# sent_id") or line.startswith("#sent_id"):
                    sent_id = line.split()[-1]
                    current_sent = ConlluDict()
                    current_sent.set_sent_id(sent_id)
                    line_removed_index = []
                continue
            if not line:  # end of sentence
                try:
                    current_sent.set_sent_len(len(current_sent[keys[0]]))
                    if UD:
                        current_sent = add_ud_form(current_sent)
                    result[sent_id] = current_sent
                except KeyError:
                    raise KeyError(
                        f"conllu sentence has to have an 'sent_id' comment, with unique ids for all data points : {sent_id}"
                    )
                current_sent = ConlluDict()
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                fields = re.split(r"\s{2,}", line)
            if isBlacklisted(fields, keys, blackListed_value):
                # if this happen, need to reconstruct the value of any element pointing to after this lineNumber. (Remove 1)
                line_number = fields[0]
                line_removed_index.append(int(line_number))
                print(current_sent)
                if 'HEAD' not in current_sent:
                    continue
                for i, h in enumerate(current_sent['HEAD']):
                    if h >= line_number and current_sent['HEAD'][i] != "_":
                        current_sent['HEAD'][i] = str(int(current_sent['HEAD'][i]) - 1)
                continue
            if "-" in fields[0]:
                # aglutination case.
                ln1, ln2  = fields[0].split("-")
                fields[0] = f"{int(ln1) - len(line_removed_index) }-{int(ln2) - len(line_removed_index)}"
                current_sent.extend_by_keys(keys, fields)
                continue
            try:
                head = int(fields[keys.index('HEAD')])
            except (ValueError, IndexError) as e:
                print("----------------------")
                print(line)
                print(fields)
                raise e
            fields[keys.index('HEAD')] = str(head - sum([x < head for x in line_removed_index]))
            fields[keys.index('lineNumber')]=str(int(fields[keys.index('lineNumber')]) - len(line_removed_index))
            current_sent.extend_by_keys(keys, fields)
        if not current_sent.is_empty(keys):
            if UD:
                current_sent = add_ud_form(current_sent)
            result[sent_id] = current_sent
        return result

def isBlacklisted(fields: List, keys: List, blackList: dict):
    '''
    Return True if one of the column of the conllu contain a blacklisted element.
    '''
    if blackList is None:
        return False
    for b_key, item in blackList.items():
        try:
            index_fields = keys.index(b_key)
        except AttributeError as e:
            print(f" Blacklisted key : {b_key} not in the key loading list {keys}")
            raise e
        for it in item:
            if fields[index_fields] == it:
                return True
    return False





def add_ud_form(sentence: ConlluDict):
    """
    Add a new key named, ud_form containing only the form of the text needing to be embedded later on
    ie: Remove aglutination such as : 1-2 du
                                      1   de
                                      2   le

    @param sentence:
    @return:
    """
    for line_num, form in zip(sentence[list(sentence.keys())[0]], sentence["words"]):
        # Maybe testing on wether this is a number is better ?
        if "-" in line_num:
            continue
        if "ud_form" in sentence.keys():
            sentence["ud_form"].append(form)
        else:
            sentence["ud_form"] = [form]
    return sentence


if __name__ == "__main__":
    import doctest

    doctest.testmod()
