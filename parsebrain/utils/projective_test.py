from parsebrain.dataio import load_data_conllu

conll_key = [
    "lineNumber",
    "words",
    "lemmas",
    "POS",
    "UPOS",
    "tags",
    "HEAD",
    "DEP",
    "tags2",
    "tags3",
    "timestamp_begin",
    "timestamp_end",
    "speaker",
]


def is_projective(data: dict):
    """
    >>> data = {'lineNumber': [1,2,3,4], 'HEAD': [4, 4, 1, 0]}
    >>> is_projective(data)
    False
    >>> data = {'lineNumber': [1,2,3,4], 'HEAD': [0, 4, 1, 1]}
    >>> is_projective(data)
    False
    >>> data = {'lineNumber': [1,2,3,4,5], 'HEAD':[2, 0, 2, 5, 3]}
    >>> is_projective(data)
    True
    >>> data = {'lineNumber': [1,2,3,4,5,6,7,8], 'HEAD':[4,4,4,0,6,4,6,7]}
    >>> is_projective(data)
    True
    >>> data = {'lineNumber': [1,2,3,4,5],'HEAD': [3,3,0,3,3]}
    >>> is_projective(data)
    True
    >>> data = {'lineNumber': list(range(1,16)), 'HEAD' : [9,1,2,3,9,9,9,9,0,9,10,10,14,12,8]}
    >>> is_projective(data)
    False
    >>> data = {'lineNumber':list(range(1,16)), 'HEAD': [7, 7, 4, 2, 6, 7, 0, 7, 11, 11, 8, 4, 14, 12, 14, 15]}
    >>> is_projective(data)
    False

    test with sentence : cefc-coralrom-ffamdl06-107 and cefc-coralrom-fnatps01-73, cefc-tcof-Aqua_05-186
    """
    print(data["HEAD"])
    if data["HEAD"][0] == 7:
        print("x")
    for i, h in zip(data["lineNumber"], data["HEAD"]):
        # root case
        if h == 0:
            continue
        i = int(i)
        if i == 12:
            print("z")
        h = int(h)
        if i < h:
            r1 = list(range(i + 1, h))
        else:
            r1 = list(range(h + 1, i))
        for y in r1:
            y_head = int(data["HEAD"][y - 1])
            if y_head == 0:
                continue
            if i < h:
                r2 = list(range(i, h + 1))
            else:
                r2 = list(range(h, i + 1))
            if y_head not in r2 and y_head != 0:
                return False
    return True


def write_conllu_from_dict(key: str, data: dict, f):
    f.write(f"# sent_id = {key}\n")
    for i in range(len(data[conll_key[0]])):
        l = [data[k][i] for k in conll_key]
        f.write("\t".join(l) + "\n")
    f.write("\n")


def write_list(l: list, path):
    with open(path, "w", encoding="utf-8") as f:
        for (key, data) in l:
            write_conllu_from_dict(key, data, f)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    """
    conllu_path = "/home/getalp/pupiera/thesis/endtoend_asr_multitask/src/conllu/gold/orfeo_shuf.train"
    proj_path = "proj"
    non_proj_path = "non_proj"
    conllu = load_data_conllu(conllu_path, conll_key)
    proj = []
    non_proj = []
    for key, item in conllu.items():
        if is_projective(item):
            proj.append((key, item))
        else: 
            non_proj.append((key, item))
    write_list(proj, proj_path)
    write_list(non_proj, non_proj_path)
    """
