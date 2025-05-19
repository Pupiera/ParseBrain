HEAD = 6


def remove_multi_root(conll_dict: dict):
    new_data = {}
    for key, item in conll_dict.items():
        if is_multi_root(item):
            continue
        new_data[key] = item
    return new_data


def is_multi_root(sent: list):
    head = [x[HEAD] for x in sent]
    return head.count("0") > 1


def has_cyle(sent: list):
    head = [int(x[HEAD]) for x in sent]
    for i, h in enumerate(head, start=1):
        visited = [i]
        if h == 0:
            continue
        h1 = h
        visited.append(h1)
        while h1 != 0:
            # minus 1 because index start at 1 in conll
            h1 = head[h1 - 1]
            # check if an element has already been seen
            if h1 in visited:
                return True
            visited.append(h1)
    return False


def remove_cycle(conll_dict):
    new_data = {}
    for key, item in conll_dict.items():
        if has_cyle(item):
            continue
        new_data[key] = item
    return new_data


def write_conll(conll_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as out:
        for key, items in conll_dict.items():
            out.write(f"# sent_id = {key}\n")
            for it in items:
                out.write("\t".join(it))
            out.write("\n")


if __name__ == "__main__":
    conll_path = "/home/getalp/pupiera/thesis/endtoend_asr_multitask/src/conllu/gold/orfeo_shuf.dev"
    data = {}
    sentence = []
    output_path = "/home/getalp/pupiera/thesis/endtoend_asr_multitask/src/conllu/gold/orfeo_shuf_no_multiroot.dev"
    with open(conll_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=")[-1].strip()
                continue
            if line == "\n":
                data[sent_id] = sentence
                sent_id = None
                sentence = []
                continue
            sentence.append(line.split("\t"))
    data = remove_multi_root(data)
    data = remove_cycle(data)
    write_conll(data, output_path)
