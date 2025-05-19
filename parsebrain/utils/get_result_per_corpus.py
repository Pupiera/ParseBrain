from parsebrain.dataio import load_data_conllu
from parsebrain.processing.dependency_parsing.eval.speech_conll18_ud_eval import (
    SpeechEval,
)
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.dataio.wer import print_alignments

from types import SimpleNamespace
import os
import argparse
import collections
from speechbrain.utils.edit_distance import accumulatable_wer_stats


def eval(path_gold, path_system, alignment):
    s_eval = SpeechEval()
    d = SimpleNamespace()
    d.gold_file = path_gold
    d.system_file = path_system
    d.alignment_file = alignment
    result = s_eval.evaluate_wrapper(d)
    return result


def split_data_per_corpus(data):
    """
    @return: A dict of dict where each key is one corpus and each corpus is a dict of key sent_id and a list of line.
    """
    res = {}
    for k, item in data.items():
        splitted = k.split("-")
        corpus = splitted[1]
        if corpus in res:
            res[corpus][k] = item
        else:
            res[corpus] = {}
            res[corpus][k] = item
    return res


def write_tmp(data: dict, path_tmp: str):
    list_key = list(data[list(data.keys())[0]].keys())
    with open(path_tmp, "w", encoding="utf-8") as file:
        for k, item in data.items():
            sent = f"# sent_id = {k}\n"
            text = " ".join(item["words"])
            sent += f"# text = {text}\n"
            line = ""
            for i in range(len(item["lineNumber"])):
                for y, inner_dict_k in enumerate(list_key):
                    # ignore seq_len (int) and sent_id name
                    if y == len(list_key) - 1 or y == 0:
                        continue
                    if y == len(list_key) - 2:
                        # don't put the \t
                        # next is seq_len
                        line += f"{item[inner_dict_k][i]}\n"
                    else:
                        line += f"{item[inner_dict_k][i]}\t"
            sent += line
            sent += "\n"
            file.write(sent)


def get_alignment(data_gold: dict, data_system: dict):
    sents_gold = []
    sents_system = []
    ids = list(data_gold.keys())
    stats = collections.Counter()

    for key, item in data_gold.items():
        system_item = data_system[key]
        sent_gold = [x.upper() for x in item["words"]]
        sent_system = [x.upper() for x in system_item["words"]]
        sents_gold.append(sent_gold)
        sents_system.append(sent_system)
    stats = accumulatable_wer_stats(sents_gold, sents_system, stats)
    details = wer_details_for_batch(
        ids=ids, refs=sents_gold, hyps=sents_system, compute_alignments=True
    )
    return details, stats


def write_res(corpus: str, result: dict, path_result: str):
    result_str = f"{corpus} & {round(result['WER'], 2)} & {round(result['UPOS'].f1, 2)} & {round(result['UAS'].f1,2)}, & {round(result['LAS'].f1,2)}\n"

    if not os.path.isfile(path_result):
        with open(path_result, "w", encoding="utf-8") as fout:
            fout.write("Corpus & WER & UPOS & UAS & LAS\n")
            fout.write(result_str)
    else:
        with open(path_result, "a", encoding="utf-8") as fout:
            fout.write(result_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file")
    parser.add_argument("system_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    path_conllu_gold = args.gold_file
    path_conllu_system = args.system_file
    result_test = args.output_file

    conllu_keys_system = [
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
    ]
    conllu_keys_gold = [
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

    data_gold = load_data_conllu(path_conllu_gold, conllu_keys_gold)
    data_system = load_data_conllu(path_conllu_system, conllu_keys_system)

    data_gold_corpus = split_data_per_corpus(data_gold)
    data_system_corpus = split_data_per_corpus(data_system)
    corpus_list = list(data_gold_corpus.keys())

    for k in corpus_list:
        path_gold_tmp = f"gold_{k}"
        path_system_tmp = f"system_{k}"
        alig_file = f"alignment_{k}"
        write_tmp(data_gold_corpus[k], path_gold_tmp)
        write_tmp(data_system_corpus[k], path_system_tmp)
        alignment, stats = get_alignment(data_gold_corpus[k], data_system_corpus[k])
        with open(alig_file, "w", encoding="utf-8") as f_out:
            print_alignments(alignment, file=f_out)
        res = eval(path_gold_tmp, path_system_tmp, alig_file)
        res = {**res, **stats}
        write_res(k, res, result_test)
