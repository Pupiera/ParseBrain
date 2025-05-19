with open(
    "results/wav2vec2_ctc_FR_AUDIO/1234/alignment_file", "r", encoding="utf-8"
) as file:
    aligned_token = []
    count = 0
    i = 0
    header_read = False
    di = {}
    for line in file:
        line = line.replace("\n", "")
        if "and  ; hypothesis ; on ; the ; third ; <eps>" in line:
            header_read = True
            continue
        if not header_read:
            continue
        if "======" in line:
            count = 0
            continue
        if count < 1:
            count += 1
            continue
        elif count == 1:
            target = line.split(";")
            count += 1
        elif count == 2:
            alignment = line.split(";")
            count += 1
        elif count == 3:
            pred = line.split(";")
            for t, a, p in zip(target, alignment, pred):
                a = a.replace("=", "C").replace(" ", "")
                if a not in di:
                    di[a] = 1
                else:
                    di[a] += 1
                t = t.replace(" ", "")
                p = p.replace(" ", "")
                if a == "I":
                    assert t == "<eps>"
                if a == "D":
                    assert p == "<eps>"
                tok = {"type": a, "ref": t, "hyp": p}
                aligned_token.append(tok)
            count = 0

    for x in aligned_token:
        if x["type"] == "C" or x["type"] == "S" or x["type"] == "I":
            if x["hyp"] == "":
                x["hyp"] = "[EMPTY_ASR_WRD]"
            print(x["hyp"])
