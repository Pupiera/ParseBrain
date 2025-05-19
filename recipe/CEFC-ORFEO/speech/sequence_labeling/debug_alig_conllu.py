with open("results/wav2vec2_ctc_FR_AUDIO/1234/Eval-Valid_AUDIO") as file:
    for line in file:
        if line.startswith("#") or line == "\n":
            continue
        f = line.split("\t")
        print(f[1])
