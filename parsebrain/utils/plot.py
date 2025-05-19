import matplotlib.pyplot as plt


def create_plot_epoch(d: dict):
    epoch = d["epoch"]
    plt.xlabel("Epoch")
    plt.ylabel("Error rate and accuracy")
    for key, items in d.items():
        if key == "epoch":
            continue
        plt.plot(epoch, items, label=key)
    plt.legend()
    plt.savefig(
        "/home/getalp/pupiera/package/ParseBrain/recipe/CEFC-ORFEO/speech/sequence_labeling/test.png"
    )


def read_output_metrics_and_plot(path: str, key: list[str]):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("speechbrain.utils.train_logger"):
                continue
            fields = line.split()
            for k in key:
                for i, f in enumerate(fields):
                    if k in f:
                        if k in d:
                            d[k].append(float(fields[i + 1].replace(",", "")))
                        else:
                            d[k] = [float(fields[i + 1].replace(",", ""))]
    print(d)
    create_plot_epoch(d)


if __name__ == "__main__":
    read_output_metrics_and_plot(
        "/home/getalp/pupiera/package/ParseBrain/recipe/CEFC-ORFEO/speech/sequence_labeling/OAR.54814.stdout",
        ["epoch", "WER", "UPOS", "UAS", "LAS"],
    )
