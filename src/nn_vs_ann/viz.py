import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from nn_vs_ann import ROOT


def main(results_file: Union[Path, str]):
    results = pd.read_csv(results_file)
    results["num_embeddings"] = results["num_embeddings"].astype("int64")
    fig, ax = plt.subplots(1, 1)

    reshaped = (
        results.groupby(["num_embeddings", "lib"])["avg_time"].mean().unstack(level=1)
    )
    reshaped.plot(kind="barh", ax=ax)
    ax.semilogx()

    ax.set_xlabel("Time (s) (log scale)")
    ax.set_ylabel("Number of Embeddings")
    ax.set_yticklabels([f"{int(label.get_text()):,}" for label in ax.get_yticklabels()])
    fig.suptitle("Top-k query time (k=5)")
    fig.tight_layout()

    plt.savefig("assets/results.png", dpi=200)


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = ROOT / "assets" / "results.csv"
    main(filename)
