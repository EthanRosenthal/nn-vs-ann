import pandas as pd

from nn_vs_ann import ROOT


def main():
    """Generate the README from a template."""
    results = pd.read_csv(ROOT / "assets" / "results.csv")
    reshaped = (
        results.groupby(["num_embeddings", "lib"])["avg_time"]
        .mean()
        .unstack(level=1)
        .reset_index()
    )
    reshaped["num_embeddings"] = reshaped["num_embeddings"].apply(lambda x: f"{x:,}")

    template = ROOT / "templates" / "README.md.template"
    readme = ROOT / "README.md"
    with template.open("r") as fin, readme.open("w") as fout:
        fout.write(fin.read().format(table=reshaped.to_markdown(index=False)))


if __name__ == "__main__":
    main()
