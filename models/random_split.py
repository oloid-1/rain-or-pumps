"""
A random train/valid/test split, built because the year-blocked split
(train 2000-2014, valid 2015-2017, test 2018-2022) is not what was asked for
this round: the mentor wants the validation rows chosen at random, not by
calendar year.

The one thing a plain `train_test_split` on all 216,455 rows would get wrong:
each of the 2,759 wells contributes about 78 rows, and consecutive readings
from the same well are near-duplicates of each other (same lat/lon, same
aquifer, same well depth, overlapping rain history). A row-level random split
puts some of a well's readings in train and others in valid, so the model
can partly recognise the well rather than generalise to it, and every metric
below comes out optimistic. That is leakage with a different shape than the
year-block split had.

So this is a *grouped* random split: the group is `well_uid`, assigned
whole to one split. It is still "random, not by year" exactly as asked -
every well is placed by a shuffle, not by when it happened to be read - it
just does the shuffle at the unit that keeps rows independent.

Target shares: 70% train / 15% valid / 15% test, by well count, seed 42.
"""

import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
from build_training_data import OUT_DIR as T  # noqa: E402
SEED = 42


def make_random_split(tab, seed=SEED, valid_frac=0.15, test_frac=0.15):
    wells = tab["well_uid"].drop_duplicates().to_numpy()
    rng = np.random.RandomState(seed)
    rng.shuffle(wells)

    n = len(wells)
    n_test = int(round(n * test_frac))
    n_valid = int(round(n * valid_frac))
    test_wells = set(wells[:n_test])
    valid_wells = set(wells[n_test:n_test + n_valid])
    # everything else is train

    def label(w):
        if w in test_wells:
            return "test"
        if w in valid_wells:
            return "valid"
        return "train"

    split = tab["well_uid"].map(label)
    return split.astype(str)


def build_and_save():
    tab = pd.read_parquet(T / "tabular.parquet")
    tab["split_random"] = make_random_split(tab)

    out = tab[["well_uid", "split_random"]].copy()
    out.to_csv(T / "random_split_assignment.csv", index=False)

    report = []
    report.append("Random (well-grouped) validation split")
    report.append("=" * 50)
    report.append("")
    report.append("Why grouped by well_uid, not a plain random row split:")
    report.append("  each well contributes ~78 near-duplicate rows (same location,")
    report.append("  same aquifer, overlapping rain history). A row-level shuffle")
    report.append("  would leak the same well into train and valid. Grouping by")
    report.append("  well_uid keeps the split 'random, not by year' while keeping")
    report.append("  each well's rows on one side only.")
    report.append("")
    counts = tab["split_random"].value_counts()
    wcounts = tab.groupby("split_random")["well_uid"].nunique()
    report.append(f"{'split':<8}{'rows':>10}{'wells':>10}{'row %':>10}")
    for k in ["train", "valid", "test"]:
        report.append(f"{k:<8}{counts[k]:>10,}{wcounts[k]:>10,}{100*counts[k]/len(tab):>9.1f}%")
    report.append("")

    # leakage check: no well in two splits
    cross = tab.groupby("well_uid")["split_random"].nunique()
    bad = int((cross > 1).sum())
    report.append(f"wells appearing in more than one split: {bad} (must be 0)")
    report.append("")

    # sanity: target distribution should look similar across splits (it's a
    # random split, so large differences would mean a bug, not a finding)
    report.append("delta_h_m by split (mean / std) - should be close across splits:")
    g = tab.groupby("split_random")["delta_h_m"].agg(["mean", "std", "count"])
    for k in ["train", "valid", "test"]:
        r = g.loc[k]
        report.append(f"  {k:<6} mean {r['mean']:+.3f}  std {r['std']:.3f}  n {int(r['count']):,}")
    report.append("")

    # compare against the year-blocked split so the difference is on record
    report.append("For reference, the year-blocked split it replaces (this run):")
    g2 = tab.groupby("split")["delta_h_m"].agg(["mean", "std", "count"])
    for k in ["train", "valid", "test"]:
        r = g2.loc[k]
        report.append(f"  {k:<6} mean {r['mean']:+.3f}  std {r['std']:.3f}  n {int(r['count']):,}")

    text = "\n".join(report)
    (T / "random_split_report.txt").write_text(text)
    print(text)
    return tab


if __name__ == "__main__":
    build_and_save()
