"""Which years and which districts a model is allowed to see.

Two questions need two different splits. "Does it work next year?" is answered by
training on early years and testing on later ones. "Does it work in a district we
never monitored?" is answered by holding out whole districts. Both are reported.

Test years stay untouched while models are chosen: tuning uses the validation
years only. Attribution needs an expected level for every year, including the
training ones, so it uses cross-fitting instead: the record is cut into blocks of
years and each block is predicted by a model that never saw it.

Run from ml/:  python -m bits_ml.splits
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TRAIN_YEARS = tuple(range(2000, 2015))
VAL_YEARS = (2015, 2016, 2017)
TEST_YEARS = (2018, 2019, 2020, 2021, 2022)
FINAL_TRAIN_YEARS = TRAIN_YEARS + VAL_YEARS  # refit before touching the test years
CROSS_FIT_BLOCKS = ((2000, 2004), (2005, 2009), (2010, 2014), (2015, 2018), (2019, 2022))
N_GROUP_FOLDS = 5


@dataclass(frozen=True)
class YearSplit:
    """Row masks for one table."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def counts(self) -> dict[str, int]:
        return {"train": int(self.train.sum()), "val": int(self.val.sum()), "test": int(self.test.sum())}


def year_split(table: pd.DataFrame, train=TRAIN_YEARS, val=VAL_YEARS, test=TEST_YEARS) -> YearSplit:
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("train, validation and test years overlap")
    year = table["year"].to_numpy()
    return YearSplit(np.isin(year, train), np.isin(year, val), np.isin(year, test))


def group_folds(table: pd.DataFrame, group: str = "district", n_folds: int = N_GROUP_FOLDS, years=FINAL_TRAIN_YEARS):
    """Folds that hold out whole districts, so no well of a test district is in training."""
    rows = np.flatnonzero(np.isin(table["year"].to_numpy(), list(years)))
    groups = table[group].to_numpy()[rows]
    for fit, held in GroupKFold(n_splits=n_folds).split(rows, groups=groups):
        yield rows[fit], rows[held]


def cross_fit_blocks(table: pd.DataFrame, blocks=CROSS_FIT_BLOCKS):
    """(fit rows, predict rows) for each block of years, each block predicted out of sample.

    Also returns the years that a model for this block may use to define normals,
    so the table can be rebuilt leakage free for it.
    """
    year = table["year"].to_numpy()
    for lo, hi in blocks:
        held = (year >= lo) & (year <= hi)
        if not held.any():
            continue
        fit_years = tuple(y for y in sorted(set(year.tolist())) if not (lo <= y <= hi))
        yield np.flatnonzero(~held), np.flatnonzero(held), fit_years


def check(table: pd.DataFrame, group: str = "district") -> pd.DataFrame:
    """Row counts per split, and proof that no district and no year is on both sides."""
    split = year_split(table)
    rows = [{"split": k, "rows": v, "wells": int(table.loc[m, "well_uid"].nunique()),
             "years": f"{table.loc[m, 'year'].min()}-{table.loc[m, 'year'].max()}" if m.any() else "-"}
            for (k, v), m in zip(split.counts().items(), (split.train, split.val, split.test))]
    for i, (fit, held) in enumerate(group_folds(table, group)):
        shared = set(table[group].to_numpy()[fit]) & set(table[group].to_numpy()[held])
        if shared:
            raise ValueError(f"district fold {i} shares districts: {sorted(shared)[:3]}")
        rows.append({"split": f"district fold {i + 1}", "rows": len(held),
                     "wells": int(table.iloc[held]["well_uid"].nunique()),
                     "years": f"held-out districts: {table.iloc[held][group].nunique()}"})
    return pd.DataFrame(rows)


def main() -> None:
    from .dataset import build

    table = build()
    print(check(table).to_string(index=False))
    print("\ncross-fitting blocks (for attribution):")
    for fit, held, fit_years in cross_fit_blocks(table):
        years = table.iloc[held]["year"]
        print(f"   predict {years.min()}-{years.max()}: {len(held):,} rows, fitted on {len(fit):,} rows "
              f"from {min(fit_years)}-{max(fit_years)}")


if __name__ == "__main__":
    main()
