"""Download the M5 (Walmart) dataset and write the small subset the exercises use.

Writes, for 50 steady-selling items in store CA_1:
    data/processed/m5_ca1_daily.csv   item_id, cat_id, date, wm_yr_wk, units, price
    data/processed/m5_ca1_weekly.csv  item_id, cat_id, wm_yr_wk, week_start, units, price

The raw data comes from Nixtla's public copy of the M5 competition files.
Run: .venv/bin/python data/prepare_m5.py
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "m5"
OUT = ROOT / "processed"
STORE = "CA_1"
PICK = {"FOODS": 20, "HOUSEHOLD": 15, "HOBBIES": 15}  # items per category
SEED = 42


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "m5.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
    needed = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
    if not all((RAW / f).exists() for f in needed):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW, members=needed)


def pick_items(sales, days):
    """Items that sold every week of the last 2 years, averaging 10-150 units a week."""
    last_two_years = sales[days[-728:]].to_numpy()
    weekly = last_two_years.reshape(len(sales), -1, 7).sum(axis=2)
    ok = (weekly.min(axis=1) > 0) & (weekly.mean(axis=1) >= 10) & (weekly.mean(axis=1) <= 150)
    eligible = sales[ok]
    rng = np.random.default_rng(SEED)
    chosen = [rng.choice(eligible.loc[eligible.cat_id == cat, "item_id"].to_numpy(), n, replace=False)
              for cat, n in PICK.items()]
    return sorted(np.concatenate(chosen))


def main():
    download()
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[sales.store_id == STORE].reset_index(drop=True)
    days = [c for c in sales.columns if c.startswith("d_")]
    items = pick_items(sales, days)

    calendar = pd.read_csv(RAW / "calendar.csv", usecols=["date", "wm_yr_wk"])
    calendar["d"] = [f"d_{i}" for i in range(1, len(calendar) + 1)]  # this copy has no "d" column; rows are in day order
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[(prices.store_id == STORE) & prices.item_id.isin(items)].drop(columns="store_id")

    daily = (sales[sales.item_id.isin(items)]
             .melt(id_vars=["item_id", "cat_id"], value_vars=days, var_name="d", value_name="units")
             .merge(calendar, on="d").drop(columns="d")
             .merge(prices, on=["item_id", "wm_yr_wk"], how="left")
             .rename(columns={"sell_price": "price"})
             .sort_values(["item_id", "date"]))
    # Drop days before an item was first sold (no price yet)
    daily = daily.dropna(subset=["price"])

    weekly = (daily.groupby(["item_id", "cat_id", "wm_yr_wk"])
              .agg(week_start=("date", "min"), days=("date", "size"), units=("units", "sum"), price=("price", "mean"))
              .reset_index())
    weekly = weekly[weekly.days == 7].drop(columns="days")  # full weeks only

    OUT.mkdir(exist_ok=True)
    daily.to_csv(OUT / "m5_ca1_daily.csv", index=False)
    weekly.to_csv(OUT / "m5_ca1_weekly.csv", index=False)
    print(f"{len(items)} items -> {OUT / 'm5_ca1_daily.csv'} ({len(daily):,} rows), "
          f"{OUT / 'm5_ca1_weekly.csv'} ({len(weekly):,} rows)")


if __name__ == "__main__":
    main()
