"""Check the assignment's main data and evaluation requirements."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path(__file__).resolve().parents[1]
cars = pd.read_csv(root / "data/processed/integrated_cars.csv")
checks = json.loads((root / "data/processed/data_checks.json").read_text())
assert len(cars) >= 1000 and cars["vin"].is_unique
assert cars["source"].nunique() >= 2
assert all(n >= 500 for n in checks["unique_valid_before_integration"].values())
assert 3 <= cars["make"].nunique() <= 5
assert cars.groupby("make")["model"].nunique().between(3, 5).all()
assert cars["original_msrp"].gt(0).all()
assert np.allclose(cars["retention_ratio"], cars["price_usd"] / cars["original_msrp"])
assert cars["msrp_source_url"].str.startswith("https://www.cargurus.com/research/").all()
assert cars["source_url"].notna().all() and cars["collected_at"].notna().all()
original_attributes = ["vin", "listing_id", "make", "model", "year", "trim", "price_usd",
                       "mileage", "body_type", "color", "title", "city", "state", "description"]
assert sum(cars[c].notna().any() for c in original_attributes) >= 10
msrp = pd.read_csv(root / "data/raw/msrp.csv")
for _, row in cars.iterrows():
    match = msrp[(msrp["make"] == row["make"]) & (msrp["model"] == row["model"])
                 & (msrp["year"] == row["year"]) & (msrp["trim"] == row["msrp_trim"])]
    assert row["original_msrp"] in match["original_msrp"].values
for column in ["mentions_leather", "mentions_navigation"]:
    assert cars[column].dropna().nunique() == 2

split = pd.read_csv(root / "reports/data_split.csv")
train_vins = set(split.loc[split["split"] == "train", "vin"])
test_vins = set(split.loc[split["split"] == "test", "vin"])
assert not train_vins.intersection(test_vins)
assert train_vins.union(test_vins) == set(cars["vin"])
predictions = pd.read_csv(root / "reports/test_predictions.csv")
assert set(predictions["vin"]) == test_vins
results = pd.read_csv(root / "reports/model_results.csv")
assert len(results) == 3
for _, row in results.iterrows():
    y, pred = predictions["retention_ratio"], predictions[row["model"]]
    assert np.isclose(row["test_mae"], mean_absolute_error(y, pred))
    assert np.isclose(row["test_rmse"], np.sqrt(mean_squared_error(y, pred)))
    assert np.isclose(row["test_r2"], r2_score(y, pred))
assert len(list((root / "reports/figures").glob("*.png"))) >= 6
print("PASS: source counts, model coverage, unique VINs, MSRP joins, text features, split, metrics, and figures.")
