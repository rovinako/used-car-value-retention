"""Clean both sources, remove duplicate cars, merge MSRP, and make features."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
TARGETS = {"Toyota": ["Camry", "Corolla", "RAV4"],
           "Honda": ["Accord", "Civic", "CR-V"],
           "Ford": ["Fusion", "Escape", "Explorer"]}


def normalize_trim(text):
    """Ignore formatting only; do not guess between trims or engines."""
    text = str(text).upper()
    text = re.sub(r"\b(SEDAN|SUV|4DR|4D)\b", " ", text)
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", text).split())


def description_flag(text, pattern):
    # NaN means no description was available. Zero means no matching phrase.
    flags = text.fillna("").str.contains(pattern, case=False, regex=True).astype(float)
    flags[text.fillna("").str.strip().eq("")] = np.nan
    return flags


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cg = pd.read_csv(RAW / "cargurus_listings.csv")
    isee_raw = pd.read_csv(RAW / "iseecars_listings.csv")
    isee = isee_raw.rename(columns={
        "id": "listing_id", "VIN": "vin", "manufacturer": "make",
        "model_name": "model", "model_year": "year", "trim_name": "trim",
        "asking_price": "price_usd", "odometer": "mileage",
        "distance_unit": "mileage_unit", "body_style": "body_type",
        "exterior_color": "color", "listing_title": "title"})
    location = isee["location"].str.rsplit(",", n=1, expand=True)
    isee["city"], isee["state"] = location[0].str.strip(), location[1].str.strip()
    data = pd.concat([cg, isee], ignore_index=True)
    audit = [{"step": "Parsed rows", "rows": len(data)}]

    for column in ["year", "price_usd", "mileage"]:
        data[column] = pd.to_numeric(data[column].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    # Both collected sources use miles; keep conversion explicit for future runs.
    km = data["mileage_unit"].fillna("").str.lower().isin(["km", "kilometers"])
    data.loc[km, "mileage"] *= 0.621371
    data["mileage_unit"] = "miles"
    for column in ["make", "model", "trim", "city", "state"]:
        data[column] = data[column].astype("string").str.strip()
    for column in ["body_type", "color"]:
        data[column] = data[column].astype("string").str.strip().str.lower()
    data["vin"] = data["vin"].fillna("").str.strip().str.upper()

    selected = data.apply(lambda r: r["model"] in TARGETS.get(r["make"], []), axis=1)
    data = data[selected & data["year"].between(2015, 2025)].copy()
    audit.append({"step": "Selected makes, models and years 2015-2025", "rows": len(data)})
    valid = (data["price_usd"].between(1000, 150000)
             & data["mileage"].between(0, 500000)
             & data["vin"].str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}"))
    data = data[valid].copy()
    data["year"] = data["year"].astype(int)
    audit.append({"step": "Valid price, mileage and VIN", "rows": len(data)})
    data.drop_duplicates(["source", "vin"], inplace=True)
    audit.append({"step": "Unique VIN within each source", "rows": len(data)})
    source_counts = data.groupby("source").size().to_dict()
    # Retain the richer CarGurus record when the same VIN appears on both sites.
    duplicates = data[data.duplicated("vin", keep=False)].sort_values("vin")
    duplicates.to_csv(OUT / "cross_source_duplicates.csv", index=False)
    data.sort_values("source", inplace=True)
    data.drop_duplicates("vin", inplace=True)
    audit.append({"step": "Unique VIN across both sources", "rows": len(data)})

    msrp = pd.read_csv(RAW / "msrp.csv").drop_duplicates(["make", "model", "year", "trim_id"])
    msrp["trim_key"] = msrp["trim"].map(normalize_trim)
    # A shortened trim name is accepted only when every candidate has the same MSRP.
    msrp["short_key"] = msrp["trim_key"].str.replace(r"\b(FWD|AWD|RWD|4WD|2WD)\b", "", regex=True).str.split().str.join(" ")
    matched, unmatched = [], []
    for _, row in data.iterrows():
        options = msrp[(msrp["make"] == row["make"]) & (msrp["model"] == row["model"])
                       & (msrp["year"] == row["year"])]
        trim_key = normalize_trim(row["trim"])
        method = "exact trim ID"
        match = options[options["trim_id"] == row.get("trim_id")]
        if match.empty:
            method = "exact normalized trim name"
            match = options[options["trim_key"] == trim_key]
        if match.empty and not re.search(r"\b(FWD|AWD|RWD|4WD|2WD)\b", trim_key):
            method = "unambiguous shortened trim name"
            short = " ".join(re.sub(r"\b(FWD|AWD|RWD|4WD|2WD)\b", "", trim_key).split())
            match = options[options["short_key"] == short]
        if match.empty or match["original_msrp"].nunique() != 1:
            row["exclusion_reason"] = "MSRP unavailable or trim match ambiguous"
            unmatched.append(row.to_dict())
            continue
        row["original_msrp"] = match.iloc[0]["original_msrp"]
        row["msrp_source_url"] = match.iloc[0]["msrp_source_url"]
        row["msrp_trim"] = match.iloc[0]["trim"]
        row["msrp_match_method"] = method
        matched.append(row.to_dict())
    data = pd.DataFrame(matched)
    pd.DataFrame(unmatched).to_csv(OUT / "excluded_msrp.csv", index=False)
    audit.append({"step": "Matched historical MSRP without guessing", "rows": len(data)})

    data["vehicle_age"] = pd.to_datetime(data["collected_at"], utc=True).dt.year - data["year"]
    data["miles_per_year"] = data["mileage"] / data["vehicle_age"].clip(lower=1)
    data["make_model"] = data["make"] + " " + data["model"]
    data["mentions_leather"] = description_flag(data["description"], r"\bleather\b")
    data["mentions_navigation"] = description_flag(data["description"], r"\bnavigation\b|\bGPS\b")
    data["description_available"] = data["description"].fillna("").str.strip().ne("").astype(int)
    data["retention_ratio"] = data["price_usd"] / data["original_msrp"]
    data.sort_values(["make", "model", "year", "vin"], inplace=True)
    data.to_csv(OUT / "integrated_cars.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "cleaning_steps.csv", index=False)
    data.describe(include="all").to_csv(OUT / "attribute_statistics.csv")
    data.isna().mean().rename("missing_fraction").to_csv(OUT / "missing_values.csv")
    data.groupby(["make", "model", "source"]).size().rename("listings").to_csv(OUT / "model_counts.csv")

    checks = {
        "raw_rows": {"CarGurus": len(cg), "iSeeCars": len(isee_raw)},
        "unique_valid_before_integration": source_counts,
        "final_rows": len(data), "final_source_counts": data["source"].value_counts().to_dict(),
        "makes": sorted(data["make"].unique().tolist()),
        "models_per_make": data.groupby("make")["model"].nunique().to_dict(),
        "duplicate_vins": int(data["vin"].duplicated().sum()),
        "missing_msrp": int(data["original_msrp"].isna().sum()),
        "text_available": int(data["description_available"].sum()),
        "text_feature_counts": {c: int(data[c].eq(1).sum()) for c in ["mentions_leather", "mentions_navigation"]},
    }
    (OUT / "data_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    assert all(n >= 500 for n in source_counts.values()), "Need 500 unique valid rows from EACH source."
    assert len(source_counts) >= 2 and data["source"].nunique() >= 2
    assert len(data) >= 1000, "Need at least 1,000 usable, unique, MSRP-matched cars."
    assert data["vin"].is_unique and data["original_msrp"].gt(0).all()
    assert 3 <= data["make"].nunique() <= 5
    assert data.groupby("make")["model"].nunique().between(3, 5).all()


if __name__ == "__main__":
    main()
