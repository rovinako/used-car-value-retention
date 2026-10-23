"""Compare three regression models and save results and figures."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
NUMERIC = ["vehicle_age", "mileage", "original_msrp", "mentions_leather",
           "mentions_navigation", "description_available"]
CATEGORICAL = ["make", "model", "trim", "body_type", "color", "city", "source"]


def preprocessing():
    numbers = Pipeline([("fill_missing", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categories = Pipeline([
        ("fill_missing", SimpleImputer(strategy="constant", fill_value="unknown")),
        # Group rare trim/city values to avoid hundreds of tiny categories.
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=10, sparse_output=False)),
    ])
    return ColumnTransformer([("numeric", numbers, NUMERIC),
                              ("category", categories, CATEGORICAL)])


def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data/processed/integrated_cars.csv")
    assert data["vin"].is_unique, "Duplicate vehicles must be removed before splitting."
    X, y = data[NUMERIC + CATEGORICAL].copy(), data["retention_ratio"]
    X[NUMERIC] = X[NUMERIC].astype(float)
    # Stratifying by make/model keeps all nine models represented in both sets.
    train_ids, test_ids = train_test_split(data.index, test_size=0.2,
                                         random_state=42, stratify=data["make_model"])
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    data[["vin"]].assign(split=np.where(data.index.isin(train_ids), "train", "test")).to_csv(
        REPORTS / "data_split.csv", index=False)
    models = {
        "Ridge regression": Ridge(alpha=10.0),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                               max_features=0.8, random_state=42, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                                       max_depth=3, min_samples_leaf=5, random_state=42),
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    results, fitted = [], {}
    predictions = data.loc[test_ids, ["vin", "make", "model", "source", "retention_ratio"]].copy()
    for name, model in models.items():
        pipeline = Pipeline([("prepare", preprocessing()), ("model", model)])
        # Preprocessing is refitted inside each fold, never on the held-out rows.
        scores = -cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="neg_root_mean_squared_error", n_jobs=1)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        predictions[name] = pred
        fitted[name] = pipeline
        results.append({"model": name, "cv_rmse": scores.mean(), "cv_rmse_sd": scores.std(),
                        "test_mae": mean_absolute_error(y_test, pred),
                        "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
                        "test_r2": r2_score(y_test, pred)})
        print(name, results[-1], flush=True)
    scores = pd.DataFrame(results)
    scores.to_csv(REPORTS / "model_results.csv", index=False)
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
    # Choose using training CV, not the test scores.
    best_name = scores.loc[scores["cv_rmse"].idxmin(), "model"]
    best = fitted[best_name]
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    summary = {"best_by_training_cv": best_name, "training_rows": len(train_ids),
               "test_rows": len(test_ids), "dummy_test_mae": mean_absolute_error(y_test, dummy_pred),
               "dummy_test_rmse": np.sqrt(mean_squared_error(y_test, dummy_pred)),
               "dummy_test_r2": r2_score(y_test, dummy_pred),
               "encoded_columns": int(best.named_steps["prepare"].transform(X_train).shape[1]),
               "train_unique_trims": int(X_train["trim"].nunique()),
               "train_unique_cities": int(X_train["city"].nunique())}
    (REPORTS / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(best, REPORTS / "best_model.joblib")

    importance = permutation_importance(best, X_test, y_test, n_repeats=10,
                                       random_state=42, scoring="neg_root_mean_squared_error")
    ranking = pd.DataFrame({"feature": X.columns, "rmse_increase": importance.importances_mean,
                            "std": importance.importances_std}).sort_values("rmse_increase", ascending=False)
    ranking.to_csv(REPORTS / "feature_importance.csv", index=False)
    ridge = fitted["Ridge regression"]
    names = ridge.named_steps["prepare"].get_feature_names_out()
    pd.DataFrame({"feature": names, "coefficient": ridge.named_steps["model"].coef_}).to_csv(
        REPORTS / "ridge_coefficients.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col, title, label in zip(axes, ["price_usd", "mileage", "retention_ratio"],
                                    ["Listing prices", "Mileage", "Value retention"],
                                    ["USD", "Miles", "Price / original MSRP"]):
        ax.hist(data[col], bins=25, color="#3678a8", edgecolor="white")
        ax.set(title=title, xlabel=label, ylabel="Cars")
    save_figure("01_distributions.png")

    counts = data.groupby(["make_model", "source"]).size().unstack(fill_value=0)
    counts.plot.barh(stacked=True, figsize=(9, 5), color=["#3678a8", "#e59f45"])
    plt.xlabel("Unique cars after cleaning and MSRP matching")
    plt.ylabel("")
    plt.title("Listings by model and source")
    save_figure("02_model_counts.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric, label in zip(axes, ["test_mae", "test_rmse", "test_r2"], ["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"]):
        ax.bar(["Ridge", "Forest", "Boosting"], scores[metric], color=["#3678a8", "#e59f45", "#4c956c"])
        ax.set_title(label)
        for i, value in enumerate(scores[metric]):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
        ax.margins(y=0.2)
    save_figure("03_model_performance.png")

    plot_ranking = ranking.sort_values("rmse_increase")
    plt.figure(figsize=(9, 5))
    plt.barh(plot_ranking["feature"], plot_ranking["rmse_increase"], xerr=plot_ranking["std"], color="#3678a8")
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("Increase in test RMSE when feature is shuffled")
    plt.title(f"Feature importance: {best_name}")
    save_figure("04_feature_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ["vehicle_age", "mileage"], ["Vehicle age (years)", "Mileage (miles)"]):
        ax.scatter(data[col], data["retention_ratio"], s=10, alpha=0.3, color="#3678a8")
        ax.set(xlabel=label, ylabel="Price / original MSRP")
    fig.suptitle("Observed depreciation patterns (not causal effects)")
    save_figure("05_depreciation.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(best, X_train, ["vehicle_age", "mileage"],
                                           ax=axes, grid_resolution=20)
    fig.suptitle("Model's average predicted retention when one input changes", y=1.03)
    save_figure("06_partial_dependence.png")

    by_source = []
    for source in predictions["source"].unique():
        part = predictions[predictions["source"] == source]
        by_source.append({"source": source, "test_rows": len(part),
                          "mae": mean_absolute_error(part["retention_ratio"], part[best_name]),
                          "rmse": np.sqrt(mean_squared_error(part["retention_ratio"], part[best_name]))})
    pd.DataFrame(by_source).to_csv(REPORTS / "performance_by_source.csv", index=False)
    print("Selected by training CV:", best_name)


if __name__ == "__main__":
    main()
