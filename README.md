# Predicting Used Car Value Retention

This project will use car listings from different websites to predict how much of a car's original value it keeps and look at what affects depreciation.

**Value retention ratio = current listing price / original MSRP**

## Status

Project setup only. Data collection, model training, and results are not completed yet.

## Project checklist

- Scrape at least two independent listing websites, with at least 500 listings from each.
- Keep at least 1,000 listings after cleaning and removing duplicates.
- Include 3–5 makes and 3–5 models per make.
- Collect at least 10 original attributes, including listing descriptions.
- Collect reliable MSRP data by make, model, and model year, then merge it with the listings. Record the source URLs and collection dates.
- Compare attribute statistics and fix inconsistent names, units, formats, and missing values.
- Create vehicle age and at least two description features, such as one-owner and accident mentions.
- Apply and discuss an encoding method for high-cardinality features.
- Compare at least three regression models using a train/test split or cross-validation and at least two metrics, such as MAE, RMSE, and R².
- Fit preprocessing on training data only. Do not use listing price or anything calculated from the target as a predictor.
- Study feature importance and the effects of age, mileage, and other factors.
- Plot data distributions, feature rankings, and model performance.

## Planned folders

- `data/raw/`: separate parsed files from each listing source and MSRP data.
- `data/processed/`: cleaned and combined data.
- `src/`: collection, cleaning, feature engineering, and model code.
- `notebooks/`: exploration and experiments.
- `reports/`: final report and figures.

## Final report

Cover the project goal, data collection, preprocessing and integration, feature engineering, model development, evaluation, depreciation analysis, and visualizations. Finish with the main findings, what was learned, challenges, and limitations.

## AI usage statement

OpenAI Codex helped create this repository and draft this project outline. Update this section as work continues to explain which tasks used AI and which were completed independently.
