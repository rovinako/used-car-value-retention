# Used Car Value Retention

This project predicts how much value a used car keeps.

**Value retention = listing price / original MSRP**

## Data

The listings were scraped from CarGurus and iSeeCars. MSRP prices came from CarGurus Research. There are **1,325 cars** after cleaning, with three makes and three models per make. Both sources had over 500 unique valid listings before merging.

## What the code does

1. Combines the files and removes repeated cars.
2. Matches MSRP and adds age, leather and navigation features.
3. Fills missing values and encodes categories. Rare city and trim values are grouped together.
4. Trains ridge regression, random forest and gradient boosting on the same split.
5. Compares results and makes charts.

Gradient boosting did best: **MAE 0.0472, RMSE 0.0602, R² 0.9122**. Ridge was close. Mileage, make and age mattered most.

## Open these first

- [Notebook](notebooks/project_walkthrough.ipynb): code in steps, including Colab setup.
- [Report](reports/final_report.pdf): results and discussion.
- [Checklist](reports/assignment_checklist.txt): where to find each requirement.

## Run it

The data is included. Use Python 3.11–3.13 and run these commands from the project folder:

```bash
pip install -r requirements.txt
python run_project.py
python src/check_project.py
```

Code is in `src/`, data is in `data/`, and results are in `reports/`.

To scrape again, run `python src/collect_data.py`. It reuses cached pages. Back up the included data first; websites and listings can change.

## Limits

These are asking prices, not sale prices. MSRP may miss options or fees. The text features use short equipment descriptions, and missing text stays unknown. The sample does not cover the whole used-car market.

## AI usage

I used AI to assist with coding and help me write format the report and the files in the repo.
