# Used Car Value Retention

This project predicts how much of its original value a used car keeps.

**Value retention = listing price / original MSRP**

## What is included?

- Real listings scraped from **CarGurus and iSeeCars**.
- **1,325 unique cars** after cleaning and matching MSRP.
- Three makes: Toyota, Honda and Ford, with three models each.
- Raw data, cleaned data, code, six charts and a final report.
- Three models: ridge regression, random forest and gradient boosting.

Each source had over 500 unique valid listings before merging (887 CarGurus and 736 iSeeCars). MSRP comes from CarGurus Research.

## Start here

1. Open the [notebook](notebooks/project_walkthrough.ipynb) for the steps.
2. Read the [final report](reports/final_report.pdf).
3. See the [assignment checklist](reports/assignment_checklist.txt).

## Run the project

The data is already included. From the project folder, with Python 3.11–3.13:

```bash
pip install -r requirements.txt
python run_project.py
python src/check_project.py
```

The notebook also has instructions for Google Colab.

## Main steps

1. Clean column names and remove duplicate cars using VIN.
2. Match each car with MSRP by make, model, year and trim.
3. Create age and two text features: leather and navigation mentions.
4. Fill missing values, scale numbers and one-hot encode categories. Rare city and trim values are grouped together.
5. Use the same 80/20 split for all models. Choose a model using three-fold training cross-validation.
6. Compare MAE, RMSE and R², then plot feature importance.

## Results

Gradient boosting did best: **MAE 0.0472, RMSE 0.0602, R² 0.9122**. Ridge was close. Mileage, make and age mattered most.

These are asking prices, not sale prices. MSRP may miss options and fees. Text features use short equipment descriptions, not full seller paragraphs. Missing text is recorded as unknown. The sample covers selected search pages, so it does not represent every used car.

## Files

- `src/`: scraping, cleaning, training, report and checking code.
- `data/raw/`: separate source files, MSRP and source links.
- `data/processed/`: cleaned data, statistics and cleaning counts.
- `reports/`: results, charts and report.
- `notebooks/`: step-by-step notebook.

To collect again, run `python src/collect_data.py`. It reuses cached pages in `.cache/`. Back up the supplied raw files first. Listings change and sites may block scraping, so new results may differ.

## AI usage

OpenAI Codex wrote and ran the code, checked results and drafted the report. The student supplied the assignment and requested simple wording. Generated work is not claimed as independent student work. Add any work you personally complete before submitting.
