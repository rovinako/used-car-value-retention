# Predicting Used Car Value Retention

This project looks at how much of a used car's original value it keeps. It uses real listings scraped from **CarGurus and iSeeCars**, plus historical MSRP tables from **CarGurus Research**.

**Value retention ratio = listing price / original MSRP**

The study covers Toyota (Camry, Corolla, RAV4), Honda (Accord, Civic, CR-V), and Ford (Fusion, Escape, Explorer), model years 2015–2025. Prices are US dollars and distances are miles.

The saved October 23, 2026 snapshot has **1,325 unique cars** after cleaning and MSRP matching. Before integration, the sources supplied **887** valid unique CarGurus listings and **736** valid unique iSeeCars listings. Gradient boosting had the lowest training cross-validation RMSE; its test MAE was **0.0472**, RMSE **0.0602**, and R² **0.9122**. Mileage, make, and vehicle age were the strongest predictors.

## Start here

- [Final report](reports/final_report.pdf): methods, results, charts, discussion, limitations, and AI usage.
- [Notebook](notebooks/project_walkthrough.ipynb): a step-by-step way to run the project, including in Google Colab.
- [Model results](reports/model_results.csv) and [data requirement checks](data/processed/data_checks.json).
- [Assignment checklist](reports/assignment_checklist.txt): where to find each required part.

## Run with the included data

Use Python 3.11–3.13. From this repository's folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_project.py
python src/check_project.py
```

On Windows, use `.venv\Scripts\activate` instead of the `source` command.

The raw data is included, so **you do not need to scrape again** to reproduce the experiment. The main script cleans the data, trains the models, and builds the report. Exact package versions used for the saved results are listed in `reports/environment.txt`.

## Collect a fresh sample (optional)

```bash
python src/collect_data.py
python run_project.py
```

The scraper uses Requests and BeautifulSoup to parse public HTML and the structured data embedded in it. It waits between requests and caches successful responses in `.cache/`. Existing cached pages are reused. Remove that local cache only if you intentionally want a fresh collection. Websites change and may block requests; the script records failed pages and does not bypass CAPTCHAs or access restrictions. Fresh results will differ from the included snapshot. The script writes raw CSV files, so back up the supplied data before collecting again.

## Files

- `src/collect_data.py`: both listing scrapers and the MSRP scraper.
- `src/prepare_data.py`: combines field names, checks values, removes duplicate VINs, matches MSRP, and creates features.
- `src/train_models.py`: ridge regression, random forest, gradient boosting, evaluation, and six charts.
- `src/make_report.py`: builds the report using saved results.
- `src/check_project.py`: checks the main assignment requirements and recalculates the evaluation metrics.
- `data/raw/cargurus_listings.csv` and `data/raw/iseecars_listings.csv`: initially parsed records, including repeated listings.
- `data/raw/msrp.csv`: original MSRP by make, model, year, and trim, with source URLs.
- `data/raw/collection_log.csv`: page URLs, collection timestamps, and HTTP status codes.
- `data/raw/collection_errors.csv`: unsuccessful collection attempts.
- `data/processed/integrated_cars.csv`: final, unique, MSRP-matched dataset.
- `data/processed/cleaning_steps.csv`: record counts after each cleaning step.
- `data/processed/excluded_msrp.csv`: cars excluded because MSRP was unavailable or the trim match was ambiguous.
- `reports/`: metrics, predictions, split IDs, feature rankings, figures, and final report.

## Main choices

The same 80/20 split is used for all three models. Three-fold cross-validation uses only the training set to choose the model. Imputation, scaling, and encoding are fitted inside each training fold. High-cardinality trim and city values use one-hot encoding with rare categories grouped together.

Two text features detect leather and navigation in the listing's short equipment descriptions. These are descriptive equipment phrases, **not full seller-written paragraphs**. iSeeCars search cards do not expose this text, so missing descriptions are recorded as unknown. A phrase not appearing is not proof that a vehicle lacks the equipment.

The target uses historical trim MSRP rather than a dealer's crossed-out price. MSRP matches must be exact by trim ID/name or unambiguous after simple formatting changes. The code does not fill unmatched cars with an average MSRP. CarGurus listing prices use the base asking price where fees are separately available; iSeeCars prices follow the amount shown on the card. Fee treatment, optional packages, and destination charges can still differ.

Listing price, the retention ratio, VIN, URLs, and dealer price estimates are **not predictors**. Original MSRP is a known vehicle specification and is allowed as a predictor. VIN is used to remove duplicates before splitting.

These are advertised prices, not completed sale prices. The sample is a convenience sample from search pages, not a random sample of the whole used-car market. Associations and feature importance do not prove what causes depreciation.

## AI usage

OpenAI Codex helped create the repository, research accessible sources, write and run the collection and analysis code, check the outputs, and draft the report. The student supplied the assignment and requested simple code and wording. No part of the generated code, data collection, analysis, or report is claimed to have been independently completed by the student. The student should review the work, understand the code, and update this statement with any independent changes before submitting.
