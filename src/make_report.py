"""Make a short PDF report from the saved data, metrics, and figures."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
WIDTH = 510


def main():
    cars = pd.read_csv(ROOT / "data/processed/integrated_cars.csv")
    checks = json.loads((ROOT / "data/processed/data_checks.json").read_text())
    steps = pd.read_csv(ROOT / "data/processed/cleaning_steps.csv")
    results = pd.read_csv(REPORTS / "model_results.csv")
    ranking = pd.read_csv(REPORTS / "feature_importance.csv")
    summary = json.loads((REPORTS / "experiment_summary.json").read_text())
    best_name = summary["best_by_training_cv"]
    best = results.set_index("model").loc[best_name]
    coeff = pd.read_csv(REPORTS / "ridge_coefficients.csv").set_index("feature")["coefficient"]
    by_source = pd.read_csv(REPORTS / "performance_by_source.csv")
    date = pd.to_datetime(cars["collected_at"], utc=True).min().strftime("%B %d, %Y")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=10.2,
                              leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name="SmallNote", fontSize=8.5, leading=11, spaceAfter=6))
    styles["Title"].fontSize = 21
    styles["Title"].leading = 25
    styles["Title"].alignment = 0
    styles["Heading2"].fontSize = 13
    styles["Heading2"].spaceBefore = 10
    styles["Heading2"].spaceAfter = 8
    story, text_copy = [], []

    def paragraph(text, style="ReportBody"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        text_copy.append(str(text) + "\n")

    def heading(text):
        paragraph(text, "Heading2")

    def table(rows, widths=None):
        cells = [[Paragraph(escape(str(cell)), styles["SmallNote"]) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.lightgrey),
        ]))
        story.extend([item, Spacer(1, 9)])
        text_copy.extend([" | ".join(map(str, row)) for row in rows])

    def figure(name, caption, max_height=230):
        path = REPORTS / "figures" / name
        w, h = ImageReader(str(path)).getSize()
        scale = min(WIDTH / w, max_height / h)
        story.append(Image(str(path), width=w * scale, height=h * scale, hAlign="LEFT"))
        paragraph(caption, "SmallNote")

    def page():
        story.append(PageBreak())
        text_copy.append("\n---\n")

    paragraph("Predicting Used Car Value Retention", "Title")
    paragraph("Multi-source data and machine learning | " + date, "SmallNote")
    heading("1. Project description")
    paragraph("The goal was to predict how much of a car's original value it keeps and find which features are most related to depreciation. The target is the current listing price divided by original MSRP. For example, a ratio of 0.60 means the asking price is 60% of the original MSRP. This predicts an asking-price ratio, not a final sale price.")
    heading("2. Data collection")
    paragraph("Listings were scraped from CarGurus and iSeeCars using Requests and BeautifulSoup. CarGurus data came from public model, year, and color search pages. iSeeCars data came from model searches around Portland, Dallas, Chicago, Miami, Los Angeles, and Philadelphia. This is a convenience sample of search results.")
    table([["Source", "Initially parsed", "Unique valid before merge", "Final dataset"],
           *[[s, checks["raw_rows"][s], checks["unique_valid_before_integration"][s], checks["final_source_counts"][s]] for s in ["CarGurus", "iSeeCars"]]], [105, 100, 175, 130])
    paragraph(f"The final dataset has {len(cars):,} cars and no repeated VINs. There are three makes with three models each: Toyota Camry, Corolla and RAV4; Honda Accord, Civic and CR-V; and Ford Fusion, Escape and Explorer. The selected model years are 2015-2025. Ford Fusion research coverage ends at 2020.")
    paragraph("Historical MSRP was scraped from CarGurus Research tables by make, model, year and trim. Each retained car has a source URL for its MSRP. The MSRP is the trim's starting price, not a VIN-specific sticker with every optional package. Listing URLs and UTC collection times are saved in the raw files.")
    heading("Data at a glance")
    table([["Attribute", "Median", "Range"],
           ["Listing price (USD)", f"{cars.price_usd.median():,.0f}", f"{cars.price_usd.min():,.0f} - {cars.price_usd.max():,.0f}"],
           ["Mileage (miles)", f"{cars.mileage.median():,.0f}", f"{cars.mileage.min():,.0f} - {cars.mileage.max():,.0f}"],
           ["Age (years)", f"{cars.vehicle_age.median():.0f}", f"{cars.vehicle_age.min():.0f} - {cars.vehicle_age.max():.0f}"],
           ["Retention ratio", f"{cars.retention_ratio.median():.3f}", f"{cars.retention_ratio.min():.3f} - {cars.retention_ratio.max():.3f}"]], [185, 125, 200])

    page()
    heading("3. Preprocessing and integration")
    paragraph("The raw sources used different field names. For example, manufacturer became make, model_year became year, and odometer became mileage. The code removed currency punctuation, converted numeric fields, made colors and body types consistent, and split location into city and state. Prices use USD and mileage uses miles.")
    paragraph("The original attributes include VIN, listing ID, make, model, year, trim, price, mileage, body type, color, title and location. CarGurus also provides transmission, drivetrain, fuel type, engine, seller name, equipment descriptions and days on market. This exceeds the ten-original-attribute requirement. Full descriptive statistics and missing-value rates are saved as CSV files.")
    table([["Cleaning step", "Rows remaining"], *steps.values.tolist()], [395, 115])
    invalid_rows = int(steps.iloc[1]['rows'] - steps.iloc[2]['rows'])
    paragraph(f"The code keeps the chosen makes/models and years, checks VINs, and accepts prices from $1,000 to $150,000 and mileage from 0 to 500,000. These broad rules remove invalid or suspicious records. For this snapshot, the numeric/VIN checks removed {invalid_rows} extra rows after the year/model filter.")
    paragraph("Repeated VINs were removed within each source and then across sources. When both sites listed the same VIN, the richer CarGurus record was kept. MSRP matching uses trim IDs first, then normalized trim names. A shortened name is accepted only if all matching candidates have the same MSRP. Ambiguous matches are excluded rather than filled with a guessed price.")
    paragraph("Missing values stay visible in the saved data. During model training, numeric predictors use the training median and categorical predictors use 'unknown'. No test rows are used to learn these values. Ratios above 1 are kept because a used asking price can exceed the original base MSRP.")
    figure("02_model_counts.png", "Figure 1. All nine models are represented. Source contributions are uneven after cleaning and MSRP matching.", 200)

    page()
    heading("4. Feature engineering")
    paragraph("Vehicle age is collection year minus model year. Miles per year is also calculated for exploration, but the fitted models use mileage and age separately. Make and model are combined only to keep the train/test split balanced across the nine models.")
    paragraph(f"Two text features come from the short equipment descriptions attached to CarGurus listings: mentions_leather and mentions_navigation. For example, 'Leather Seats; Navigation System' gives 1 for both. There are {checks['text_feature_counts']['mentions_leather']} leather mentions and {checks['text_feature_counts']['mentions_navigation']} navigation mentions. These are listing equipment phrases, not full seller-written paragraphs.")
    paragraph(f"Descriptions are available for {checks['text_available']} of {len(cars)} cars. iSeeCars search cards do not expose this text, so their text features are missing, not automatically zero. A separate description_available flag tells the model when text was missing. The flags show what was advertised; they do not verify the equipment or vehicle history.")
    heading("Handling many categories")
    paragraph(f"Trim and city have many different values: {summary['train_unique_trims']} trims and {summary['train_unique_cities']} cities in the training set. One-hot encoding turns categories into numeric columns. Categories appearing fewer than 10 times in a training fold are grouped into an infrequent category. Unseen categories use that group where it exists. This keeps the representation smaller; the final fitted preprocessing produces {summary['encoded_columns']} columns.")
    paragraph("This method is easy to understand and does not use the target to encode a category. Its downside is that rare places or trims lose their individual detail. The encoder is fitted again inside every cross-validation fold, which prevents test information from leaking into training.")
    heading("5. Model development")
    paragraph(f"The split uses 80% training ({summary['training_rows']} cars) and 20% testing ({summary['test_rows']} cars), random_state=42, stratified by make/model. All models use the same rows and predictors. Three-fold shuffled cross-validation on the training set chooses the model with the lowest mean RMSE. Parameters were fixed in advance; there was no broad hyperparameter search.")
    table([["Model", "Main settings"],
           ["Ridge regression", "Linear regression with L2 regularization; alpha=10."],
           ["Random forest", "200 trees; min_samples_leaf=3; max_features=0.8; random_state=42."],
           ["Gradient boosting", "150 trees; learning_rate=0.05; max_depth=3; min_samples_leaf=5; random_state=42."]], [145, 365])
    paragraph("Numeric predictors are age, mileage, original MSRP and the text/availability flags. Categorical predictors are make, model, trim, body type, color, city and source. Numeric inputs are standardized after imputation. Listing price, VIN, URLs, the target and dealer price estimates are excluded. MSRP is a known vehicle specification, so using it does not reveal the used listing price.")

    page()
    heading("6. Evaluation results")
    paragraph("MAE is the average size of an error. RMSE gives bigger errors more weight. Lower values are better for both. R-squared measures how much of the variation in the test target is explained; it is not classification accuracy. All errors below are in retention-ratio units.")
    table([["Model", "CV RMSE", "Test MAE", "Test RMSE", "Test R²"],
           *[[r['model'], f"{r['cv_rmse']:.4f}", f"{r['test_mae']:.4f}", f"{r['test_rmse']:.4f}", f"{r['test_r2']:.4f}"] for _, r in results.iterrows()]], [150, 90, 90, 90, 90])
    paragraph(f"{best_name} had the lowest training CV RMSE and was selected. Its test MAE of {best['test_mae']:.4f} means an average error of about {best['test_mae'] * 100:.2f} percentage points of original value. Its test R-squared was {best['test_r2']:.3f}. A simple model that always predicts the training mean had test RMSE {summary['dummy_test_rmse']:.3f}, so the trained models improved a lot over that baseline.")
    paragraph("Ridge was very close to gradient boosting and did better than the random forest in this experiment. This suggests a simple model already captures much of the pattern. The small gap between ridge and boosting should not be treated as proof that boosting will always be better. More splits or future data would be needed to judge that.")
    figure("03_model_performance.png", f"Figure 2. Test-set comparison using the same {summary['test_rows']} cars for each model. These are regression metrics, not accuracy percentages.", 200)
    paragraph("The training CV standard deviations were small, but they describe only three folds of this particular sample. They are not confidence intervals for future-market performance.")
    table([["Selected model by source", "Test cars", "MAE", "RMSE"],
           *[[r['source'], int(r['test_rows']), f"{r['mae']:.4f}", f"{r['rmse']:.4f}"] for _, r in by_source.iterrows()]], [220, 90, 100, 100])
    paragraph("The source-level errors are different. The two websites have different samples and available fields, so this comparison does not show that one source is inherently better. Both sources appear in training and testing; this is not a test of transfer to an unseen website.")

    page()
    heading("7. Depreciation analysis and feature ranking")
    top = ", ".join(ranking.head(3)["feature"].str.replace("_", " "))
    paragraph(f"The strongest predictors were {top}. Importance was measured by shuffling one original input column in the test set and seeing how much RMSE increased. The shuffle was repeated ten times. A larger increase means the model relied more on that feature.")
    figure("04_feature_importance.png", "Figure 3. Permutation importance for the CV-selected model. Error bars show variation across ten shuffles, not confidence intervals for causal effects.", 260)
    paragraph(f"Mileage and age both had negative relationships with retention. Their sample correlations with retention were {cars['mileage'].corr(cars['retention_ratio']):.3f} and {cars['vehicle_age'].corr(cars['retention_ratio']):.3f}. The ridge coefficients were also negative: {coeff['numeric__mileage']:.3f} for mileage and {coeff['numeric__vehicle_age']:.3f} for age, per one training standard-deviation increase, with the other model inputs held fixed.")
    medians = cars.groupby('make')['retention_ratio'].median()
    paragraph(f"Make also mattered. The sample median retention ratios were about {medians['Ford']:.2f} for Ford, {medians['Honda']:.2f} for Honda and {medians['Toyota']:.2f} for Toyota. These are descriptive comparisons, not an isolated brand effect. The makes also have different age and mileage distributions. Model selection, condition, buyer demand and other unmeasured differences could explain part of the gap.")
    paragraph("Original MSRP and city added some predictive information. Leather and navigation mentions had importance close to zero. Their information may already be reflected in trim and MSRP, and missing descriptions limit what can be learned. A tiny negative importance value is sampling noise, not evidence that leather reduces a car's value.")
    paragraph("Age and mileage are correlated, and make overlaps with model. Permutation importance can split importance between related inputs or create unusual combinations when one is shuffled. These rankings describe this fitted model; they do not prove the causes of depreciation.")

    page()
    heading("8. Visualizations and what they show")
    figure("01_distributions.png", "Figure 4. Price, mileage and retention distributions for the final dataset.", 155)
    paragraph(f"The median asking price was ${cars.price_usd.median():,.0f} and median mileage was {cars.mileage.median():,.0f} miles. Mean retention was {cars.retention_ratio.mean():.1%}, which corresponds to a nominal loss of about {1-cars.retention_ratio.mean():.1%} from trim MSRP across this sample. That is not an annual depreciation rate. The few ratios above 1 may reflect options, fees, inflation, market demand or asking-price choices.")
    figure("05_depreciation.png", "Figure 5. Older and higher-mileage cars generally have lower retention, with substantial variation at the same age or mileage.", 165)
    figure("06_partial_dependence.png", "Figure 6. Average model predictions when age or mileage is changed while other inputs are held at their training values.", 165)
    paragraph("The partial-dependence plots help show the model's overall direction. They are not a forecast of how one specific car will age. Since age and mileage move together in real life, some combinations used by these plots are less realistic.")

    page()
    heading("9. Discussion and conclusion")
    paragraph(f"The project met the required sample size with {len(cars):,} unique cars, two listing sources, three makes and nine models. {best_name} was the best of the tested models by training cross-validation, but ridge was a close and simpler alternative. Mileage, make and age were the main predictors in this sample. Higher age and mileage generally went with lower retained value.")
    heading("Main lessons")
    paragraph("The project shows that collecting and matching the data takes as much care as fitting a model. The same car can appear on several pages or websites. A wrong MSRP match changes the target itself, so checking the make, model, year and trim is important. It also shows that adding text features does not automatically improve predictions, and a more complex model is not always much better than a simple one.")
    heading("Challenges and limitations")
    excluded_count = int(steps.iloc[-2]['rows'] - steps.iloc[-1]['rows'])
    paragraph(f"Several sites tested at the start blocked automated requests, so collection used accessible CarGurus and iSeeCars pages. The two sites expose different attributes. Full seller paragraphs were not available in these search pages, and description-based features use the available equipment text. Matching short trim names was another challenge; {excluded_count} otherwise valid unique cars were excluded for missing or ambiguous MSRP matches.")
    paragraph("The sample is limited to nine models, selected search pages, and one collection period. It is not representative of every car or seller. The two sources can share dealer inventory; VIN deduplication removes exact overlaps, but it does not remove all shared market biases. The test is a random split within this snapshot, not a future-year, unseen-dealer or unseen-source test.")
    paragraph("Asking prices may include different fees and are not actual sale prices. MSRP does not include every vehicle's options and is not adjusted for inflation. Accident history, maintenance, condition and local demand are not fully measured. The results describe associations, not causal depreciation rates. A useful next step would be a later collection period and verified condition/history data.")
    heading("10. AI usage statement")
    paragraph("OpenAI Codex was used to set up the repository, find accessible sources, generate and run the scraping and analysis code, validate the results, and draft this report. The student supplied the assignment requirements, chose the repository visibility, and requested simple code and wording. The generated code, data collection, analysis and report are not claimed as independently completed student work. Any independent review or changes should be described here by the student before submission.")
    heading("Sources and reproducibility")
    paragraph("CarGurus listings: https://www.cargurus.com/Cars/l-Used-Toyota-Camry-d292\niSeeCars listings: https://www.iseecars.com/used_cars-t10341-toyota-camry-portland-or\nMSRP example: https://www.cargurus.com/research/2020-Toyota-Camry-c29167", "SmallNote")
    paragraph("Every collected page is listed in data/raw/collection_log.csv, and each MSRP row includes its research URL. Encoding and importance follow the scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html and https://scikit-learn.org/stable/modules/permutation_importance.html. Run python run_project.py and python src/check_project.py from the repository to reproduce and check the saved-data analysis.", "SmallNote")

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(51, 25, "Used Car Value Retention | " + date)
        canvas.drawRightString(561, 25, str(doc.page))

    pdf = SimpleDocTemplate(str(REPORTS / "final_report.pdf"), pagesize=(612, 792),
                            leftMargin=51, rightMargin=51, topMargin=35, bottomMargin=40,
                            title="Predicting Used Car Value Retention", author="Rovena Nako")
    pdf.build(story, onFirstPage=footer, onLaterPages=footer)
    (REPORTS / "final_report.txt").write_text("\n".join(text_copy))
    print("Saved reports/final_report.pdf")


if __name__ == "__main__":
    main()
