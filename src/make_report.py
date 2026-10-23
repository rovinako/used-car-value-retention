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
    summary = json.loads((REPORTS / "experiment_summary.json").read_text())
    best_name = summary["best_by_training_cv"]
    best = results.set_index("model").loc[best_name]
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

    paragraph("Used Car Value Retention", "Title")
    paragraph("Rovena Nako | " + date, "SmallNote")
    heading("1. Project description")
    paragraph("This project predicts how much of its original value a used car keeps. Value retention = listing price / original MSRP. For example, 0.60 means the car is listed for 60% of its original price.")
    heading("2. Data collection")
    paragraph("The data was scraped from CarGurus and iSeeCars with Requests and BeautifulSoup. Original MSRP came from CarGurus Research tables. The raw files save the page links and collection dates.")
    table([["Source", "Raw rows", "Unique before merge", "Final rows"],
           *[[s, checks["raw_rows"][s], checks["unique_valid_before_integration"][s], checks["final_source_counts"][s]] for s in ["CarGurus", "iSeeCars"]]], [110, 95, 180, 125])
    paragraph(f"The final dataset has {len(cars):,} unique cars from 2015-2025. It includes Toyota (Camry, Corolla, RAV4), Honda (Accord, Civic, CR-V), and Ford (Fusion, Escape, Explorer). Fusion years end at 2020. Each source had more than 500 unique valid listings before merging.")
    paragraph("Collected fields include VIN, listing ID, make, model, year, trim, price, mileage, body type, color, title and location. This meets the requirement of at least ten original attributes.")
    heading("3. Cleaning and combining the data")
    paragraph("Column names and formats were made consistent. Prices use dollars and mileage uses miles. The code checks VINs and keeps prices from $1,000 to $150,000 and mileage from 0 to 500,000. Duplicate VINs are removed. If both sites list the same car, the CarGurus row is kept because it has more details.")
    table([["Step", "Rows left"], *steps.values.tolist()], [395, 115])
    paragraph("MSRP is matched by make, model, year and trim. Unclear matches are removed instead of guessing. Missing numbers are filled with the training median and missing categories with 'unknown' during training. The saved dataset still shows missing values.")

    page()
    heading("4. Feature engineering")
    paragraph("Vehicle age = collection year - model year. Miles per year is calculated for exploration, but the models use age and mileage separately. The target is listing price / MSRP.")
    paragraph(f"Text features check for leather and navigation in equipment descriptions: {checks['text_feature_counts']['mentions_leather']} leather and {checks['text_feature_counts']['mentions_navigation']} navigation mentions. Text is available for {checks['text_available']} cars. These are short phrases, not full seller paragraphs. Missing text stays unknown, with a flag for description availability.")
    paragraph(f"One-hot encoding turns categories into columns. The training data has {summary['train_unique_cities']} cities and {summary['train_unique_trims']} trims. Values appearing fewer than ten times are grouped together. This saves columns but loses rare-category details.")
    heading("5. Models")
    paragraph(f"All three models use the same 80/20 split: {summary['training_rows']} training cars and {summary['test_rows']} test cars. The split is balanced by make/model and uses random_state=42. Three-fold cross-validation on training data chooses the best model. Missing-value filling, scaling and encoding are learned only from training rows.")
    table([["Model", "Settings"],
           ["Ridge regression", "alpha=10"],
           ["Random forest", "200 trees, min_samples_leaf=3, max_features=0.8"],
           ["Gradient boosting", "150 trees, learning_rate=0.05, max_depth=3, min_samples_leaf=5"]], [150, 360])
    paragraph("Both tree models use seed 42. Inputs are age, mileage, MSRP, text flags, make, model, trim, body type, color, city and source. Numbers are standardized. Price and retention are excluded to avoid giving away the answer.")
    heading("6. Results")
    table([["Model", "CV RMSE", "Test MAE", "Test RMSE", "Test R²"],
           *[[r['model'], f"{r['cv_rmse']:.4f}", f"{r['test_mae']:.4f}", f"{r['test_rmse']:.4f}", f"{r['test_r2']:.4f}"] for _, r in results.iterrows()]], [150, 90, 90, 90, 90])
    paragraph(f"Lower MAE and RMSE are better. Higher R² is better; it is not accuracy. {best_name} had the lowest training CV error. Its test MAE of {best['test_mae']:.4f} means an average error of about {best['test_mae'] * 100:.2f} percentage points of original value. Ridge was close, so boosting may not always be the best on new data.")
    figure("03_model_performance.png", "Figure 1. All models were tested on the same cars.", 120)

    page()
    heading("7. Data charts")
    figure("01_distributions.png", "Figure 2. Price, mileage and retention in the cleaned data.", 160)
    paragraph(f"The median price was ${cars.price_usd.median():,.0f}, median mileage was {cars.mileage.median():,.0f} miles, and median age was {cars.vehicle_age.median():.0f} years. Mean retention was {cars.retention_ratio.mean():.1%}. This is a snapshot of asking prices, not an annual depreciation rate. Full statistics and missing-value counts are saved in the data folder.")
    figure("02_model_counts.png", "Figure 3. All nine models are included. The two sources have different numbers of cars after cleaning.", 215)
    figure("05_depreciation.png", "Figure 4. Older cars and cars with more miles generally keep less value. Cars of the same age still vary a lot.", 165)

    page()
    heading("8. Depreciation and feature ranking")
    paragraph("Feature importance was measured by shuffling one input and checking how much worse the predictions became. This was repeated ten times. A bigger increase in error means the feature mattered more to the model.")
    figure("04_feature_importance.png", "Figure 5. Mileage, make and age were the main predictors. Bars show the average change in RMSE; error lines show variation across shuffles.", 230)
    paragraph("Higher mileage and older age generally went with lower value retention. Make also mattered, but the brands have different mixes of age, mileage and models. The results do not prove that the brand alone caused the difference.")
    paragraph("MSRP and city added some information. Leather and navigation mentions had little importance. Trim and MSRP may already capture these details, and many descriptions were missing. Related inputs can share importance, so the ranking is not exact proof of what causes depreciation.")
    figure("06_partial_dependence.png", "Figure 6. The model's average prediction when age or mileage changes. These show model patterns, not how one specific car will lose value.", 180)

    page()
    heading("9. Conclusion")
    paragraph(f"The project used {len(cars):,} unique cars from two websites and compared three models. {best_name} performed best in this experiment, with ridge close behind. Mileage, make and age were the most important factors. Older cars and cars with higher mileage usually kept less value.")
    heading("What the project taught")
    paragraph("Cleaning the data and matching MSRP are just as important as training the models. A wrong MSRP changes the answer we are trying to predict. More complicated models are not always much better, and adding text features does not always help.")
    heading("Challenges")
    excluded_count = int(steps.iloc[-2]['rows'] - steps.iloc[-1]['rows'])
    paragraph(f"Some websites blocked scraping. The two usable sites had different column names and missing details. Cars also appeared more than once. Matching trim names was difficult, and {excluded_count} unique cars were removed because MSRP could not be matched clearly.")
    heading("Limitations and next steps")
    paragraph("The data covers only nine models from selected search pages at one point in time. It does not represent the whole market. Listing prices are asking prices, not final sale prices. Original MSRP is the trim's starting price and may miss options or fees. Inflation is not included.")
    paragraph("Condition, accidents and maintenance are not fully measured. The test uses cars from the same snapshot and websites as training, so future results may be different. A next step would be collecting later listings and adding reliable condition and history information.")
    heading("10. AI usage")
    paragraph("OpenAI Codex wrote and ran the scraping and model code, checked the results, and drafted this report. The student provided the assignment and asked for simpler code and wording. No generated code or analysis is claimed as independent student work. The student should add any work they personally complete before submitting.")
    heading("Sources and running the project")
    paragraph("Listings: cargurus.com and iseecars.com. MSRP: CarGurus Research (example: https://www.cargurus.com/research/2020-Toyota-Camry-c29167). Exact page links and collection dates are in data/raw/collection_log.csv. Each MSRP row also has its source link.", "SmallNote")
    paragraph("Methods: scikit-learn documentation for OneHotEncoder and permutation_importance. Open notebooks/project_walkthrough.ipynb for the steps, or run python run_project.py and python src/check_project.py with the included data.", "SmallNote")

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
