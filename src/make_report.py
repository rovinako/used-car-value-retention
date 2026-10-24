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
    paragraph("This project looks at how much value a used car keeps. The goal is to predict its value and see what affects it most. Value retention = listing price / original MSRP. A ratio of 0.60 means the asking price is 60% of the original MSRP.")
    heading("2. Data collection")
    paragraph("The listings came from CarGurus and iSeeCars. Requests and BeautifulSoup were used to scrape them. MSRP prices came from CarGurus Research. The links and dates are saved with the data.")
    table([["Source", "Raw rows", "Unique before merge", "Final rows"],
           *[[s, checks["raw_rows"][s], checks["unique_valid_before_integration"][s], checks["final_source_counts"][s]] for s in ["CarGurus", "iSeeCars"]]], [110, 95, 180, 125])
    paragraph(f"The final dataset has {len(cars):,} unique cars from 2015-2025. It includes Toyota (Camry, Corolla, RAV4), Honda (Accord, Civic, CR-V), and Ford (Fusion, Escape, Explorer). Fusion years end at 2020. Each source had more than 500 unique valid listings before merging.")
    paragraph("Collected fields include VIN, listing ID, make, model, year, trim, price, mileage, body type, color, title and location. That gives more than ten original fields.")
    heading("3. Cleaning and combining the data")
    paragraph("The column names were changed so both files could be combined. Prices are in dollars and mileage is in miles. The code checks VINs, prices ($1,000-$150,000) and mileage (0-500,000). Repeated cars are removed using VIN. When a car is on both sites, the CarGurus row is kept because it has more details.")
    table([["Step", "Rows left"], *steps.values.tolist()], [395, 115])
    paragraph("Each car is matched to MSRP using make, model, year and trim. Cars without a clear match are left out. For training, missing numbers use the training median and missing categories use 'unknown'.")

    page()
    heading("4. Feature engineering")
    paragraph("Age is calculated by subtracting the model year from the collection year. Miles per year is also saved, but the models use age and mileage separately.")
    paragraph(f"Two new columns check whether the listing mentions leather or navigation: {checks['text_feature_counts']['mentions_leather']} leather and {checks['text_feature_counts']['mentions_navigation']} navigation mentions. Text is available for {checks['text_available']} cars. These are short equipment descriptions. Missing text stays unknown. Another column shows whether a description was available.")
    paragraph(f"One-hot encoding makes a 0/1 column for each category. The training data has {summary['train_unique_cities']} cities and {summary['train_unique_trims']} trims. Categories seen fewer than ten times are grouped together. This keeps the number of columns smaller, but loses some detail.")
    heading("5. Models")
    paragraph(f"All three models use the same 80/20 split: {summary['training_rows']} training cars and {summary['test_rows']} test cars. Each make/model is included in both sets, using random_state=42. Three-fold cross-validation compares models within the training set. Filling missing values, scaling and encoding use training data only.")
    table([["Model", "Settings"],
           ["Ridge regression", "alpha=10"],
           ["Random forest", "200 trees, min_samples_leaf=3, max_features=0.8"],
           ["Gradient boosting", "150 trees, learning_rate=0.05, max_depth=3, min_samples_leaf=5"]], [150, 360])
    paragraph("Both tree models use seed 42. Inputs are age, mileage, MSRP, text columns, make, model, trim, body type, color, city and source. Numbers are scaled to a similar range. Listing price and retention are left out because they would give away the answer.")
    heading("6. Results")
    table([["Model", "CV RMSE", "Test MAE", "Test RMSE", "Test R²"],
           *[[r['model'], f"{r['cv_rmse']:.4f}", f"{r['test_mae']:.4f}", f"{r['test_rmse']:.4f}", f"{r['test_r2']:.4f}"] for _, r in results.iterrows()]], [150, 90, 90, 90, 90])
    paragraph(f"MAE is the average error. RMSE gives more weight to large errors. Lower is better for both. Higher R² is better; it is not accuracy. {best_name} had the lowest training CV error. Its test MAE of {best['test_mae']:.4f} means an average error of about {best['test_mae'] * 100:.2f} percentage points of original value. Ridge was close, so a simpler model also worked well.")
    figure("03_model_performance.png", "Figure 1. All models were tested on the same cars.", 120)

    page()
    heading("7. Data charts")
    figure("01_distributions.png", "Figure 2. Price, mileage and retention in the cleaned data.", 160)
    paragraph(f"The median price was ${cars.price_usd.median():,.0f}, median mileage was {cars.mileage.median():,.0f} miles, and median age was {cars.vehicle_age.median():.0f} years. Mean retention was {cars.retention_ratio.mean():.1%}. These are asking prices from one collection period. They do not show how much value a car loses each year. More statistics are saved in the data folder.")
    figure("02_model_counts.png", "Figure 3. All nine models are included. The two sources have different numbers of cars after cleaning.", 215)
    figure("05_depreciation.png", "Figure 4. Older cars and cars with more miles generally keep less value. Cars of the same age still vary a lot.", 165)

    page()
    heading("8. What affects car value?")
    paragraph("To check which features mattered, one column was shuffled and the model was tested again. This was done ten times. If the error went up a lot, that feature was important.")
    figure("04_feature_importance.png", "Figure 5. Mileage, make and age were the main predictors. Bars show the average change in RMSE; error lines show variation across shuffles.", 230)
    paragraph("Older cars and cars with more miles usually kept less value. Make also mattered. But the brands had different ages and mileage, so the difference cannot be explained by brand alone.")
    paragraph("MSRP and city helped a little. Leather and navigation mentions did not add much. Many descriptions were missing, and trim or MSRP may already explain these differences. The ranking shows what the model used, not proof of cause and effect.")
    figure("06_partial_dependence.png", "Figure 6. The model's average prediction when age or mileage changes. These show model patterns, not how one specific car will lose value.", 180)

    page()
    heading("9. Conclusion")
    paragraph(f"The project used {len(cars):,} unique cars from two websites and compared three models. {best_name} performed best in this experiment, with ridge close behind. Mileage, make and age were the most important factors. Older cars and cars with higher mileage usually kept less value.")
    heading("What this project shows")
    paragraph("The main lesson is that clean data matters. If MSRP is wrong, the value retention is wrong too. A complicated model is not always much better than a simple one.")
    heading("Challenges")
    excluded_count = int(steps.iloc[-2]['rows'] - steps.iloc[-1]['rows'])
    paragraph(f"Some websites blocked scraping. The two sites used different names for columns, had missing details and listed some cars more than once. Trim names were hard to match, so {excluded_count} unique cars were removed because MSRP could not be matched clearly.")
    heading("What could be better")
    paragraph("The data only covers nine models and one collection period. It does not cover the whole market. Asking prices may differ from sale prices. MSRP may miss extra options and fees, and inflation is not included.")
    paragraph("Condition, accidents and maintenance are not fully known. The test cars came from the same websites and period as the training cars. More recent listings and better car-history data would help check whether the results hold up.")
    heading("10. AI usage")
    paragraph("OpenAI Codex wrote and ran the code, checked the results and helped write this report. I provided the assignment and asked for simpler wording. I have not claimed the AI-generated code or analysis as work I completed on my own. Any work I do myself should be added here before submitting.")
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
