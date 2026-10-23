"""Collect public HTML pages. Run from the repository: python src/collect_data.py.

No login, private APIs, CAPTCHA bypass, or invented records are used.
The saved CSV files let the analysis run without scraping the websites again.
"""
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
CACHE = ROOT / ".cache"
MODELS = {
    "Toyota": {"Camry": "d292", "Corolla": "d295", "RAV4": "d306"},
    "Honda": {"Accord": "d585", "Civic": "d586", "CR-V": "d589"},
    "Ford": {"Fusion": "d845", "Escape": "d330", "Explorer": "d334"},
}
CITIES = ["portland-or", "dallas-tx", "chicago-il", "miami-fl",
          "los-angeles-ca", "philadelphia-pa"]
MIN_YEAR, MAX_YEAR = 2015, 2025
MANIFEST = []


def get_page(url):
    """Cache successful pages and pause between requests to reduce traffic."""
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    cache_file = CACHE / (key + ".html.gz")
    date_file = CACHE / (key + ".date")
    if cache_file.exists():
        html = gzip.open(cache_file, "rt").read()
        collected = date_file.read_text()
        status = 200
    else:
        time.sleep(0.8)
        response = requests.get(url, timeout=35)
        status = response.status_code
        collected = datetime.now(timezone.utc).isoformat()
        MANIFEST.append({"url": url, "collected_at": collected, "status": status})
        response.raise_for_status()
        html = response.text
        with gzip.open(cache_file, "wt") as file:
            file.write(html)
        date_file.write_text(collected)
    MANIFEST.append({"url": url, "collected_at": collected, "status": status})
    return BeautifulSoup(html, "html.parser"), collected


def cargurus_data(soup, suffix):
    """Read the JSON embedded in the public, server-rendered HTML."""
    for script in soup.find_all("script"):
        text = script.get_text()
        marker = "window.__remixContext = "
        if marker in text:
            state = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
            for name, value in state["state"]["loaderData"].items():
                if name.endswith(suffix):
                    return value
    raise ValueError("Expected page data not found; the website may have changed.")


def year_ids(value, make, model):
    """Find published model-year IDs in year links and price-trend labels."""
    result = {}
    if isinstance(value, dict):
        entity = str(value.get("entityId", ""))
        label = str(value.get("label", value.get("title", "")))
        match = re.match(r"(20\d{2})\b", label)
        correct_model = label in [f"{match[1]} {model}", f"{match[1]} {make} {model}"] if match else False
        if entity.startswith("c") and match and correct_model:
            result[int(match[1])] = entity
        for child in value.values():
            result.update(year_ids(child, make, model))
    elif isinstance(value, list):
        for child in value:
            result.update(year_ids(child, make, model))
    return result


def parse_cargurus(data, url, collected):
    rows = []
    for tile in data["search"]["tiles"]:
        car = tile.get("data", {})
        if not car.get("vin") or car.get("isNew"):
            continue
        vehicle = car["ontologyData"]
        seller = car.get("sellerData", {})
        # The equipment phrases are the listing's descriptive text, not reviews.
        description = "; ".join(car.get("vehicleFeatures", []))
        rows.append({
            "listing_id": car["id"], "vin": car["vin"],
            "make": vehicle.get("makeName"), "model": vehicle.get("modelName"),
            "year": vehicle.get("carYear"), "trim": vehicle.get("trimName"),
            "trim_id": vehicle.get("entityId"),
            "price_usd": car.get("priceData", {}).get("current"),
            "price_with_fees_usd": car.get("priceData", {}).get("totalPrice"),
            "mileage": car.get("mileageData", {}).get("value"),
            "mileage_unit": car.get("mileageData", {}).get("unit"),
            "body_type": vehicle.get("bodyTypeName"),
            "color": car.get("exteriorColorData", {}).get("normalized"),
            "transmission": car.get("localizedTransmission"),
            "drivetrain": car.get("localizedDrivetrain"),
            "fuel_type": car.get("fuelData", {}).get("localizedType"),
            "engine": car.get("localizedEngineName"),
            "city": seller.get("city"), "state": seller.get("region"),
            "seller_name": seller.get("serviceProviderName"),
            "description": description, "description_type": "equipment phrases",
            "title": car.get("listingTitle"), "days_on_market": car.get("daysOnMarket"),
            "source": "CarGurus", "source_url": url,
            "listing_url": f"https://www.cargurus.com/details/{car['id']}",
            "collected_at": collected,
        })
    return rows


def parse_iseecars(soup, url, collected):
    rows = []
    cards = soup.select("[data-listing-eid]")
    vehicles = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            item = json.loads(script.get_text())
            if item.get("@type") == "Vehicle":
                vehicles.append(item)
        except (ValueError, AttributeError):
            continue
    if len(vehicles) != len(cards):
        raise ValueError("Vehicle metadata and visible cards do not align")
    for vehicle, card in zip(vehicles, cards):
        title = card.select_one(".srp-listing-title").get_text(" ", strip=True)
        parts = card.select(".card__list-info li")
        location = parts[-1].get_text(" ", strip=True) if parts else ""
        make, model = vehicle.get("brand"), vehicle.get("model")
        prefix = f"{title[:4]} {make} {model}"
        if not title.startswith(prefix):
            raise ValueError("Vehicle metadata does not match the visible title")
        offers = vehicle.get("offers", {})
        rows.append({
            "id": vehicle.get("sku"), "VIN": vehicle.get("vehicleIdentificationNumber"),
            "manufacturer": make, "model_name": model, "model_year": title[:4],
            "trim_name": title.removeprefix(prefix).strip(),
            "asking_price": offers.get("price"), "currency": offers.get("priceCurrency"),
            "odometer": vehicle.get("mileageFromOdometer"), "distance_unit": "miles",
            "body_style": vehicle.get("bodyType"), "exterior_color": vehicle.get("color"),
            "location": location, "condition": offers.get("itemCondition"),
            "listing_title": title,
            # Search cards do not expose seller descriptions: keep this missing.
            "description": None, "source": "iSeeCars", "source_url": url,
            "listing_url": card.get("data-listing-url"), "collected_at": collected,
        })
    return rows


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    cg_rows, isee_rows, msrp_rows, failures = [], [], [], []
    model_years = {}
    for make, models in MODELS.items():
        for model, entity in models.items():
            base = f"https://www.cargurus.com/Cars/l-Used-{make}-{model}-{entity}"
            soup, date = get_page(base)
            data = cargurus_data(soup, ".Cars.$seoPath")
            cg_rows += parse_cargurus(data, base, date)
            years = year_ids(data, make, model)
            # Use published year/color links rather than undocumented pagination.
            links = [item["url"] for group in data["seoLinkGroups"]
                     if group["name"] in ["lg_cars_in_model", "lg_shop_by_color"]
                     for item in group["urls"]]
            for url in links:
                try:
                    soup, date = get_page(url)
                    page = cargurus_data(soup, ".Cars.$seoPath")
                    cg_rows += parse_cargurus(page, url, date)
                    years.update(year_ids(page, make, model))
                except (requests.RequestException, ValueError) as error:
                    failures.append({"url": url, "error": str(error)})
            model_years[(make, model)] = years
            pd.DataFrame(cg_rows).to_csv(RAW / "cargurus_listings.csv", index=False)
            for city in CITIES:
                url = f"https://www.iseecars.com/used_cars-t10341-{make.lower()}-{model.lower()}-{city}"
                try:
                    soup, date = get_page(url)
                    isee_rows += parse_iseecars(soup, url, date)
                except (requests.RequestException, ValueError) as error:
                    failures.append({"url": url, "error": str(error)})
            pd.DataFrame(isee_rows).to_csv(RAW / "iseecars_listings.csv", index=False)
            print(make, model, "parsed:", len(cg_rows), len(isee_rows), flush=True)

    # Research tables provide original MSRP, NOT a dealer's crossed-out price.
    for (make, model), years in model_years.items():
        visited = set()
        while True:
            remaining = [y for y in years if MIN_YEAR <= y <= MAX_YEAR and y not in visited]
            if not remaining:
                break
            year = max(remaining)
            visited.add(year)
            url = f"https://www.cargurus.com/research/{year}-{make}-{model}-{years[year]}"
            try:
                soup, date = get_page(url)
                data = cargurus_data(soup, ".research.$vehicleSlug")
                if (data.get("makerName"), data.get("modelName"), int(data.get("year", 0))) != (make, model, year):
                    raise ValueError("Research page identity does not match requested make/model/year")
                years.update(year_ids(data, make, model))
                for trim_id, trim in data.get("trimsDetails", {}).items():
                    if trim.get("msrp", 0) > 0:
                        msrp_rows.append({"make": make, "model": model, "year": year,
                                          "trim_id": trim_id, "trim": trim["name"],
                                          "original_msrp": trim["msrp"],
                                          "drivetrain": trim.get("driveTrain"),
                                          "msrp_source_url": url, "collected_at": date})
            except (requests.RequestException, ValueError) as error:
                failures.append({"url": url, "error": str(error)})
            pd.DataFrame(msrp_rows).to_csv(RAW / "msrp.csv", index=False)
        print("MSRP", make, model, len(msrp_rows), flush=True)
    pd.DataFrame(MANIFEST).drop_duplicates("url").to_csv(RAW / "collection_log.csv", index=False)
    pd.DataFrame(failures, columns=["url", "error"]).to_csv(RAW / "collection_errors.csv", index=False)
    print("Finished. Raw rows:", len(cg_rows), len(isee_rows), "MSRP rows:", len(msrp_rows))


if __name__ == "__main__":
    main()
