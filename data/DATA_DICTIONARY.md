# Data notes

The two raw listing files preserve the different names used by the sources. The processed file uses common column names. All rows have a source page and UTC collection timestamp. Raw records can repeat when a vehicle appears on multiple search pages.

## Original listing attributes

These are collected values, not engineered features:

- `listing_id`: website listing identifier (`id` on iSeeCars).
- `vin`: vehicle identification number (`VIN` on iSeeCars).
- `make`: manufacturer (`manufacturer`).
- `model`: vehicle model (`model_name`).
- `year`: model year (`model_year`).
- `trim`: trim name (`trim_name`).
- `price_usd`: asking price (`asking_price`), in USD.
- `mileage`: odometer reading (`odometer`), standardized to miles.
- `body_type`: sedan, SUV, etc. (`body_style`).
- `color`: exterior color (`exterior_color`).
- `title`: listing title (`listing_title`).
- `city` and `state`: location; iSeeCars supplies these in its `location` field.
- `transmission`, `drivetrain`, `fuel_type`, and `engine`: CarGurus specifications; missing when not exposed.
- `seller_name`: public dealership name, when exposed. Contact details are not collected.
- `description`: CarGurus equipment description phrases joined with semicolons; unavailable on iSeeCars search cards.
- `days_on_market`: CarGurus listing age, when exposed.
- `condition`: iSeeCars condition label.

The final dataset therefore has more than the required ten original attributes. Some fields are source-specific; missing values remain missing in the stored dataset. Only predictors are imputed inside training pipelines.

## MSRP fields

- `original_msrp`: historical manufacturer's suggested retail price for the matched trim in USD, from CarGurus Research. It is a trim starting price, not a VIN-specific window sticker with all options.
- `msrp_trim`: matched research-table trim name.
- `msrp_match_method`: exact trim ID, exact normalized name, or an unambiguous shortened name. Shortening removes drivetrain tokens only when all candidates have the same MSRP.
- `msrp_source_url`: research page for that make, model, and model year.

## Engineered features and target

- `vehicle_age`: collection year minus model year; not an exact in-service age.
- `miles_per_year`: mileage divided by age (minimum denominator 1). Saved for exploration, not used in the fitted models.
- `make_model`: make and model together; used to stratify the split.
- `mentions_leather`: 1 if the description mentions leather, 0 if it does not, missing if no description is available.
- `mentions_navigation`: 1 if the description mentions navigation or GPS, with the same missing-value rule.
- `description_available`: 1 if nonempty descriptive text was collected, otherwise 0.
- `retention_ratio`: asking price divided by original MSRP. Values above 1 are kept; nominal asking price can exceed original base MSRP.

The text flags describe what is advertised, not verified equipment or vehicle history. No accident-free or one-owner claims are inferred from missing text.
