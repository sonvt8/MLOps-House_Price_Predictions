"""Feature engineering (deterministic only).

Args:
  --input   path to cleaned CSV (e.g., data/processed/cleaned_data.csv)
  --output  path to write featured CSV (e.g., data/featured/featured_house_data.csv)

This script adds derived columns that don't learn parameters from data:
- price_per_sqft = price / sqft
- bed_bath_ratio = bedrooms / bathrooms
- house_age      = current_year - year_built

Run:
  python src/e_featuring/engineer.py \
    --input data/processed/cleaned_data.csv \
    --output data/featured/featured_house_data.csv

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging() -> None:
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)


def add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"price", "sqft"}.issubset(df.columns):
        df["price_per_sqft"] = df["price"] / df["sqft"].replace({0: np.nan})
    if {"bedrooms", "bathrooms"}.issubset(df.columns):
        df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace({0: np.nan})
        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    if "year_built" in df.columns:
        df["house_age"] = pd.Timestamp.now().year - df["year_built"]
    return df


def process(input_csv: Path, output_csv: Path) -> None:
    logging.info(f"Loading cleaned data: {input_csv}")
    df = pd.read_csv(input_csv)
    df = add_deterministic_features(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved featured data -> {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic feature engineering")
    p.add_argument("--input", "-i", required=True, type=str)
    p.add_argument("--output", "-o", required=True, type=str)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    process(Path(args.input).expanduser(), Path(args.output).expanduser())


if __name__ == "__main__":
    main()
