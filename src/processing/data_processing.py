"""Minimal data processing CLI.

Args:
  --input:  path to raw CSV file
  --output: directory to write 'cleaned_data.csv'

Pipeline:
  1) Load CSV
  2) Standardize column names (lowercase, underscores)
  3) Impute missing values (numeric->median, categorical->mode)
  4) Remove outliers on 'price' via IQR (1.5 whisker) if 'price' exists & numeric
  5) Drop obvious negatives on common numeric cols (if present)
  6) Drop duplicates
  7) Save to <output>/cleaned_data.csv

Run:
  python src/processing/data_processing.py \
    --input data/raw/house_data.csv \
    --output data/processed

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


# ------------------------ Core steps ------------------------ #
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else ""
            df[col] = df[col].fillna(fill_val)
    return df


def remove_price_outliers_iqr(df: pd.DataFrame, target_col: str = "price") -> pd.DataFrame:
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        logging.info("Skip IQR: 'price' missing or non-numeric.")
        return df
    s = df[target_col].astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s >= lower) & (s <= upper)
    removed = (~mask).sum()
    logging.info(f"IQR on 'price': removed {removed} rows.")
    return df.loc[mask].copy()


def drop_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["price", "sqft", "bedrooms", "bathrooms", "year_built"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df = df[df[col] >= 0]
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logging.info(f"Dropped {before - len(df)} duplicate rows.")
    return df


# ------------------------ Orchestration ------------------------ #
def process_dataset(input_csv: Path, output_dir: Path) -> Path:
    logging.info(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv)

    df = standardize_column_names(df)
    df = impute_missing_values(df)
    df = drop_negative_values(df)
    df = remove_price_outliers_iqr(df, target_col="price")
    df = drop_duplicates(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned_data.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Saved: {output_path}")
    return output_path


# ------------------------------ CLI ------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean raw dataset and write cleaned_data.csv")
    p.add_argument("--input", "-i", required=True, type=str, help="Path to raw CSV file")
    p.add_argument("--output", "-o", required=True, type=str, help="Output directory")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
    args = parse_args()
    input_csv = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    process_dataset(input_csv, output_dir)


if __name__ == "__main__":
    main()
