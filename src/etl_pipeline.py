"""
ETL Pipeline — Extract, Transform, Load
========================================
Owns: src/etl_pipeline.py, data/reviews.csv

Responsibilities:
  1. Download & explore the raw reviews.csv dataset
  2. Identify data quality issues (nulls, duplicates, type mismatches)
  3. Implement full data cleaning logic
  4. Implement transformation logic (sentiment scores + rolling average)
  5. Implement the LOAD step (insert cleaned data into SQLite)
  6. Add try/except error handling for file I/O operations
  7. Schema compatibility with Member 2's FastAPI (app.py / models.py)

Usage:
    python etl_pipeline.py
"""

import os
import sys
import sqlite3
import unicodedata

import pandas as pd
import numpy as np
from textblob import TextBlob


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "reviews.csv")
DB_PATH = os.path.join(BASE_DIR, "..", "data", "reviews_db.sqlite")
ROLLING_WINDOW = 3


# ══════════════════════════════════════════════
#  EXTRACT
# ══════════════════════════════════════════════
def extract(csv_path: str) -> pd.DataFrame:
    """Load raw CSV with error handling for file I/O."""
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[EXTRACT] Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path}")
        return df
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise
    except pd.errors.ParserError as e:
        print(f"ERROR: Failed to parse CSV — {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error reading CSV — {e}")
        raise


# ══════════════════════════════════════════════
#  DATA QUALITY ASSESSMENT
# ══════════════════════════════════════════════
def assess_quality(df: pd.DataFrame) -> None:
    """Identify data quality issues: nulls, duplicates, type mismatches."""
    print("\n" + "=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)

    # 1. Null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n[1] NULL VALUES — Total: {total_nulls}")
    if total_nulls > 0:
        print(null_counts[null_counts > 0])
    else:
        print("    No null values found.")

    # 2. Duplicates
    full_dupes = df.duplicated().sum()
    id_dupes = df.duplicated(subset=["review_id"]).sum()
    print(f"\n[2] DUPLICATES — Full rows: {full_dupes}, Duplicate review_ids: {id_dupes}")
    if id_dupes > 0:
        dup_ids = df[df.duplicated(subset=["review_id"], keep=False)].sort_values("review_id")
        print(dup_ids[["review_id", "product_id", "customer_id", "rating", "review_date"]])

    # 3. Type mismatches
    print(f"\n[3] TYPE MISMATCHES")
    checks = {
        "price": pd.to_numeric(df["price"], errors="coerce").isna() & df["price"].notna(),
        "rating": pd.to_numeric(df["rating"], errors="coerce").isna() & df["rating"].notna(),
        "customer_age": pd.to_numeric(df["customer_age"], errors="coerce").isna() & df["customer_age"].notna(),
        "helpful_votes": pd.to_numeric(df["helpful_votes"], errors="coerce").isna() & df["helpful_votes"].notna(),
        "review_date": pd.to_datetime(df["review_date"], errors="coerce").isna() & df["review_date"].notna(),
    }
    issues_found = False
    for col_name, mask in checks.items():
        bad = mask.sum()
        if bad > 0:
            issues_found = True
            print(f"    {col_name}: {bad} invalid values")
            print(df.loc[mask, ["review_id", col_name]])
    if not issues_found:
        print("    No type mismatches found.")

    print("=" * 60 + "\n")


# ══════════════════════════════════════════════
#  TRANSFORM — Cleaning
# ══════════════════════════════════════════════
def _normalize_text(s: str) -> str:
    """Normalize unicode characters and strip whitespace."""
    if isinstance(s, str):
        s = unicodedata.normalize("NFKC", s)
        s = s.strip()
    return s


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full data cleaning:
      1. Handle missing values
      2. Normalize text fields
      3. Parse and validate date formats
      4. Remove duplicates
    """
    print("[CLEAN] Starting data cleaning...")
    initial_rows = len(df)

    # 1. Handle missing values
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns
    df[text_cols] = df[text_cols].fillna("")
    df[num_cols] = df[num_cols].fillna(0)
    print(f"  [1] Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")

    # 2. Normalize text fields
    for col in text_cols:
        df[col] = df[col].apply(_normalize_text)

    # Lowercase email
    if "customer_email" in df.columns:
        df["customer_email"] = df["customer_email"].str.lower()

    # Title-case name/category fields
    title_cols = ["product_name", "brand", "customer_name",
                  "customer_country", "customer_city", "category"]
    for col in title_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    print("  [2] Text fields normalized (unicode, whitespace, casing).")

    # 3. Parse and validate date formats
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    invalid_dates = df["review_date"].isna().sum()
    if invalid_dates > 0:
        print(f"  [3] WARNING: {invalid_dates} rows with unparseable dates (set to NaT).")
    else:
        print(f"  [3] All review_date values parsed successfully. Dtype: {df['review_date'].dtype}")

    # 4. Remove duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    removed = initial_rows - len(df)
    print(f"  [4] Duplicates removed: {removed} rows dropped.")

    print(f"[CLEAN] Done. Shape: {df.shape}\n")
    return df


# ══════════════════════════════════════════════
#  TRANSFORM — Sentiment & Rolling Average
# ══════════════════════════════════════════════
def _get_sentiment(text: str) -> float:
    """Return polarity score in [-1.0, 1.0] using TextBlob."""
    if not text or not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sentiment scores from review_text."""
    print("[TRANSFORM] Calculating sentiment scores...")
    df["sentiment_score"] = df["review_text"].apply(_get_sentiment)
    print(f"  Sentiment stats: mean={df['sentiment_score'].mean():.4f}, "
          f"min={df['sentiment_score'].min():.4f}, max={df['sentiment_score'].max():.4f}")
    return df


def add_rolling_average(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calculate rolling average sentiment per product, sorted by date."""
    print(f"[TRANSFORM] Calculating rolling average sentiment (window={window})...")
    df = df.sort_values(["product_id", "review_date"]).reset_index(drop=True)
    df["rolling_avg_sentiment"] = (
        df.groupby("product_id")["sentiment_score"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    print(f"  Rolling avg stats: mean={df['rolling_avg_sentiment'].mean():.4f}, "
          f"min={df['rolling_avg_sentiment'].min():.4f}, max={df['rolling_avg_sentiment'].max():.4f}")
    return df


# ══════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════
def load(df: pd.DataFrame, db_path: str) -> None:
    """Insert cleaned data into SQLite database with error handling."""
    try:
        conn = sqlite3.connect(db_path)

        # Prepare data for SQLite (convert datetime to string)
        load_df = df.copy()
        load_df["review_date"] = load_df["review_date"].dt.strftime("%Y-%m-%d")

        # 1. Full cleaned reviews table
        load_df.to_sql("reviews", conn, if_exists="replace", index=False)
        print(f"[LOAD] 'reviews' table: {len(load_df)} rows written.")

        # 2. product_rolling_sentiment table (schema matches app.py / models.py)
        #    API expects: product_id, product_name, rating (as latest_sentiment_score),
        #                 rolling_average_sentiment, date (for ORDER BY)
        rolling_df = load_df[["product_id", "product_name", "rating",
                              "rolling_avg_sentiment", "review_date"]].copy()
        rolling_df = rolling_df.rename(columns={
            "rolling_avg_sentiment": "rolling_average_sentiment",
            "review_date": "date",
        })
        rolling_df.to_sql("product_rolling_sentiment", conn, if_exists="replace", index=False)
        print(f"[LOAD] 'product_rolling_sentiment' table: {len(rolling_df)} rows written.")

        # 3. Verify
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        print(f"\n  Tables in database: {tables['name'].tolist()}")
        for tbl in tables["name"]:
            cnt = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {tbl}", conn).iloc[0, 0]
            print(f"    {tbl}: {cnt} rows")

        conn.close()
        print(f"\n[LOAD] Database saved to: {os.path.abspath(db_path)}\n")

    except sqlite3.Error as e:
        print(f"ERROR: SQLite error — {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error during LOAD — {e}")
        raise


# ══════════════════════════════════════════════
#  MAIN — Run full ETL pipeline
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  ETL PIPELINE — START")
    print("=" * 60 + "\n")

    # EXTRACT
    df = extract(CSV_PATH)

    # DATA QUALITY ASSESSMENT
    assess_quality(df)

    # TRANSFORM — Clean
    df = clean(df)

    # TRANSFORM — Sentiment
    df = add_sentiment(df)

    # TRANSFORM — Rolling Average Sentiment per Product ★
    df = add_rolling_average(df, window=ROLLING_WINDOW)

    # LOAD
    load(df, DB_PATH)

    print("=" * 60)
    print("  ETL PIPELINE — COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
