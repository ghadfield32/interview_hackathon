import logging
from pathlib import Path
import duckdb, boto3, logging
import pandas as pd
import re
import os


def configure_logging(level: int = logging.INFO, log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(Path(log_dir, "data_pipeline.log")),
            logging.StreamHandler()
        ]
    )


def to_duck(table: str, parquet: Path) -> None:
    con = duckdb.connect(database="nba.duckdb", read_only=False)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} AS
        SELECT * FROM parquet_scan('{parquet}')
        """)
    con.execute(f"COPY (SELECT * FROM parquet_scan('{parquet}')) TO '{table}.parquet' (FORMAT PARQUET);")
    con.close()
    logging.info("%s appended to DuckDB", parquet)

def maybe_upload(parquet: Path) -> None:
    if os.getenv("AWS_ACCESS_KEY_ID"):
        s3 = boto3.client("s3")
        bucket = os.getenv("NBA_S3_BUCKET", "nba-bronze")
        s3.upload_file(str(parquet), bucket, f"bronze/{parquet.name}")
        logging.info("Uploaded %s to s3://%s/bronze/", parquet.name, bucket)

def _write_parquet_safe(df: pd.DataFrame, out_path: Path) -> None:
    """
    Atomic parquet write to avoid partial files.
    ALSO: drop any duplicate column names before writing.
    """
    # 1) Drop duplicate columns by name (keep the first occurrence)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    # 2) Write to a temporary file and atomically replace
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)


# --- NEW final writer ---
def write_final_dataset(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_safe(df, out_path)
    logging.info("Final merged parquet -> %s", out_path)
    return out_path
