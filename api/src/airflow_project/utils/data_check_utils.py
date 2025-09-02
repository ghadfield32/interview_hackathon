"""
Reusable data-pull + data-quality checks (clear-arg edition)

Dimensions covered:
- Min/Max summary, Nulls, Duplicates, Outliers (IQR), Consistency (rules),
  Completeness, Accuracy (vs. reference or lookup lists), Validity (column rules),
  Uniqueness (primary key).

Style & references:
- Naming follows PEP 8 / Google Python Style Guide.
- Data-quality dimensions per common industry taxonomy.
- Outliers via 1.5×IQR; duplicates via pandas .duplicated().
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# =========================
# Report dataclasses
# =========================
@dataclass
class NullsReport:
    per_column: pd.DataFrame  # columns: column, null_count, total_rows, null_pct
    total_rows: int

@dataclass
class DuplicatesReport:
    has_duplicates: bool
    duplicate_row_count: int
    sample_rows: pd.DataFrame

@dataclass
class OutliersReport:
    per_column_summary: pd.DataFrame  # column, outlier_count, pct, lower_fence, upper_fence
    sample_rows: pd.DataFrame

@dataclass
class ConsistencyReport:
    violations_by_rule: Dict[str, int]
    sample_rows_by_rule: Dict[str, pd.DataFrame]

@dataclass
class CompletenessReport:
    required_columns: List[str]
    nonnull_ratio_by_column: Dict[str, float]
    failing_columns: List[str]

@dataclass
class AccuracyReport:
    check_status_by_name: Dict[str, str]
    sample_rows_by_check: Dict[str, pd.DataFrame]

@dataclass
class ValidityReport:
    status: str  # "passed" | "failed" | "skipped"
    details: Dict[str, Any]

@dataclass
class UniquenessReport:
    primary_key_columns: List[str]
    has_violations: bool
    duplicate_key_row_count: int
    sample_duplicate_keys: pd.DataFrame

@dataclass
class RangeReport:
    numeric_summary: pd.DataFrame   # column, count, min, max, mean, std

@dataclass
class DataQualityReport:
    minmax: RangeReport
    nulls: NullsReport
    duplicates: DuplicatesReport
    outliers: OutliersReport
    consistency: ConsistencyReport
    completeness: CompletenessReport
    accuracy: AccuracyReport
    validity: ValidityReport
    uniqueness: UniquenessReport

    def to_json(self, pretty: bool = True) -> str:
        """
        Convert the report to JSON format.

        Args:
            pretty: If True, format with indentation for readability

        Returns:
            JSON string representation of the report
        """
        def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
            return df.to_dict(orient="records") if not df.empty else []

        data = {
            "minmax": {"numeric_summary": df_to_records(self.minmax.numeric_summary)},
            "nulls": {
                "total_rows": self.nulls.total_rows,
                "per_column": df_to_records(self.nulls.per_column),
            },
            "duplicates": {
                "has_duplicates": self.duplicates.has_duplicates,
                "duplicate_row_count": self.duplicates.duplicate_row_count,
                "sample_rows": df_to_records(self.duplicates.sample_rows),
            },
            "outliers": {
                "per_column_summary": df_to_records(self.outliers.per_column_summary),
                "sample_rows": df_to_records(self.outliers.sample_rows),
            },
            "consistency": {
                "violations_by_rule": self.consistency.violations_by_rule,
                "sample_rows_by_rule": {k: df_to_records(v) for k, v in self.consistency.sample_rows_by_rule.items()},
            },
            "completeness": {
                "required_columns": self.completeness.required_columns,
                "nonnull_ratio_by_column": self.completeness.nonnull_ratio_by_column,
                "failing_columns": self.completeness.failing_columns,
            },
            "accuracy": {
                "check_status_by_name": self.accuracy.check_status_by_name,
                "sample_rows_by_check": {k: df_to_records(v) for k, v in self.accuracy.sample_rows_by_check.items()},
            },
            "validity": {"status": self.validity.status, "details": self.validity.details},
            "uniqueness": {
                "primary_key_columns": self.uniqueness.primary_key_columns,
                "has_violations": self.uniqueness.has_violations,
                "duplicate_key_row_count": self.uniqueness.duplicate_key_row_count,
                "sample_duplicate_keys": df_to_records(self.uniqueness.sample_duplicate_keys),
            },
        }
        return json.dumps(data, indent=2 if pretty else None)


def _json_or_none(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return None

def report_to_rows(report: DataQualityReport) -> list[dict]:
    """Flatten DataQualityReport into long rows suitable for a single table."""
    rows: list[dict] = []

    # ---- Min/Max
    for rec in report.minmax.numeric_summary.to_dict("records"):
        rows.append({
            "section": "minmax", "subsection": rec["column"], "field": "count", "value": rec["count"], "extra": None
        })
        for k in ("min", "max", "mean", "std"):
            rows.append({"section": "minmax", "subsection": rec["column"], "field": k, "value": rec[k], "extra": None})

    # ---- Nulls
    for rec in report.nulls.per_column.to_dict("records"):
        rows.append({
            "section": "nulls", "subsection": rec["column"], "field": "null_count",
            "value": rec["null_count"], "extra": {"total_rows": rec["total_rows"], "null_pct": rec["null_pct"]}
        })

    # ---- Duplicates
    rows.append({
        "section": "duplicates", "subsection": "_overall", "field": "has_duplicates",
        "value": bool(report.duplicates.has_duplicates),
        "extra": {"duplicate_row_count": int(report.duplicates.duplicate_row_count),
                  "sample_rows": report.duplicates.sample_rows.head(5).to_dict("records")}
    })

    # ---- Outliers (IQR fences: Q1-1.5*IQR, Q3+1.5*IQR)
    for rec in report.outliers.per_column_summary.to_dict("records"):
        rows.append({
            "section": "outliers", "subsection": rec["column"], "field": "outlier_count",
            "value": rec.get("outlier_count", 0),
            "extra": {"pct": rec.get("pct"), "lower_fence": rec.get("lower_fence"), "upper_fence": rec.get("upper_fence")}
        })
    if not report.outliers.sample_rows.empty:
        rows.append({
            "section": "outliers", "subsection": "_samples", "field": "sample_rows",
            "value": None, "extra": report.outliers.sample_rows.head(5).to_dict("records")
        })

    # ---- Consistency
    for name, cnt in (report.consistency.violations_by_rule or {}).items():
        rows.append({"section": "consistency", "subsection": name, "field": "violations", "value": int(cnt), "extra": None})
        ex = report.consistency.sample_rows_by_rule.get(name)
        if isinstance(ex, pd.DataFrame) and not ex.empty:
            rows.append({"section": "consistency", "subsection": name, "field": "sample_rows",
                         "value": None, "extra": ex.head(5).to_dict("records")})

    # ---- Completeness
    rows.append({
        "section": "completeness", "subsection": "_overall", "field": "required_columns",
        "value": None, "extra": report.completeness.required_columns
    })
    for col, ratio in (report.completeness.nonnull_ratio_by_column or {}).items():
        rows.append({"section": "completeness", "subsection": col, "field": "nonnull_ratio", "value": float(ratio), "extra": None})
    if report.completeness.failing_columns:
        rows.append({"section": "completeness", "subsection": "_overall", "field": "failing_columns",
                     "value": None, "extra": report.completeness.failing_columns})

    # ---- Accuracy
    for name, status in (report.accuracy.check_status_by_name or {}).items():
        rows.append({"section": "accuracy", "subsection": name, "field": "status", "value": status, "extra": None})
        ex = report.accuracy.sample_rows_by_check.get(name)
        if isinstance(ex, pd.DataFrame) and not ex.empty:
            rows.append({"section": "accuracy", "subsection": name, "field": "sample_rows",
                         "value": None, "extra": ex.head(5).to_dict("records")})

    # ---- Validity
    rows.append({"section": "validity", "subsection": "_overall", "field": "status", "value": report.validity.status, "extra": None})
    if report.validity.details:
        rows.append({"section": "validity", "subsection": "_overall", "field": "details",
                     "value": None, "extra": report.validity.details})

    # ---- Uniqueness
    rows.append({"section": "uniqueness", "subsection": "_overall", "field": "primary_key_columns",
                 "value": None, "extra": report.uniqueness.primary_key_columns})
    rows.append({"section": "uniqueness", "subsection": "_overall", "field": "has_violations",
                 "value": bool(report.uniqueness.has_violations), "extra": None})
    rows.append({"section": "uniqueness", "subsection": "_overall", "field": "duplicate_key_row_count",
                 "value": int(report.uniqueness.duplicate_key_row_count), "extra": None})
    if isinstance(report.uniqueness.sample_duplicate_keys, pd.DataFrame) and not report.uniqueness.sample_duplicate_keys.empty:
        rows.append({"section": "uniqueness", "subsection": "_samples", "field": "sample_duplicate_keys",
                     "value": None, "extra": report.uniqueness.sample_duplicate_keys.head(5).to_dict("records")})

    return rows

def report_to_table(report: DataQualityReport) -> pd.DataFrame:
    """Return a tidy DataFrame with columns: section, subsection, field, value, extra(json)."""
    rows = report_to_rows(report)
    tbl = pd.DataFrame.from_records(rows, columns=["section", "subsection", "field", "value", "extra"])
    # store 'extra' as a compact JSON string for easy viewing
    if "extra" in tbl.columns:
        tbl["extra"] = tbl["extra"].map(_json_or_none)
    return tbl.sort_values(["section", "subsection", "field"]).reset_index(drop=True)

# ---------- Render helpers ----------
def report_to_markdown_table(report: DataQualityReport) -> str:
    """Render the flattened table to Markdown (great for README/Slack)."""
    df = report_to_table(report)
    return df.to_markdown(index=False)

def report_to_html_table(report: DataQualityReport, caption: str | None = None) -> str:
    """Render a compact HTML table (email/dashboard)."""
    df = report_to_table(report)
    styler = df.style.set_table_attributes('class="table table-sm"').hide(axis="index")
    if caption:
        styler = styler.set_caption(caption)
    return styler.to_html()


# =========================
# Core check helpers
# =========================
def summarize_min_max(df: pd.DataFrame) -> RangeReport:
    """
    Numeric min/max/mean/std summary. Returns empty summary if no numeric columns.
    """
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        empty = pd.DataFrame(columns=["column", "count", "min", "max", "mean", "std"])
        return RangeReport(empty)
    desc = pd.DataFrame({
        "column": num.columns,
        "count": num.count().values,
        "min": num.min().values,
        "max": num.max().values,
        "mean": num.mean().values,
        "std": num.std(ddof=1).values,
    }).sort_values("column").reset_index(drop=True)
    return RangeReport(desc)


def summarize_nulls(df: pd.DataFrame) -> NullsReport:
    """
    Per-column null counts and percentages.
    """
    total = len(df)
    rows = []
    for col in df.columns:
        n = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "null_count": n,
            "total_rows": total,
            "null_pct": (n / total) if total else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("null_pct", ascending=False).reset_index(drop=True)
    return NullsReport(per_column=out, total_rows=total)


def find_duplicates(
    df: pd.DataFrame,
    *,
    key_columns: Optional[Sequence[str]] = None,
    sample_size: int = 10,
) -> DuplicatesReport:
    """
    Detect duplicate rows. If key_columns is provided, checks duplicates over that subset;
    otherwise considers the entire row. Uses pandas .duplicated().
    """
    mask = df.duplicated(subset=key_columns, keep=False)
    dupes = df.loc[mask].head(sample_size)
    count = int(mask.sum())
    return DuplicatesReport(
        has_duplicates=count > 0,
        duplicate_row_count=count,
        sample_rows=dupes
    )


def detect_outliers_iqr(
    df: pd.DataFrame,
    *,
    numeric_columns: Optional[Sequence[str]] = None,
    sample_size: int = 10,
) -> OutliersReport:
    """
    IQR-based outliers per column: values < Q1-1.5*IQR or > Q3+1.5*IQR.
    """
    num = df.select_dtypes(include=[np.number])
    if numeric_columns:
        num = num[[c for c in numeric_columns if c in num.columns]]
    summary_rows = []
    any_outlier_mask = pd.Series(False, index=df.index)
    for col in num.columns:
        s = num[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (num[col] < lower) | (num[col] > upper)
        any_outlier_mask |= mask.fillna(False)
        summary_rows.append({
            "column": col,
            "outlier_count": int(mask.sum()),
            "pct": float(mask.mean()) if mask.size else 0.0,
            "lower_fence": float(lower),
            "upper_fence": float(upper),
        })
    summary = (pd.DataFrame(summary_rows)
               if summary_rows else
               pd.DataFrame(columns=["column","outlier_count","pct","lower_fence","upper_fence"]))
    summary = summary.sort_values("outlier_count", ascending=False).reset_index(drop=True)
    examples = df.loc[any_outlier_mask].head(sample_size)
    return OutliersReport(per_column_summary=summary, sample_rows=examples)


def check_consistency_rules(
    df: pd.DataFrame,
    *,
    rules_by_name: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    sample_size: int = 5,
) -> ConsistencyReport:
    """
    Apply boolean-violation rules (fn returns mask of violations). Examples:
    - season string format; set membership; cross-field dependencies.
    """
    violations, examples = {}, {}
    for name, fn in rules_by_name.items():
        mask = fn(df)
        cnt = int(mask.sum())
        violations[name] = cnt
        if cnt:
            examples[name] = df.loc[mask].head(sample_size)
    return ConsistencyReport(violations_by_rule=violations, sample_rows_by_rule=examples)


def check_completeness(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    min_nonnull_ratio: float = 1.0,
) -> CompletenessReport:
    """
    Ensure required columns exist and meet a non-null ratio (default 100%).
    """
    ratios: Dict[str, float] = {}
    failing: List[str] = []
    total = len(df)
    for col in required_columns:
        if col not in df.columns:
            ratios[col] = 0.0
            failing.append(col)
            continue
        ratio = 1.0 if total == 0 else float(df[col].notna().mean())
        ratios[col] = ratio
        if ratio < min_nonnull_ratio:
            failing.append(col)
    return CompletenessReport(
        required_columns=list(required_columns),
        nonnull_ratio_by_column=ratios,
        failing_columns=failing,
    )


def check_accuracy(
    df: pd.DataFrame,
    *,
    allowed_values_by_column: Optional[Dict[str, Iterable[Any]]] = None,
    reference_table: Optional[pd.DataFrame] = None,
    reference_join_keys: Optional[Sequence[str]] = None,
    sample_size: int = 5,
) -> AccuracyReport:
    """
    Two simple accuracy patterns:
      (A) allowed_values_by_column: domain/lookup checks for categorical fields.
      (B) reference join coverage: ensure rows match a reference table on keys.
    """
    checks: Dict[str, str] = {}
    examples: Dict[str, pd.DataFrame] = {}

    if allowed_values_by_column:
        for col, allowed in allowed_values_by_column.items():
            if col in df.columns:
                bad_mask = ~df[col].isin(list(allowed)) & df[col].notna()
                cnt = int(bad_mask.sum())
                checks[f"lookup:{col}"] = "ok (0 violations)" if cnt == 0 else f"violations={cnt}"
                if cnt:
                    examples[f"lookup:{col}"] = df.loc[bad_mask].head(sample_size)

    if reference_table is not None and reference_join_keys:
        left = df[list(reference_join_keys)].drop_duplicates()
        right = reference_table[list(reference_join_keys)].drop_duplicates()
        merged = left.merge(right, on=list(reference_join_keys), how="left", indicator=True)
        missing = merged["_merge"] == "left_only"
        cnt = int(missing.sum())
        checks["reference_join_coverage"] = "ok (0 missing)" if cnt == 0 else f"missing={cnt}"
        if cnt:
            examples["reference_join_coverage"] = merged.loc[missing].drop(columns=["_merge"]).head(sample_size)

    return AccuracyReport(check_status_by_name=checks, sample_rows_by_check=examples)


def check_validity(
    df: pd.DataFrame,
    *,
    column_rules: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ValidityReport:
    """
    Lightweight validity (type/regex/range/membership). For strict contracts, consider Pandera/GE.
    Example column_rules:
    {
      "PLAYER_ID":   {"dtype": "int64", "ge": 0},
      "SEASON":      {"regex": r"^\\d{4}-\\d{2}$"},
      "SEASON_TYPE": {"in": ["Regular Season","Playoffs","Pre Season","All Star"]},
      "E_OFF_RATING":{"ge": 0, "le": 200}
    }
    """
    if not column_rules:
        return ValidityReport(status="skipped", details={})

    details = {"errors": []}
    for col, rules in column_rules.items():
        if col not in df.columns:
            details["errors"].append({"column": col, "error": "missing column"})
            continue
        s = df[col]
        if "dtype" in rules:
            expected = rules["dtype"]
            if str(s.dtype) != expected:
                details["errors"].append({"column": col, "error": f"dtype {s.dtype} != {expected}"})
        if "regex" in rules:
            pat = re.compile(rules["regex"])
            bad = s.dropna().astype(str).map(lambda x: not bool(pat.match(x)))
            if bad.any():
                details["errors"].append({"column": col, "error": f"regex_fail_count={int(bad.sum())}"})
        if "in" in rules:
            allowed = set(rules["in"])
            bad = ~s.isin(allowed) & s.notna()
            if bad.any():
                details["errors"].append({"column": col, "error": f"in_fail_count={int(bad.sum())}"})
        if s.dtype.kind in "fi":
            if "ge" in rules:
                bad = s.dropna() < rules["ge"]
                if bad.any():
                    details["errors"].append({"column": col, "error": f"ge_fail_count={int(bad.sum())}"})
            if "le" in rules:
                bad = s.dropna() > rules["le"]
                if bad.any():
                    details["errors"].append({"column": col, "error": f"le_fail_count={int(bad.sum())}"})
    status = "passed" if not details["errors"] else "failed"
    return ValidityReport(status=status, details=details)


def check_uniqueness(
    df: pd.DataFrame,
    *,
    primary_key_columns: Sequence[str],
    sample_size: int = 10,
) -> UniquenessReport:
    """
    Primary-key-level uniqueness across `primary_key_columns`.
    Uses pandas .duplicated(subset=...).
    """
    if not primary_key_columns:
        return UniquenessReport([], False, 0, pd.DataFrame())
    mask = df.duplicated(subset=primary_key_columns, keep=False)
    dup_count = int(mask.sum())
    examples = df.loc[mask, list(primary_key_columns)].drop_duplicates().head(sample_size)
    return UniquenessReport(
        primary_key_columns=list(primary_key_columns),
        has_violations=dup_count > 0,
        duplicate_key_row_count=dup_count,
        sample_duplicate_keys=examples,
    )


# ==========================================
# Orchestrator: run all checks & build report
# ==========================================
def run_quality_suite(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    primary_key_columns: Sequence[str],
    outlier_numeric_columns: Optional[Sequence[str]] = None,
    consistency_rules_by_name: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    allowed_values_by_column: Optional[Dict[str, Iterable[Any]]] = None,
    reference_table: Optional[pd.DataFrame] = None,
    reference_join_keys: Optional[Sequence[str]] = None,
    validity_column_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    output_format: str = "table",
    table_caption: Optional[str] = None,
) -> Union[DataQualityReport, pd.DataFrame, str]:
    """
    One-call quality suite with explicit names and flexible output formatting.

    Args:
        df: The DataFrame to analyze
        required_columns: columns that must exist and meet min non-null ratio (default 100%)
        primary_key_columns: columns forming the business key (checked for uniqueness)
        outlier_numeric_columns: numeric columns to scan for IQR outliers (None = all numeric)
        consistency_rules_by_name: {rule_name: fn(df)->mask_of_violations}
        allowed_values_by_column: {column: iterable of allowed values} for lookup/domain checks
        reference_table: DataFrame to test join coverage/accuracy against
        reference_join_keys: columns used to join df to reference_table
        validity_column_rules: {column: {dtype/regex/in/ge/le}} lightweight validity rules
        output_format: Output format - "table" (default), "json", "markdown", "html", or "report"
        table_caption: Optional caption for HTML table format

    Returns:
        - "table": pd.DataFrame with flattened results (default)
        - "json": JSON string
        - "markdown": Markdown table string  
        - "html": HTML table string
        - "report": Raw DataQualityReport object for programmatic access

    Examples:
        # Default table format
        df_results = run_quality_suite(df, required_columns=["id"], primary_key_columns=["id"])

        # JSON format for APIs
        json_results = run_quality_suite(df, required_columns=["id"], primary_key_columns=["id"], 
                                       output_format="json")

        # Markdown for documentation
        md_results = run_quality_suite(df, required_columns=["id"], primary_key_columns=["id"], 
                                     output_format="markdown")

        # Raw report object for complex logic
        report = run_quality_suite(df, required_columns=["id"], primary_key_columns=["id"], 
                                 output_format="report")
    """
    # Validate output format
    valid_formats = ["table", "json", "markdown", "html", "report"]
    if output_format not in valid_formats:
        raise ValueError(f"output_format must be one of {valid_formats}, got '{output_format}'")

    # Run all quality checks
    minmax = summarize_min_max(df)
    nulls = summarize_nulls(df)
    duplicates = find_duplicates(df, key_columns=None)
    outliers = detect_outliers_iqr(df, numeric_columns=outlier_numeric_columns)
    consistency = check_consistency_rules(df, rules_by_name=(consistency_rules_by_name or {}))
    completeness = check_completeness(df, required_columns=required_columns)
    accuracy = check_accuracy(
        df,
        allowed_values_by_column=allowed_values_by_column,
        reference_table=reference_table,
        reference_join_keys=reference_join_keys,
    )
    validity = check_validity(df, column_rules=validity_column_rules)
    uniqueness = check_uniqueness(df, primary_key_columns=primary_key_columns)

    # Create the report object
    report = DataQualityReport(
        minmax=minmax,
        nulls=nulls,
        duplicates=duplicates,
        outliers=outliers,
        consistency=consistency,
        completeness=completeness,
        accuracy=accuracy,
        validity=validity,
        uniqueness=uniqueness,
    )

    # Return in requested format
    if output_format == "report":
        return report
    elif output_format == "json":
        return report.to_json()
    elif output_format == "table":
        return report_to_table(report)
    elif output_format == "markdown":
        return report_to_markdown_table(report)
    elif output_format == "html":
        return report_to_html_table(report, caption=table_caption)
    else:
        # This shouldn't happen due to validation above, but just in case
        raise ValueError(f"Unsupported output format: {output_format}")


# ------------------------------------------
# Convenience functions for specific formats
# ------------------------------------------
def run_quality_suite_json(df: pd.DataFrame, **kwargs) -> str:
    """
    Convenience function to run quality suite and return JSON.
    All kwargs are passed to run_quality_suite except output_format.
    """
    kwargs.pop('output_format', None)  # Remove if present
    return run_quality_suite(df, output_format="json", **kwargs)


def run_quality_suite_table(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to run quality suite and return DataFrame table.
    All kwargs are passed to run_quality_suite except output_format.
    """
    kwargs.pop('output_format', None)  # Remove if present
    return run_quality_suite(df, output_format="table", **kwargs)


def run_quality_suite_markdown(df: pd.DataFrame, **kwargs) -> str:
    """
    Convenience function to run quality suite and return Markdown.
    All kwargs are passed to run_quality_suite except output_format.
    """
    kwargs.pop('output_format', None)  # Remove if present
    return run_quality_suite(df, output_format="markdown", **kwargs)


def run_quality_suite_html(df: pd.DataFrame, **kwargs) -> str:
    """
    Convenience function to run quality suite and return HTML.
    All kwargs are passed to run_quality_suite except output_format.
    """
    kwargs.pop('output_format', None)  # Remove if present
    return run_quality_suite(df, output_format="html", **kwargs)


# ------------------------------------------
# Backward-compatibility aliases (soft-deprecate)
# ------------------------------------------
def _alias_run_quality_suite(
    df: pd.DataFrame,
    *,
    required_cols: Sequence[str],
    pk: Sequence[str],
    iqr_outlier_cols: Optional[Sequence[str]] = None,
    consistency_rules: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    accuracy_lookups: Optional[Dict[str, Iterable[Any]]] = None,
    accuracy_reference_df: Optional[pd.DataFrame] = None,
    accuracy_join_key: Optional[Sequence[str]] = None,
    validity_schema: Optional[Dict[str, Dict[str, Any]]] = None,
    output_format: str = "table",
) -> Union[DataQualityReport, pd.DataFrame, str]:
    """Deprecated arg names wrapper. Prefer run_quality_suite(...)."""
    return run_quality_suite(
        df,
        required_columns=required_cols,
        primary_key_columns=pk,
        outlier_numeric_columns=iqr_outlier_cols,
        consistency_rules_by_name=consistency_rules,
        allowed_values_by_column=accuracy_lookups,
        reference_table=accuracy_reference_df,
        reference_join_keys=accuracy_join_key,
        validity_column_rules=validity_schema,
        output_format=output_format,
    )


# =========================
# Example usage
# =========================
def _example_fetch() -> pd.DataFrame:
    return pd.DataFrame({
        "PLAYER_ID": [1, 1, 2, 3, 4],
        "PLAYER_NAME": ["A", "A", "B", "C", None],
        "SEASON": ["2023-24", "2023-24", "2023-24", "2023-24", "BAD"],
        "SEASON_TYPE": ["Regular Season", "Regular Season", "Regular Season", "Playoffs", "Regular Season"],
        "E_OFF_RATING": [110.0, 110.0, 5000.0, 108.2, 105.0],
        "TEAM_ID": [10, 10, 11, None, 13],
        "TEAM_NAME": ["X", "X", "Y", "Z", "W"],
    })


def _example_rules() -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "season_format": lambda df: ~df["SEASON"].astype(str).str.match(r"^\d{4}-\d{2}$"),
        "team_id_name_coupling": lambda df: (
            (df["TEAM_ID"].isna() & df["TEAM_NAME"].notna()) |
            (df["TEAM_ID"].notna() & df["TEAM_NAME"].isna())
        ),
        "season_type_allowed": lambda df: ~df["SEASON_TYPE"].isin(
            ["Regular Season", "Playoffs", "Pre Season", "All Star"]
        ),
    }


def _example_validity_rules() -> Dict[str, Dict[str, Any]]:
    return {
        "PLAYER_ID": {"dtype": "int64", "ge": 0},
        "PLAYER_NAME": {"dtype": "object"},
        "SEASON": {"regex": r"^\d{4}-\d{2}$"},
        "SEASON_TYPE": {"in": ["Regular Season", "Playoffs", "Pre Season", "All Star"]},
        "E_OFF_RATING": {"ge": 0, "le": 200},
    }


def example_usage() -> None:
    """Demonstrates different output formats."""
    df = _example_fetch()

    # Default table format
    print("=== TABLE FORMAT (DEFAULT) ===")
    table_result = run_quality_suite(
        df,
        required_columns=["PLAYER_ID", "PLAYER_NAME", "SEASON", "SEASON_TYPE"],
        primary_key_columns=["PLAYER_ID", "SEASON", "SEASON_TYPE"],
        outlier_numeric_columns=["E_OFF_RATING"],
        consistency_rules_by_name=_example_rules(),
        allowed_values_by_column={"SEASON_TYPE": ["Regular Season","Playoffs","Pre Season","All Star"]},
        reference_table=pd.DataFrame({"TEAM_ID": [10, 11, 12, 13], "TEAM_NAME": ["X","Y","Z","W"]}),
        reference_join_keys=["TEAM_ID"],
        validity_column_rules=_example_validity_rules(),
    )
    print(table_result)
    print(f"Table shape: {table_result.shape}")

    # JSON format
    print("\n=== JSON FORMAT ===")
    json_result = run_quality_suite_json(
        df,
        required_columns=["PLAYER_ID", "PLAYER_NAME"],
        primary_key_columns=["PLAYER_ID"],
    )
    print(json_result[:500] + "...")

    # Markdown format
    print("\n=== MARKDOWN FORMAT ===")
    md_result = run_quality_suite_markdown(
        df,
        required_columns=["PLAYER_ID"],
        primary_key_columns=["PLAYER_ID"],
    )
    print(md_result[:500] + "...")

    # Raw report for programmatic access
    print("\n=== REPORT OBJECT ===")
    report = run_quality_suite(
        df,
        required_columns=["PLAYER_ID"],
        primary_key_columns=["PLAYER_ID"],
        output_format="report"
    )
    print(f"Report type: {type(report)}")
    print(f"Nulls found: {report.nulls.per_column.shape[0]} columns checked")
    print(f"Duplicates found: {report.duplicates.has_duplicates}")

    # Save outputs
    out = Path("data/quality_reports")
    out.mkdir(parents=True, exist_ok=True)

    # Save in different formats
    (out / "example_report.json").write_text(json_result)
    (out / "example_report.md").write_text(md_result)
    table_result.to_csv(out / "example_report.csv", index=False)

    html_result = run_quality_suite_html(
        df,
        required_columns=["PLAYER_ID"],
        primary_key_columns=["PLAYER_ID"],
        table_caption="Data Quality Report - Player Metrics"
    )
    (out / "example_report.html").write_text(html_result)

    print(f"\nReports saved to {out}/")


if __name__ == "__main__":
    example_usage()
