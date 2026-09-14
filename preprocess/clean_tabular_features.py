#!/usr/bin/env python3
"""Clean the carotid plaque tabular feature workbook.

Two datasets are written:

1. A standardized dataset retaining every row and column. Safe formatting
   errors are fixed, while genuine missing values are left for train-only
   imputation.
2. A zero-missing dataset obtained by greedily removing rows and columns.
   ``--row-priority`` controls the row/column trade-off.

A report workbook records corrections, removed rows/columns, candidate
trade-offs, missing statistics, and suspicious values. The input is untouched.
"""

from __future__ import annotations

import argparse
import math
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("feature/data_1238.xlsx")
DEFAULT_PROTECTED_COLUMNS = ["name", "斑块稳定性", "label"]
MISSING_TEXT = {
    "",
    "-",
    "--",
    "/",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "未知",
    "未测",
    "缺失",
}
BINARY_TEXT_TO_NUMBER = {
    "0": 0,
    "0.0": 0,
    "no": 0,
    "n": 0,
    "false": 0,
    "否": 0,
    "无": 0,
    "女": 0,
    "1": 1,
    "1.0": 1,
    "yes": 1,
    "y": 1,
    "true": 1,
    "是": 1,
    "有": 1,
    "男": 1,
}
BINARY_COLUMNS = {
    "label",
    "性别[1=男;0=女]",
    "卒中史[1=Yes;0=No]",
    "高血压史[1=Yes;0=No]",
    "是否规律服药[1=Yes;0=No]",
    "血压稳定[1=Yes;0=No]",
    "糖尿病史[1=Yes;0=No]",
    "是否规律服药",
    "血糖是否稳定",
    "高脂血症史[1=Yes;0=No]",
    "是否规律服药.1",
    "冠心病史[1=Yes;0=No]",
    "是否规律服药.2",
    "病情稳定",
    "是否有心功能不全",
    "吸烟[1=Yes;0=No]",
    "是否戒烟[1=Yes;0=No]",
    "饮酒[1=Yes;0=No]",
    "是否戒酒[1=Yes;0=No]",
}
CONDITIONAL_FIELDS = {
    "高血压史[1=Yes;0=No]": [
        "高血压时间（年）",
        "是否规律服药[1=Yes;0=No]",
        "血压稳定[1=Yes;0=No]",
    ],
    "糖尿病史[1=Yes;0=No]": [
        "糖尿病时间（年）",
        "是否规律服药",
        "血糖是否稳定",
    ],
    "高脂血症史[1=Yes;0=No]": [
        "高脂血症时间（年）",
        "是否规律服药.1",
    ],
    "冠心病史[1=Yes;0=No]": [
        "冠心病时间（年）",
        "是否规律服药.2",
        "病情稳定",
    ],
    "吸烟[1=Yes;0=No]": [
        "吸烟时间（年）",
        "吸烟量（支/天）",
        "是否戒烟[1=Yes;0=No]",
    ],
    "是否戒烟[1=Yes;0=No]": ["戒烟时间（年）"],
    "饮酒[1=Yes;0=No]": [
        "饮酒时间（年）",
        "饮酒量（两/天）",
        "是否戒酒[1=Yes;0=No]",
    ],
    "是否戒酒[1=Yes;0=No]": ["戒酒时间（年）"],
}
LAB_START_COLUMN = "谷丙转氨酶  U/L[5-40]"
DURATION_PATTERN = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(年|years?|yrs?|y|个月|月|months?|mos?|周|weeks?|w|天|days?|d)?$",
    flags=re.IGNORECASE,
)


@dataclass
class CompleteSubsetResult:
    dataframe: pd.DataFrame
    retained_index: list[Any]
    retained_columns: list[str]
    removed_rows: pd.DataFrame
    removed_columns: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清洗特征 Excel，并生成保留样本优先的零缺失子表和完整报告。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 Excel")
    parser.add_argument("--sheet-name", default="合并数据", help="输入 sheet 名")
    parser.add_argument(
        "--standardized-output",
        type=Path,
        default=None,
        help="格式标准化后的输出；默认在输入文件名后添加 _standardized",
    )
    parser.add_argument(
        "--complete-output",
        type=Path,
        default=None,
        help="零缺失优化子表；默认在输入文件名后添加 _complete",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="清洗报告；默认在输入文件名后添加 _cleaning_report",
    )
    parser.add_argument(
        "--row-priority",
        type=float,
        default=5.0,
        help="保留行的优先级；1偏向保留更多列，数值越大越偏向保留样本，默认5",
    )
    parser.add_argument(
        "--candidate-priorities",
        default="1,2,3,5,10,20,50",
        help="报告中比较的行优先级，逗号分隔",
    )
    parser.add_argument(
        "--min-retained-columns",
        type=int,
        default=50,
        help="零缺失子表至少保留的总列数，默认50",
    )
    parser.add_argument(
        "--protect-columns",
        nargs="+",
        default=DEFAULT_PROTECTED_COLUMNS,
        help="优化时绝不删除的列",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="优化前明确删除的列，列名含空格时需加引号",
    )
    parser.add_argument(
        "--negative-labs-to-na",
        action="store_true",
        help="把化验指标负值改为缺失；默认只报告，不自动修改",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="执行分析并打印结果，但不写文件",
    )
    return parser.parse_args()


def derived_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}.xlsx")


def value_for_report(value: Any) -> Any:
    if pd.isna(value):
        return "<NA>"
    return value


def append_change(
    changes: list[dict[str, Any]],
    df: pd.DataFrame,
    row_index: Any,
    column: str,
    original: Any,
    cleaned: Any,
    rule: str,
) -> None:
    changes.append(
        {
            "excel_row": int(df.index.get_loc(row_index)) + 2,
            "name": df.at[row_index, "name"] if "name" in df.columns else row_index,
            "column": column,
            "original": value_for_report(original),
            "cleaned": value_for_report(cleaned),
            "rule": rule,
        }
    )


def normalize_text_and_placeholders(
    df: pd.DataFrame, changes: list[dict[str, Any]]
) -> None:
    for column in df.select_dtypes(include=["object", "string"]).columns:
        for row_index, value in df[column].items():
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            normalized = stripped.lower()
            if normalized in MISSING_TEXT:
                df.at[row_index, column] = np.nan
                append_change(
                    changes,
                    df,
                    row_index,
                    column,
                    value,
                    np.nan,
                    "missing_placeholder_to_na",
                )
            elif stripped != value:
                df.at[row_index, column] = stripped
                append_change(
                    changes,
                    df,
                    row_index,
                    column,
                    value,
                    stripped,
                    "strip_whitespace",
                )


def duration_to_years(value: Any) -> tuple[float | Any, str | None]:
    if pd.isna(value) or isinstance(value, numbers.Number):
        return value, None
    if not isinstance(value, str):
        return np.nan, "invalid_duration_to_na"

    match = DURATION_PATTERN.fullmatch(value.strip())
    if match is None:
        return np.nan, "invalid_duration_to_na"

    amount = float(match.group(1))
    unit = (match.group(2) or "年").lower()
    if unit in {"年", "year", "years", "yr", "yrs", "y"}:
        years = amount
    elif unit in {"个月", "月", "month", "months", "mo", "mos"}:
        years = amount / 12.0
    elif unit in {"周", "week", "weeks", "w"}:
        years = amount / 52.1775
    elif unit in {"天", "day", "days", "d"}:
        years = amount / 365.25
    else:
        return np.nan, "invalid_duration_to_na"
    return years, "duration_to_years"


def normalize_duration_columns(
    df: pd.DataFrame, changes: list[dict[str, Any]]
) -> None:
    duration_columns = [column for column in df.columns if "时间（年）" in column]
    for column in duration_columns:
        cleaned_values: list[Any] = []
        for row_index, value in df[column].items():
            cleaned, rule = duration_to_years(value)
            cleaned_values.append(cleaned)
            if rule is not None:
                append_change(
                    changes,
                    df,
                    row_index,
                    column,
                    value,
                    cleaned,
                    rule,
                )
        df[column] = pd.to_numeric(pd.Series(cleaned_values, index=df.index), errors="coerce")


def binary_to_number(value: Any) -> tuple[int | float | Any, str | None]:
    if pd.isna(value):
        return value, None
    if isinstance(value, numbers.Number):
        numeric = float(value)
        if numeric in {0.0, 1.0}:
            return int(numeric), None
        return np.nan, "invalid_binary_to_na"
    if isinstance(value, str):
        mapped = BINARY_TEXT_TO_NUMBER.get(value.strip().lower())
        if mapped is not None:
            return mapped, "binary_text_to_number"
    return np.nan, "invalid_binary_to_na"


def normalize_binary_columns(
    df: pd.DataFrame, changes: list[dict[str, Any]]
) -> None:
    for column in [column for column in df.columns if column in BINARY_COLUMNS]:
        cleaned_values: list[Any] = []
        for row_index, value in df[column].items():
            cleaned, rule = binary_to_number(value)
            cleaned_values.append(cleaned)
            if rule is not None:
                append_change(
                    changes,
                    df,
                    row_index,
                    column,
                    value,
                    cleaned,
                    rule,
                )
        df[column] = pd.Series(cleaned_values, index=df.index, dtype="Int64")


def get_lab_columns(df: pd.DataFrame) -> list[str]:
    if LAB_START_COLUMN not in df.columns:
        return []
    start = df.columns.get_loc(LAB_START_COLUMN)
    return list(df.columns[start:])


def find_negative_labs(
    df: pd.DataFrame,
    changes: list[dict[str, Any]],
    convert_to_na: bool,
) -> pd.DataFrame:
    suspicious: list[dict[str, Any]] = []
    for column in get_lab_columns(df):
        numeric = pd.to_numeric(df[column], errors="coerce")
        for row_index in df.index[numeric.lt(0)]:
            value = df.at[row_index, column]
            suspicious.append(
                {
                    "excel_row": int(df.index.get_loc(row_index)) + 2,
                    "name": df.at[row_index, "name"]
                    if "name" in df.columns
                    else row_index,
                    "column": column,
                    "value": value,
                    "action": "converted_to_na" if convert_to_na else "reported_only",
                }
            )
            if convert_to_na:
                df.at[row_index, column] = np.nan
                append_change(
                    changes,
                    df,
                    row_index,
                    column,
                    value,
                    np.nan,
                    "negative_lab_to_na",
                )
    return pd.DataFrame(suspicious)


def label_consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    required = {"name", "斑块稳定性", "label"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    expected_label = df["斑块稳定性"].map({"稳定": 0, "不稳定": 1})
    mismatch = expected_label.notna() & df["label"].notna() & expected_label.ne(df["label"])
    result = df.loc[mismatch, ["name", "斑块稳定性", "label"]].copy()
    if not result.empty:
        result.insert(0, "excel_row", [int(df.index.get_loc(i)) + 2 for i in result.index])
        result["expected_label"] = expected_label.loc[mismatch].astype("Int64")
    return result.reset_index(drop=True)


def conditional_logic_report(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for prerequisite, related_columns in CONDITIONAL_FIELDS.items():
        if prerequisite not in df.columns:
            continue
        prerequisite_values = pd.to_numeric(df[prerequisite], errors="coerce")
        for related in related_columns:
            if related not in df.columns:
                continue
            related_values = pd.to_numeric(df[related], errors="coerce")
            base_zero = prerequisite_values.eq(0)
            base_one = prerequisite_values.eq(1)
            base_missing = prerequisite_values.isna()
            records.append(
                {
                    "prerequisite": prerequisite,
                    "related_column": related,
                    "prerequisite_0_count": int(base_zero.sum()),
                    "prerequisite_0_related_missing": int(
                        (base_zero & related_values.isna()).sum()
                    ),
                    "prerequisite_0_related_zero": int(
                        (base_zero & related_values.eq(0)).sum()
                    ),
                    "prerequisite_0_related_nonzero": int(
                        (base_zero & related_values.notna() & related_values.ne(0)).sum()
                    ),
                    "prerequisite_1_count": int(base_one.sum()),
                    "prerequisite_1_related_missing": int(
                        (base_one & related_values.isna()).sum()
                    ),
                    "prerequisite_missing_count": int(base_missing.sum()),
                }
            )
    return pd.DataFrame(records)


def greedy_complete_subset(
    df: pd.DataFrame,
    protected_columns: Sequence[str],
    row_priority: float,
    min_retained_columns: int,
) -> CompleteSubsetResult:
    """Greedily select a missing-free rectangular submatrix.

    At each step, compare the row and column that remove the most current
    missing cells. The score normalizes for current matrix size. A larger
    ``row_priority`` penalizes row deletion and therefore preserves samples.
    This is a deterministic heuristic; the report includes multiple trade-offs.
    """
    if row_priority <= 0 or not math.isfinite(row_priority):
        raise ValueError("--row-priority 必须是大于0的有限数值")
    if min_retained_columns < len(protected_columns):
        raise ValueError("--min-retained-columns 不能小于受保护列数量")
    if min_retained_columns > len(df.columns):
        raise ValueError("--min-retained-columns 不能大于当前总列数")

    retained_index = list(df.index)
    retained_columns = list(df.columns)
    protected = set(protected_columns)
    removed_rows: list[dict[str, Any]] = []
    removed_columns: list[dict[str, Any]] = []
    original_row_missing = df.isna().sum(axis=1)
    original_column_missing = df.isna().sum(axis=0)
    step = 0

    while True:
        missing = df.loc[retained_index, retained_columns].isna()
        if not missing.to_numpy().any():
            break

        row_missing = missing.sum(axis=1)
        column_missing = missing.sum(axis=0)
        row_to_remove = row_missing.idxmax()
        row_count = int(row_missing.at[row_to_remove])
        row_score = row_count * len(retained_index) / row_priority

        can_remove_column = len(retained_columns) > min_retained_columns
        droppable = [
            column
            for column in retained_columns
            if column not in protected and column_missing.at[column] > 0
        ]
        if can_remove_column and droppable:
            column_to_remove = max(
                droppable,
                key=lambda column: (
                    int(column_missing.at[column]),
                    -df.columns.get_loc(column),
                ),
            )
            column_count = int(column_missing.at[column_to_remove])
            column_score = column_count * len(retained_columns)
        else:
            column_to_remove = None
            column_count = 0
            column_score = -1.0

        step += 1
        if column_to_remove is not None and column_score > row_score:
            removed_columns.append(
                {
                    "step": step,
                    "column": column_to_remove,
                    "missing_at_removal": column_count,
                    "original_missing": int(original_column_missing.at[column_to_remove]),
                    "original_missing_rate": float(
                        original_column_missing.at[column_to_remove] / len(df)
                    ),
                }
            )
            retained_columns.remove(column_to_remove)
        else:
            missing_columns = [
                column
                for column in retained_columns
                if missing.at[row_to_remove, column]
            ]
            removed_rows.append(
                {
                    "step": step,
                    "excel_row": int(df.index.get_loc(row_to_remove)) + 2,
                    "name": df.at[row_to_remove, "name"]
                    if "name" in df.columns
                    else row_to_remove,
                    "label": df.at[row_to_remove, "label"]
                    if "label" in df.columns
                    else np.nan,
                    "missing_at_removal": row_count,
                    "original_missing": int(original_row_missing.at[row_to_remove]),
                    "missing_columns_at_removal": " | ".join(missing_columns),
                }
            )
            retained_index.remove(row_to_remove)

        if not retained_index:
            raise RuntimeError("优化删除了全部样本，请降低约束或检查受保护列")

    result = df.loc[retained_index, retained_columns].copy()
    if result.isna().to_numpy().any():
        raise AssertionError("内部错误：最终子表仍包含缺失值")
    return CompleteSubsetResult(
        dataframe=result,
        retained_index=retained_index,
        retained_columns=retained_columns,
        removed_rows=pd.DataFrame(removed_rows),
        removed_columns=pd.DataFrame(removed_columns),
    )


def parse_priorities(value: str, selected: float) -> list[float]:
    priorities = [float(item.strip()) for item in value.split(",") if item.strip()]
    priorities.append(float(selected))
    invalid = [value for value in priorities if value <= 0 or not math.isfinite(value)]
    if invalid:
        raise ValueError(f"候选优先级必须大于0，收到: {invalid}")
    return sorted(set(priorities))


def candidate_report(
    df: pd.DataFrame,
    protected_columns: Sequence[str],
    priorities: Iterable[float],
    min_retained_columns: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for priority in priorities:
        result = greedy_complete_subset(
            df,
            protected_columns=protected_columns,
            row_priority=priority,
            min_retained_columns=min_retained_columns,
        )
        label_counts = (
            result.dataframe["label"].value_counts(dropna=False).to_dict()
            if "label" in result.dataframe.columns
            else {}
        )
        records.append(
            {
                "row_priority": priority,
                "retained_rows": len(result.retained_index),
                "retained_row_rate": len(result.retained_index) / len(df),
                "retained_columns": len(result.retained_columns),
                "removed_rows": len(df) - len(result.retained_index),
                "removed_columns": len(df.columns) - len(result.retained_columns),
                "retained_area": len(result.retained_index)
                * len(result.retained_columns),
                "label_0": int(label_counts.get(0, 0)),
                "label_1": int(label_counts.get(1, 0)),
            }
        )
    return pd.DataFrame(records)


def column_report(
    raw: pd.DataFrame,
    standardized: pd.DataFrame,
    result: CompleteSubsetResult,
    explicitly_dropped: Sequence[str],
) -> pd.DataFrame:
    optimizer_dropped = set(result.removed_columns.get("column", pd.Series(dtype=str)))
    explicit = set(explicitly_dropped)
    records: list[dict[str, Any]] = []
    for column in raw.columns:
        if column in explicit:
            action = "explicitly_removed"
        elif column in optimizer_dropped:
            action = "optimizer_removed"
        elif column in result.retained_columns:
            action = "retained"
        else:
            action = "not_in_optimization_input"
        records.append(
            {
                "column": column,
                "dtype_after_standardization": str(standardized[column].dtype),
                "missing_before": int(raw[column].isna().sum()),
                "missing_rate_before": float(raw[column].isna().mean()),
                "missing_after_standardization": int(standardized[column].isna().sum()),
                "missing_rate_after_standardization": float(
                    standardized[column].isna().mean()
                ),
                "action_in_complete_subset": action,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["missing_after_standardization", "column"], ascending=[False, True]
    )


def row_report(
    raw: pd.DataFrame,
    standardized: pd.DataFrame,
    result: CompleteSubsetResult,
) -> pd.DataFrame:
    retained = set(result.retained_index)
    records: list[dict[str, Any]] = []
    for row_index in standardized.index:
        records.append(
            {
                "excel_row": int(standardized.index.get_loc(row_index)) + 2,
                "name": standardized.at[row_index, "name"]
                if "name" in standardized.columns
                else row_index,
                "label": standardized.at[row_index, "label"]
                if "label" in standardized.columns
                else np.nan,
                "missing_before": int(raw.loc[row_index].isna().sum()),
                "missing_after_standardization": int(
                    standardized.loc[row_index].isna().sum()
                ),
                "retained_in_complete_subset": row_index in retained,
            }
        )
    return pd.DataFrame(records)


def summary_report(
    raw: pd.DataFrame,
    standardized: pd.DataFrame,
    optimization_input: pd.DataFrame,
    result: CompleteSubsetResult,
    row_priority: float,
    format_change_count: int,
    negative_lab_count: int,
) -> pd.DataFrame:
    raw_missing = int(raw.isna().sum().sum())
    standardized_missing = int(standardized.isna().sum().sum())
    final = result.dataframe
    values: list[tuple[str, Any]] = [
        ("original_rows", len(raw)),
        ("original_columns", len(raw.columns)),
        ("original_missing_cells", raw_missing),
        ("original_missing_rate", raw_missing / raw.size),
        ("standardized_rows", len(standardized)),
        ("standardized_columns", len(standardized.columns)),
        ("standardized_missing_cells", standardized_missing),
        ("format_changes", format_change_count),
        ("suspicious_negative_lab_cells", negative_lab_count),
        ("optimization_input_columns", len(optimization_input.columns)),
        ("selected_row_priority", row_priority),
        ("complete_rows", len(final)),
        ("complete_row_rate", len(final) / len(raw)),
        ("complete_columns", len(final.columns)),
        ("complete_missing_cells", int(final.isna().sum().sum())),
        ("removed_rows", len(raw) - len(final)),
        ("removed_columns", len(raw.columns) - len(final.columns)),
    ]
    if "label" in raw.columns and "label" in final.columns:
        for label in [0, 1]:
            values.append(
                (f"original_label_{label}", int(raw["label"].eq(label).sum()))
            )
            values.append(
                (f"complete_label_{label}", int(final["label"].eq(label).sum()))
            )
    return pd.DataFrame(values, columns=["metric", "value"])


def write_dataframe(path: Path, df: pd.DataFrame, sheet_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.reset_index(drop=True).to_excel(writer, sheet_name=sheet_name, index=False)


def write_report(
    path: Path,
    reports: Sequence[tuple[str, pd.DataFrame]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, dataframe in reports:
            output = dataframe if not dataframe.empty else pd.DataFrame({"info": ["无记录"]})
            output.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def validate_columns(
    df: pd.DataFrame,
    protected_columns: Sequence[str],
    drop_columns: Sequence[str],
) -> None:
    missing_protected = [column for column in protected_columns if column not in df.columns]
    missing_drop = [column for column in drop_columns if column not in df.columns]
    overlap = sorted(set(protected_columns) & set(drop_columns))
    if missing_protected:
        raise ValueError(f"找不到受保护列: {missing_protected}")
    if missing_drop:
        raise ValueError(f"找不到待删除列: {missing_drop}")
    if overlap:
        raise ValueError(f"列不能同时受保护和明确删除: {overlap}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    standardized_output = (
        args.standardized_output.resolve()
        if args.standardized_output
        else derived_output_path(input_path, "_standardized")
    )
    complete_output = (
        args.complete_output.resolve()
        if args.complete_output
        else derived_output_path(input_path, "_complete")
    )
    report_output = (
        args.report_output.resolve()
        if args.report_output
        else derived_output_path(input_path, "_cleaning_report")
    )

    raw = pd.read_excel(input_path, sheet_name=args.sheet_name, engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]
    if raw.columns.duplicated().any():
        duplicates = raw.columns[raw.columns.duplicated()].tolist()
        raise ValueError(f"发现重复列名，无法安全清洗: {duplicates}")
    validate_columns(raw, args.protect_columns, args.drop_columns)

    standardized = raw.copy()
    changes: list[dict[str, Any]] = []
    normalize_text_and_placeholders(standardized, changes)
    normalize_duration_columns(standardized, changes)
    normalize_binary_columns(standardized, changes)
    negative_labs = find_negative_labs(
        standardized,
        changes=changes,
        convert_to_na=args.negative_labs_to_na,
    )

    label_conflicts = label_consistency_report(standardized)
    logic_report = conditional_logic_report(standardized)
    optimization_input = standardized.drop(columns=args.drop_columns).copy()
    if args.min_retained_columns > len(optimization_input.columns):
        raise ValueError(
            "--min-retained-columns 大于明确删列后的总列数: "
            f"{args.min_retained_columns} > {len(optimization_input.columns)}"
        )

    result = greedy_complete_subset(
        optimization_input,
        protected_columns=args.protect_columns,
        row_priority=args.row_priority,
        min_retained_columns=args.min_retained_columns,
    )
    priorities = parse_priorities(args.candidate_priorities, args.row_priority)
    candidates = candidate_report(
        optimization_input,
        protected_columns=args.protect_columns,
        priorities=priorities,
        min_retained_columns=args.min_retained_columns,
    )
    format_changes = pd.DataFrame(changes)
    columns = column_report(
        raw,
        standardized,
        result,
        explicitly_dropped=args.drop_columns,
    )
    rows = row_report(raw, standardized, result)
    summary = summary_report(
        raw,
        standardized,
        optimization_input,
        result,
        row_priority=args.row_priority,
        format_change_count=len(format_changes),
        negative_lab_count=len(negative_labs),
    )

    print(summary.to_string(index=False))
    print("\n候选行列方案:")
    print(candidates.to_string(index=False))

    if args.dry_run:
        print("\n--dry-run：未写入任何文件。")
        return

    write_dataframe(standardized_output, standardized, sheet_name=args.sheet_name)
    write_dataframe(complete_output, result.dataframe, sheet_name=args.sheet_name)
    write_report(
        report_output,
        [
            ("清洗摘要", summary),
            ("候选方案", candidates),
            ("列缺失统计", columns),
            ("行缺失统计", rows),
            ("删除列", result.removed_columns),
            ("删除行", result.removed_rows),
            ("格式修正", format_changes),
            ("疑似负值", negative_labs),
            ("条件字段逻辑", logic_report),
            ("标签冲突", label_conflicts),
        ],
    )
    print(f"\n标准化全量数据: {standardized_output}")
    print(f"零缺失优化数据: {complete_output}")
    print(f"清洗报告:       {report_output}")


if __name__ == "__main__":
    main()
