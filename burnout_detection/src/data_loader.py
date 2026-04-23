"""
Data loading utilities for the burnout detection project.

This module abstracts away the details of where the raw data comes from.  Two
modes are supported:

* **synthetic** – the original generator that creates a fake multi‑source
  student behaviour dataset (`generate_dataset.generate_student_data`).
* **oulad** – helpers to read the Open University Learning Analytics (OULAD)
  files, merge the core tables at the student level and produce a minimal
  set of features and a binary target label that can be fed into the
  machine learning pipeline.

The rest of the codebase only needs to call :func:`load_data` and will get a
pandas DataFrame back.  Keeping the loading logic separate makes it easier to
experiment with different sources / formats in the future.
"""

import os
import pandas as pd
import numpy as np


def load_data(dataset_type="synthetic", path=None, **kwargs):
    """
    Load raw data into a single DataFrame.

    Parameters
    ----------
    dataset_type : str
        One of ``'synthetic'`` or ``'oulad'``.  Defaults to ``'synthetic'`` for
        backwards compatibility with the original project.
    path : str, optional
        Location of the dataset.  When ``dataset_type=='synthetic'`` this is
        ignored.  For ``'oulad'`` the value should be either a directory
        containing the OULAD CSV files or a path to a pre‑merged CSV.  See
        :func:`load_ou_data` for details.
    **kwargs :
        Extra keyword arguments passed through to the underlying loader.  For
        example ``n_students`` / ``n_weeks`` are forwarded to the synthetic
        generator.

    Returns
    -------
    pd.DataFrame
        Raw data ready for feature engineering.  If ``dataset_type`` is
        ``'oulad'`` the returned frame will typically contain one row per
        student (the merging/aggregation is performed inside this module).
    """

    dataset_type = dataset_type.lower()
    if dataset_type == "synthetic":
        from generate_dataset import generate_student_data

        return generate_student_data(**kwargs)
    elif dataset_type == "oulad":
        if path is None:
            raise ValueError("path must be provided when dataset_type='oulad'")
        return load_ou_data(path)
    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")


def load_ou_data(path):
    """
    Load and merge the Open University Learning Analytics (OULAD) files.

    The original release consists of a handful of CSV tables; the ones
    currently used by this helper are:

    * ``studentInfo.csv`` – demographic / outcome information (contains
      ``final_result`` which is used to derive a proxy ``burnout_status``).
    * ``vle.csv`` – virtual learning environment interactions (click counts).
    * ``studentAssessment.csv`` – per-assessment scores.

    The directory given by ``path`` is scanned for any CSV files; if the
    required tables are not present an error will be raised.  If ``path`` is a
    file rather than a directory it is assumed to be a pre‑joined dataset and
    is returned unmodified.

    Parameters
    ----------
    path : str
        Directory containing OULAD CSVs or path to a single CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``student_id`` with a handful of aggregated
        features and a binary ``burnout_status`` column.  Columns are
        numeric where possible, missing values are filled with zero.
    """

    # if user passed a single file, just read and return it
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return pd.read_csv(path)

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path '{path}' is not a directory or CSV file")

    csvs = {f: os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")}
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in directory {path}")

    # load the core tables, logging helpful error messages when something is
    # missing so that a new user will know what they need to download.
    def _read(name, alt=None):
        if name in csvs:
            return pd.read_csv(csvs[name])
        elif alt and alt in csvs:
            return pd.read_csv(csvs[alt])
        else:
            return None

    student_info = _read("studentInfo.csv", "student_info.csv")
    vle = _read("vle.csv")
    stud_assess = _read("studentAssessment.csv")

    if student_info is None:
        raise FileNotFoundError("studentInfo.csv not found in OULAD directory")

    # normalise column names
    if "id_student" in student_info.columns:
        student_info = student_info.rename(columns={"id_student": "student_id"})

    # derive a simple binary target: treat withdrawals and fails as
    # "at-risk" / burnout for the purposes of this prototype.
    if "final_result" in student_info.columns:
        student_info["burnout_status"] = student_info["final_result"].isin(
            ["Withdrawn", "Fail"]
        ).astype(int)
    else:
        student_info["burnout_status"] = np.nan

    agg = student_info.copy()

    if vle is not None:
        if "id_student" in vle.columns:
            vle = vle.rename(columns={"id_student": "student_id"})
        vle_agg = (
            vle.groupby("student_id").agg(
                total_clicks=("sum_click", "sum"),
                avg_clicks=("sum_click", "mean"),
                n_sessions=("sum_click", "count"),
            )
            .reset_index()
        )
        agg = agg.merge(vle_agg, on="student_id", how="left")

    if stud_assess is not None:
        if "id_student" in stud_assess.columns:
            stud_assess = stud_assess.rename(columns={"id_student": "student_id"})
        assess_agg = (
            stud_assess.groupby("student_id").agg(
                avg_score=("score", "mean"),
                max_score=("score", "max"),
                num_assessments=("score", "count"),
            )
            .reset_index()
        )
        agg = agg.merge(assess_agg, on="student_id", how="left")

    # fill numeric missing values with zero
    numeric_cols = agg.select_dtypes(include=[np.number]).columns
    agg[numeric_cols] = agg[numeric_cols].fillna(0)

    return agg
