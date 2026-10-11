import numpy as np
import pandas as pd

import reportpy
from reportpy import diagnose


def test_report_has_main_sections():
    df = pd.DataFrame({"amount": range(100), "group": ["x", "y"] * 50})
    text = diagnose(df, return_text=True)
    assert "The data contains 100 observations" in text
    assert "At a glance:" in text


def test_df_method_works_after_import():
    df = pd.DataFrame({"amount": range(100), "group": ["x", "y"] * 50})
    assert df.diagnose(return_text=True) == diagnose(df, return_text=True)


def test_constant_column_is_flagged():
    df = pd.DataFrame({"amount": range(100), "same": ["a"] * 100})
    text = diagnose(df, return_text=True)
    assert "adds nothing for analysis" in text
    assert "Drop columns that carry no information: same" in text


def test_all_nines_placeholder_is_flagged():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.integers(10, 500, 98), [99999, 99999]])
    df = pd.DataFrame({"score": values})
    text = diagnose(df, return_text=True)
    assert "made of 9s" in text


def test_numbers_stored_as_text_are_detected():
    df = pd.DataFrame({"price": ["KES 1,000", "KES 2,500"] * 50})
    text = diagnose(df, return_text=True)
    assert "stored as text" in text


def test_arrival_before_departure_is_flagged():
    dep = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame({"departure_time": dep, "arrival_time": dep - pd.Timedelta(hours=2)})
    text = diagnose(df, return_text=True)
    assert "comes before" in text


def test_brief_is_shorter_than_full():
    df = pd.DataFrame({"amount": range(100), "group": ["x", "y"] * 50})
    full = diagnose(df, return_text=True)
    brief = diagnose(df, brief=True, return_text=True)
    assert len(brief) < len(full)