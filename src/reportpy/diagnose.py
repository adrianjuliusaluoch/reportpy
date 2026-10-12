"""
diagnose - a text-based data diagnosis report, like R's report() but with
date coverage, distribution readings, and data-quality checks.

Usage in a notebook:
    import reportpy
    df.diagnose()                              # full report (works after the import)
    df.diagnose(brief=True)                    # header stats + warnings + takeaways only

    from reportpy import diagnose
    diagnose(df, date_cols=["Date"], dayfirst=True)
    text = diagnose(df, return_text=True)      # get the text instead of printing
"""

import re
import textwrap
import warnings
from collections import Counter

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "nil", "missing"}

NUMLIKE = re.compile(
    r"^\s*[-+]?\s*(?:kes|ksh\.?|usd|\$)?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*%?\s*(?:/\s*[a-z]+)?\s*$",
    re.IGNORECASE,
)

WIDTH = 100
BRIEF_LABELS = {"Key stats", "Range", "Distinct values", "Top values", "Watch", "In plain terms", "Takeaway"}


# ----------------------------------------------------------------- helpers
def _n(x, d=2):
    """Exact-ish number: integers get commas, floats get d decimals."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "NA"
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}"
    out = f"{x:,.{d}f}"
    return out[1:] if out.startswith("-") and float(out.replace(",", "")) == 0 else out  # no "-0.00"


def _c(x, d=2):
    """Compact number for sentences: 12.13B, 3.50M, otherwise normal."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "NA"
    x = float(x)
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return _n(x, d)


def _pct(part, whole):
    return f"{100 * part / whole:.2f}%" if whole else "0.00%"


def _s(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _wrap(text, indent=0, first=None):
    pad = " " * indent
    return textwrap.fill(
        text,
        width=WIDTH,
        initial_indent=pad if first is None else first,
        subsequent_indent=pad,
        break_on_hyphens=False,
    )


def _head(text):
    return _wrap(text, 4, "")


def _spark(x, bins=8):
    """Tiny histogram, e.g. ▁▂▅▇▃▁. Taller bar = more values in that slice of the range."""
    counts, _ = np.histogram(x, bins=bins)
    top = counts.max()
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[int(round(c / top * 7))] if top else "▁" for c in counts)


def _spark_counts(counts, width=40):
    """Sparkline of counts per period; squeezes long timelines into `width` slices."""
    vals = np.asarray(counts, dtype=float)
    if len(vals) > width:
        vals = np.array([c.sum() for c in np.array_split(vals, width)])
    top = vals.max()
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(("·" if v == 0 else blocks[int(round(v / top * 7))]) if top else "·" for v in vals)


def _emit(items, brief):
    """items: list of (label, text). Brief mode keeps only Watch and Takeaway."""
    out = []
    for label, text in items:
        if brief and label not in BRIEF_LABELS:
            continue
        out.append(_wrap(f"{label}: {text}", 8, " " * 6))
    return out


def _parse_unique(vals, dayfirst):
    """Parse an array of date strings. Guess one format first (fast, vectorized); else mixed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = next((v for v in vals if isinstance(v, str)), None)
        if first is not None:
            try:
                from pandas.tseries.api import guess_datetime_format

                fmt = guess_datetime_format(first, dayfirst=dayfirst)
            except Exception:
                fmt = None
            if fmt:
                try:
                    out = pd.to_datetime(vals, format=fmt, errors="coerce")
                    if pd.notna(out).mean() >= 0.9:
                        return out
                except (TypeError, ValueError):
                    pass
        try:
            return pd.to_datetime(vals, errors="coerce", dayfirst=dayfirst, format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(vals, errors="coerce", dayfirst=dayfirst)


def _parse_dates(s, dayfirst=False):
    """Parse a Series into datetimes without raising. Works on unique values, then maps back."""
    nn = s.dropna()
    if nn.empty:
        return pd.to_datetime(s, errors="coerce")
    uniq = nn.unique()
    parsed = pd.Series(pd.DatetimeIndex(_parse_unique(uniq, dayfirst)), index=pd.Index(uniq))
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    out = s.map(parsed)
    if not pd.api.types.is_datetime64_any_dtype(out):
        out = pd.to_datetime(out, errors="coerce")
    return out


def _to_numeric_text(s):
    """Returns (numbers, placeholder_count, unit_counts). Works on unique values for speed."""
    vc = s.value_counts()
    lab = pd.Series(vc.index.astype(str)).str.strip()
    ph = lab.str.lower().isin(MISSING_TOKENS).values
    lab = lab.mask(ph)
    unit_u = lab.str.extract(r"(?i)/\s*([a-z]+)\s*$")[0].str.lower()
    unit_counts = pd.Series(vc.values).groupby(unit_u.values).sum()
    cleaned = lab.str.replace(r"(?i)/\s*[a-z]+\s*$", "", regex=True)
    cleaned = cleaned.str.replace(r"(?i)kes|ksh\.?|usd|\$|,|%|\s", "", regex=True)
    nums = pd.to_numeric(cleaned, errors="coerce").values
    mapping = pd.Series(nums, index=vc.index)
    conv = s.map(mapping).astype(float)
    return conv, int(vc.values[ph].sum()), {k: int(v) for k, v in unit_counts.items()}


def _classify(s, name, force_dates, dayfirst):
    if s.isna().all():
        return "empty", s, {}
    if pd.api.types.is_bool_dtype(s):
        return "bool", s, {}
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
        return "date", s, {}
    if name in force_dates:
        parsed = _parse_dates(s, dayfirst)
        return "date", parsed, {"from_text": True, "unparsed": int(parsed.isna().sum() - s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s):
        return "numeric", s, {}

    nn = s.dropna()
    nn = nn.sample(300, random_state=0) if len(nn) > 300 else nn
    sample = nn.astype(str).str.strip()

    real = sample[~sample.str.lower().isin(MISSING_TOKENS)]
    if len(real) >= 1 and real.str.match(NUMLIKE).mean() >= 0.9:
        conv, nph, units = _to_numeric_text(s)
        cur = real.str.extract(r"(?i)(kes|ksh\.?|usd|\$)")[0].dropna().str.upper()
        return "numeric", conv, {
            "from_text": True,
            "example": str(real.iloc[0]),
            "currency": sorted(set(cur)),
            "percent": bool(real.str.contains("%").any()),
            "commas": bool(real.str.contains(",").any()),
            "unparsed": max(int(conv.isna().sum() - s.isna().sum() - nph), 0),
            "placeholders": nph,
            "units": units,
        }

    if sample.str.len().median() >= 6:
        if _parse_dates(sample, dayfirst).notna().mean() >= 0.9:
            parsed = _parse_dates(s, dayfirst)
            return "date", parsed, {"from_text": True, "unparsed": int(parsed.isna().sum() - s.isna().sum())}

    return "categorical", s, {}


# ----------------------------------------------------------------- numeric
def _sentinel(x):
    """Find a max/min made of 9s (e.g. 9999, 99999999.99, -999) that sits far from the rest."""
    found = []
    u = np.sort(x.unique())
    if len(u) < 3:
        return found
    for v, other in ((u[-1], u[-2]), (u[0], u[1])):
        digits = re.sub(r"\D", "", f"{abs(v):.2f}").rstrip("0")
        if len(digits) >= 3 and set(digits) == {"9"} and abs(v) >= 5 * max(abs(other), 1):
            found.append(v)
    return found


def _numeric_block(s, name, flags, info, brief):
    n = len(s)
    s = s.astype(float)
    n_inf = int(np.isinf(s).sum())
    x = s[np.isfinite(s)]
    k = len(x)
    miss = n - k - n_inf

    tag = "numeric, stored as text" if info.get("from_text") else "numeric"
    if k == 0:
        return [_head(f"- {name} [{tag}]: no usable numbers")]

    mean, med, sd = x.mean(), x.median(), x.std()
    mad = 1.4826 * (x - med).abs().median()  # scaled, same as R's mad()
    skew = x.skew() if k >= 3 else np.nan
    kurt = x.kurt() if k >= 4 else np.nan  # excess kurtosis, same as report()
    mn, mx = x.min(), x.max()

    lines = [_head(f"- {name} [{tag}] {_spark(x)}: {k:,} valid values, {_pct(miss, n)} missing")]
    items = [
        ("Key stats", (
            f"Mean = {_n(mean)}, SD = {_n(sd)}, Median = {_n(med)}, MAD = {_n(mad)}, "
            f"Skewness = {_n(skew)}, Kurtosis = {_n(kurt)}."
        )),
        ("Range", f"{_n(mn)} to {_n(mx)}."),
    ]

    if info.get("from_text"):
        removed = []
        for u in (info.get("units") or {}):
            removed.append(f"the unit '/{u}'")
        for c in info.get("currency") or []:
            removed.append(f"the currency label '{c}'")
        if info.get("percent"):
            removed.append("'%' signs")
        if info.get("commas"):
            removed.append("thousands commas")
        if info.get("placeholders"):
            removed.append("placeholders like '-' (set to missing)")
        if not removed:
            what = "extra text around the numbers"
        elif len(removed) == 1:
            what = removed[0]
        else:
            what = ", ".join(removed[:-1]) + " and " + removed[-1]
        items.append((
            "Watch",
            f"Stored as text in the original data (e.g. '{info.get('example', '')}'). These numbers were "
            f"extracted by removing {what}, so the stats describe the extracted values, not the raw column. "
            "Convert it properly in your cleaning step before analyzing.",
        ))
        flags.append((name, f"stored as text (e.g. '{info.get('example', '')}'); converted here only for these stats"))
    if info.get("unparsed", 0) > 0:
        items.append(("Watch", f"{_s(info['unparsed'], 'text value')} could not be converted to numbers."))
        flags.append((name, f"{info['unparsed']} values could not be converted to numbers"))
    if n_inf:
        items.append(("Watch", f"{_s(n_inf, 'infinite value')} left out of the stats above."))
        flags.append((name, f"{n_inf} infinite values"))
    if info.get("placeholders"):
        ph = info["placeholders"]
        items.append(("Watch", f"{_s(ph, 'value')} ({ph / n * 100:.1f}% of rows) are placeholders like '-', counted as missing above."))
        flags.append((name, f"{ph / n * 100:.1f}% hidden missing (placeholder text)"))
    units = info.get("units") or {}
    if units:
        if not (len(units) == 1 and sum(units.values()) >= k):
            parts = ", ".join(f"/{u} ({c:,})" for u, c in units.items())
            items.append(("Watch", f"Mixed or missing units ({parts}); the stats above treat them all as the same unit."))
            flags.append((name, "mixed or missing units"))

    nun = x.nunique()

    if nun == 1:
        items.append(("Watch", f"Every value is {_n(mn)}, so this column carries no information for analysis."))
        flags.append((name, "constant column"))
        return lines + _emit(items, brief)

    int_like = bool((x % 1 == 0).all())
    id_hint = bool(re.search(r"(^|[_\s])(id|code|no|number|index|key)($|[_\s])", name.lower()))
    if int_like and nun == k and k > 20 and (id_hint or x.is_monotonic_increasing or x.is_monotonic_decreasing):
        items.append((
            "Watch",
            "Every value is a unique whole number, which looks like an ID or counter rather than a "
            "measurement, so the averages above are not meaningful.",
        ))
        flags.append((name, "looks like an ID, not a measurement"))
        return lines + _emit(items, brief)

    if nun <= 2:
        vc = x.value_counts()
        parts = "; ".join(f"{_n(v)} ({_pct(c, n)})" for v, c in vc.sort_index().items())
        if set(vc.index) <= {0, 1}:
            items.append(("Watch", f"Only 0 and 1 appear ({parts}), so treat it as a yes/no flag, not a measurement."))
        else:
            items.append((
                "Watch",
                f"Only 2 distinct values ({parts}). That is very little variation for a measurement; "
                "check whether it was rounded, capped, or generated, or whether it should be a category.",
            ))
            flags.append((name, f"only 2 distinct values ({', '.join(_n(v) for v in sorted(vc.index))})"))
        return lines + _emit(items, brief)

    sent = _sentinel(x)
    if sent:
        rest = x[~x.isin(sent)]
        for v in sent:
            cnt = int((x == v).sum())
            items.append((
                "Watch",
                f"{_n(v)} appears {_s(cnt, 'time')} and is made of 9s, a common placeholder code for "
                f"'unknown' rather than a real value. Without it: mean {_c(rest.mean())}, "
                f"median {_c(rest.median())}, max {_c(rest.max())}.",
            ))
            flags.append((name, f"{_n(v)} looks like a placeholder code ({cnt:,} rows)"))

    if nun <= 10 and k >= 50:
        items.append((
            "Type",
            f"Only {nun} distinct values, so this is probably a count, a rating scale, or a measurement "
            "recorded in fixed steps; counts per value will say more than the mean.",
        ))
    if mn >= 0 and mx <= 1:
        items.append(("Scale", "All values lie between 0 and 1, so this is probably a ratio, share, or probability."))

    a = abs(skew) if k >= 3 else 0.0
    if k >= 3 and sd > 0:
        side = "right" if skew > 0 else "left"
        if a < 0.5:
            shape = "roughly symmetric"
        elif a < 1:
            shape = f"moderately skewed to the {side}"
        else:
            shape = f"strongly skewed to the {side}"
        if a >= 0.5:
            shape += f", with a tail of unusually {'large' if skew > 0 else 'small'} values"
        items.append(("Shape", shape + "."))

        # center: compare the gap against SD and against the robust spread, because an
        # inflated SD can hide a big gap (mean 9,415 vs median 2,000 with SD 560,000)
        rel_sd = (mean - med) / sd
        rel_mad = (mean - med) / mad if mad > 0 else 0.0
        g = max(abs(rel_sd), abs(rel_mad))
        above = mean > med
        if g < 0.1:
            c = f"Mean ({_c(mean)}) and median ({_c(med)}) are close together."
            if a >= 1:
                c += (
                    " Even so, skewness is high, which usually means extreme values on one or both ends "
                    "while the bulk of the data sits tightly around the middle."
                )
            else:
                c += " The center is well defined."
        elif g < 0.3:
            c = (
                f"Mean ({_c(mean)}) is slightly {'above' if above else 'below'} the median ({_c(med)}); "
                "not a big gap."
            )
        else:
            if med > 0 and mean > 0 and above and mean / med >= 1.5:
                how = f"{mean / med:.1f}x the size of"
            elif med > 0 and mean > 0 and not above and med / mean >= 1.5:
                how = f"only {mean / med:.2f}x the size of"
            else:
                how = "above" if above else "below"
            c = (
                f"Mean ({_c(mean)}) is {how} the median ({_c(med)}), so a few "
                f"{'large' if above else 'small'} values are dragging it; the median is the better 'typical value'."
            )
        items.append(("Center", c))

        # tails
        if not np.isnan(kurt):
            if kurt > 3:
                t = f"Kurtosis {_n(kurt)}: very heavy tails, extreme values are far more common than in a bell curve."
            elif kurt > 1:
                t = f"Kurtosis {_n(kurt)}: somewhat heavy tails, expect a few extreme values."
            elif kurt < -1:
                t = (
                    f"Kurtosis {_n(kurt)}: light tails and a flat spread with very few extremes "
                    "(close to evenly spread, like a uniform distribution)."
                )
            else:
                t = "Tail weight is close to a bell curve."
            items.append(("Tails", t))

        # spread
        if mad > 0:
            ratio = sd / mad
            if ratio >= 2:
                sp = (
                    f"SD ({_c(sd)}) is {ratio:,.1f}x the robust spread MAD ({_c(mad)}), so a handful of "
                    f"extreme values are inflating the SD. Most values sit within about {_c(mad)} of the median."
                )
            elif ratio <= 1.25:
                sp = "SD and MAD agree, so spread is consistent across the whole range."
            else:
                sp = ""
        else:
            sp = "MAD is 0: at least half of the values equal the median exactly, so one repeated value dominates."
        if mn > 0 and mean > 0:
            cv = sd / mean
            if cv >= 2:
                sp += f" SD is {cv:.1f}x the mean, so values vary enormously relative to their size."
            elif cv >= 1:
                sp += f" SD is {cv * 100:.0f}% of the mean, so values vary a lot relative to their size."
            elif cv < 0.15:
                sp += f" SD is only {cv * 100:.0f}% of the mean, so values are tightly clustered."
        if sp.strip():
            items.append(("Spread", sp.strip()))

        # clusters (skip for scale-like columns; the coefficient is unreliable there)
        if k > 30 and nun > 10 and not np.isnan(kurt) and a < 1:
            bc = (skew**2 + 1) / (kurt + 3 * (k - 1) ** 2 / ((k - 2) * (k - 3)))
            if bc > 0.555:
                items.append((
                    "Clusters",
                    f"Bimodality coefficient {bc:.2f} (above 0.555): there may be two groups mixed together; "
                    "check a histogram.",
                ))

        if a >= 2:
            flags.append((name, f"strong skew ({_n(skew)})"))

    # typical range and outliers
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    p1, p99 = x.quantile(0.01), x.quantile(0.99)
    if nun <= 10:
        vcn = x.value_counts().sort_index()
        items.append(("Values", "; ".join(f"{_n(v)} ({_pct(c, k)})" for v, c in vcn.items()) + "."))
    else:
        items.append((
            "Typical range",
            f"Middle 50% of values: {_c(q1)} to {_c(q3)}. 1st to 99th percentile: {_c(p1)} to {_c(p99)}.",
        ))
    # plain-language summary, placed first so a reader gets the picture before the stats
    if k >= 3 and sd > 0 and nun > 2:
        if nun <= 10:
            top_v = x.value_counts().index[0]
            plain = f"Most common value is {_n(top_v)} ({_pct(int((x == top_v).sum()), k)} of rows); values run from {_c(mn)} to {_c(mx)}."
        else:
            real_x = x[~x.isin(sent)] if sent else x
            plain = (
                f"A typical value is about {_c(med)}. Half of the rows fall between {_c(q1)} and {_c(q3)}; "
                f"the full range runs from {_c(real_x.min())} to {_c(real_x.max())}"
                + (" (ignoring the placeholder code)" if sent else "")
            )
            if a >= 1 and skew > 0:
                plain += f", but only 1% of rows go above {_c(p99)}, so a few very large values stretch the top end."
            elif a >= 1 and skew < 0:
                plain += f", but only 1% of rows fall below {_c(p1)}, so a few very small values stretch the bottom end."
            else:
                plain += "."
        pos = 2  # after Key stats and Range
        while pos < len(items) and items[pos][0] == "Watch":
            pos += 1
        items.insert(pos, ("In plain terms", plain))
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_lo, n_hi = int((x < lo).sum()), int((x > hi).sum())
        if n_lo + n_hi > 0:
            share = (n_lo + n_hi) / k
            parts = []
            if n_hi:
                parts.append(f"{_s(n_hi, 'value')} ({n_hi / k * 100:.1f}%) above {_c(hi)}")
            if n_lo:
                parts.append(f"{_s(n_lo, 'value')} ({n_lo / k * 100:.1f}%) below {_c(lo)}")
            text = "By the 1.5 x IQR rule, " + " and ".join(parts)
            if not n_lo:
                text += "; none are unusually low"
            elif not n_hi:
                text += "; none are unusually high"
            text += "."
            if share >= 0.02 and ((n_hi and skew >= 1) or (n_lo and skew <= -1)):
                text += " The data is skewed, so many of these are probably the natural long tail, not errors."
            items.append(("Outliers", text))
            if share > 0.05:
                flags.append((name, f"{share * 100:.1f}% of values are outliers by the IQR rule"))

    # zeros and signs
    neg, zero = int((x < 0).sum()), int((x == 0).sum())
    if zero / k >= 0.5:
        items.append((
            "Watch",
            f"{zero / k * 100:.1f}% of values are exactly zero, so most rows have none. Consider "
            "analyzing 'has any' (yes/no) and 'how many' (among non-zero rows) separately.",
        ))
        flags.append((name, f"{zero / k * 100:.0f}% zeros (zero-inflated)"))
    elif zero / k >= 0.1:
        items.append(("Zeros", f"{zero / k * 100:.1f}% of values are exactly zero."))
    if mn < 0 and mx > 0:
        items.append(("Signs", f"Mix of positive and negative values ({neg / k * 100:.1f}% negative); check what the sign means."))
    elif mn < 0:
        items.append(("Signs", "All values are zero or negative."))

    # takeaway
    if k >= 3 and sd > 0:
        if a < 0.5 and (np.isnan(kurt) or abs(kurt) < 1):
            take = "close to bell-shaped; mean and SD are safe summaries."
        elif skew >= 1 and mn > 0:
            take = "right-skewed with a long tail of large values; lean on median and percentiles, and consider a log transform."
        elif skew >= 1:
            take = "right-skewed; open the raw rows behind the extremes before deciding to trim, cap, or keep them."
        elif skew <= -1:
            take = "left-skewed; most values are high with a tail of low ones; lean on the median."
        elif not np.isnan(kurt) and kurt > 3:
            take = "centered, but extreme values at one or both ends; check those rows before trusting the mean and SD."
        else:
            take = "mildly off a bell shape; mean and median together tell the story."
        items.append(("Takeaway", take))

    return lines + _emit(items, brief)


# ------------------------------------------------------------------- dates
def _span_text(days):
    if days < 60:
        return f"{days} days"
    if days < 730:
        return f"{days} days (about {days / 30.44:.1f} months)"
    return f"{days} days (about {days / 365.25:.1f} years)"


def _date_block(s, name, flags, info, brief):
    n = len(s)
    x = s.dropna()
    k = len(x)
    tag = "date, stored as text" if info.get("from_text") else "date"
    unparsed = max(info.get("unparsed", 0), 0)

    head = f"- {name} [{tag}]: {k:,} valid dates, {_pct(n - k - unparsed, n)} missing"
    items = []
    if unparsed:
        head += f", {unparsed} not readable as dates"
        items.append(("Watch", f"{_s(unparsed, 'value')} could not be read as dates."))
        flags.append((name, f"{unparsed} values could not be read as dates"))
    lines = [_head(head)]
    if k == 0:
        return lines + _emit(items, brief)

    mn, mx = x.min(), x.max()
    has_time = bool((x.dt.normalize() != x).any())
    fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
    days = x.dt.normalize()
    distinct = int(days.nunique())
    span = int((mx.normalize() - mn.normalize()).days)

    items.append((
        "Range",
        f"{mn.strftime(fmt)} to {mx.strftime(fmt)}, a span of {_span_text(span)}, across "
        f"{_s(distinct, 'distinct day')}.",
    ))
    if x.is_monotonic_increasing:
        items.append(("Order", "Rows run oldest to newest."))
    elif x.is_monotonic_decreasing:
        items.append(("Order", "Rows run newest to oldest."))
    else:
        items.append(("Order", "Rows are not sorted by this date."))

    today = pd.Timestamp.today().normalize()
    if mx.normalize() > today:
        nf = int((days > today).sum())
        share = nf / k
        if share >= 0.5:
            items.append((
                "Note",
                f"{share * 100:.1f}% of the dates are after today, so this looks like a schedule or "
                "forecast rather than a log of past events.",
            ))
        else:
            items.append((
                "Watch",
                f"{_s(nf, 'date')} ({share * 100:.1f}%) are after today. If this is a log of past events, "
                "those dates are probably errors.",
            ))
            flags.append((name, f"{nf:,} dates after today (probably errors in a log)"))
    else:
        age = int((today - mx.normalize()).days)
        if age > 0:
            items.append(("Freshness", f"Latest record is {_s(age, 'day')} old."))

    if span > 0:
        if span <= 45:
            freq, unit = "D", "day"
        elif span <= 400:
            freq, unit = "W", "week"
        else:
            freq, unit = "M", "month"
        counts = x.dt.to_period(freq).value_counts().sort_index()
        full = pd.period_range(counts.index.min(), counts.index.max(), freq=freq)
        counts = counts.reindex(full, fill_value=0)
        P = len(counts)

        def lab(p):
            return ("week of " + p.start_time.strftime("%Y-%m-%d")) if freq == "W" else str(p)

        # first and last periods are often partial, so judge evenness on the inside
        inner = counts.iloc[1:-1] if P >= 5 else counts
        empty = int((inner == 0).sum())
        top_p = counts.idxmax()
        top_share = counts.max() / k
        cv = inner.std() / inner.mean() if len(inner) > 1 and inner.mean() > 0 else 0

        items.append((
            "Timeline",
            f"{_spark_counts(counts.values)}  (rows per {unit}, left = oldest; a dot means no records)",
        ))

        if distinct <= 10:
            dc = days.value_counts().sort_index()
            items.append(("Rows per date", "; ".join(f"{d.strftime('%Y-%m-%d')}: {c} ({_pct(c, k)})" for d, c in dc.items()) + "."))
        else:
            top3 = counts.sort_values(ascending=False).head(3)
            items.append((f"Busiest {unit}s", "; ".join(f"{lab(p)} ({c}, {_pct(c, k)})" for p, c in top3.items()) + "."))
            nz = inner[inner > 0].sort_values().head(3)
            if len(nz):
                items.append((f"Quietest {unit}s", "; ".join(f"{lab(p)} ({c})" for p, c in nz.items()) + "."))

        if empty:
            items.append(("Empty", f"{empty} of {len(inner)} {unit}s inside the range have no records at all."))

        if top_share >= 0.5 and P >= 3:
            verdict = f"Very uneven: {top_share * 100:.0f}% of all rows sit in one {unit} ({lab(top_p)})."
            flags.append((name, f"{top_share * 100:.0f}% of rows fall in a single {unit}"))
        elif cv >= 1:
            verdict = f"Highly uneven across {unit}s (the swings between {unit}s are bigger than the average count)."
            flags.append((name, f"highly uneven coverage across {unit}s"))
        elif cv >= 0.5 or empty:
            verdict = f"Moderately uneven across {unit}s."
            if empty:
                flags.append((name, f"{empty} empty {unit}s inside the range"))
        else:
            verdict = f"Fairly even across {unit}s."
        items.append(("Coverage", verdict + " Compare periods with rates or percentages, not raw counts."))

        ud = days.drop_duplicates().sort_values()
        gaps = ud.diff().dt.days
        if gaps.notna().any() and gaps.max() >= 3:
            end = ud.loc[gaps.idxmax()]
            start = ud[ud < end].max()
            items.append((
                "Gap",
                f"Longest stretch with no data: {int(gaps.max())} days "
                f"({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}).",
            ))

    if distinct >= 7 and span >= 14:
        wd = x.dt.day_name().value_counts()
        items.append((
            "Weekday",
            f"Busiest is {wd.index[0]} ({_pct(wd.iloc[0], k)}), quietest is {wd.index[-1]} ({_pct(wd.iloc[-1], k)}).",
        ))
    if has_time and k > 1:
        hrs = x.dt.hour.value_counts()
        items.append((
            "Time of day",
            f"Busiest hour is {int(hrs.index[0]):02d}:00 ({_pct(hrs.iloc[0], k)}); records run from "
            f"{int(x.dt.hour.min()):02d}:00 to {int(x.dt.hour.max()):02d}:59.",
        ))

    # brief mode: surface the coverage verdict as the takeaway
    if brief:
        for lbl, txt in items:
            if lbl == "Coverage":
                items.append(("Takeaway", txt))
                break
    return lines + _emit(items, brief)


# ------------------------------------------------------------- categorical
def _cat_block(s, name, flags, top_n, brief, hidden_map):
    n = len(s)
    vc_all = s.value_counts()  # drops missing; one pass over the data, everything else uses uniques
    miss = n - int(vc_all.sum())

    def _share(c):
        return _pct(c, n)

    labels = pd.Series(vc_all.index.astype(str), index=vc_all.index)
    ph_u = labels.str.strip().str.lower().isin(MISSING_TOKENS).values
    hidden = int(vc_all.values[ph_u].sum())

    # up to top_n values are listed, the rest are lumped into "others"
    shown = "; ".join(f"{v} ({_share(c)})" for v, c in vc_all.head(top_n).items())
    rest_vc = vc_all.iloc[top_n:]
    if len(rest_vc):
        shown += f" and {len(rest_vc):,} other{'s' if len(rest_vc) != 1 else ''} ({_share(int(rest_vc.sum()))})"
    head = f"- {name} [categorical]: {n - miss - hidden:,} valid values, {_pct(miss + hidden, n)} missing"
    if hidden:
        head += f" (counting {hidden:,} placeholders like '-' as missing)"
    lines = [_head(head)]
    items = [
        ("Distinct values", f"{len(vc_all):,}."),
        ("Top values", shown + "."),
    ]

    if hidden:
        hidden_map[name] = hidden
        items.append((
            "Watch",
            f"{_s(hidden, 'value')} ({hidden / n * 100:.1f}% of rows) are placeholders for missing "
            "('-', 'NA', blank, etc.); treat them as missing.",
        ))
        flags.append((name, f"{hidden / n * 100:.1f}% hidden missing (placeholder text)"))

    ws_u = ((labels != labels.str.strip()).values) & ~ph_u
    ws = int(vc_all.values[ws_u].sum())
    if ws:
        items.append(("Watch", f"{_s(ws, 'value')} have extra spaces at the start or end."))
        flags.append((name, f"{ws:,} values with stray whitespace"))

    vc = vc_all[~ph_u]
    lab_real = labels[~ph_u]
    norm = lab_real.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    if len(lab_real) and norm.nunique() < len(lab_real):
        grp = pd.DataFrame({"raw": lab_real.values, "norm": norm.values}).groupby("norm")["raw"].apply(list)
        grp = grp[grp.str.len() > 1].head(2)
        ex = "; ".join(" / ".join(repr(v) for v in g[:3]) for g in grp)
        items.append(("Watch", f"Same label written different ways (spacing or capitalization), e.g. {ex}."))
        flags.append((name, "inconsistent labels (case or spacing)"))
    if pd.Series(vc_all.index).map(type).nunique() > 1:
        items.append(("Watch", "Column mixes different data types (for example numbers and text)."))
        flags.append((name, "mixed data types in one column"))

    k, nun = int(vc.sum()), len(vc)
    if k == 0:
        items.append(("Watch", "Every value is a placeholder, so this column is effectively empty."))
        flags.append((name, "effectively empty (all placeholder text)"))
        return lines + _emit(items, brief)

    top_share = vc.iloc[0] / k
    if nun == 1:
        if hidden:
            items.append(("Watch", f"Only one real value ('{vc.index[0]}') appears; everything else is placeholder."))
            flags.append((name, f"only one real value ('{vc.index[0]}')"))
        else:
            items.append(("Watch", "Only one value appears, so this column adds nothing for analysis."))
            flags.append((name, "constant column"))
    elif nun == k and k > 10:
        items.append(("Note", "Every value is different, so this looks like an ID, name, or free text rather than a category."))
    elif nun / k > 0.9 and k > 20:
        repeats = k - nun
        items.append((
            "Watch",
            f"Nearly every value is unique ({nun:,} distinct in {k:,} rows); {_s(repeats, 'row')} repeat an "
            "earlier value. If this is meant to be a unique ID, look at those repeats.",
        ))
        flags.append((name, f"near-unique ID-like column with {repeats:,} repeated values"))
    elif nun == 2:
        items.append((
            "Note",
            f"Two-value column: '{vc.index[0]}' ({top_share * 100:.0f}%) vs '{vc.index[1]}' ({(1 - top_share) * 100:.0f}%).",
        ))
    elif top_share >= 0.8:
        items.append((
            "Watch",
            f"Dominated by '{vc.index[0]}' ({top_share * 100:.0f}% of real values), so comparisons "
            "between categories will rest on very few rows.",
        ))
        flags.append((name, f"one value makes up {top_share * 100:.0f}% of rows"))
    elif nun / k > 0.5 and k > 20:
        items.append(("Note", "Many distinct values for the number of rows; likely names, IDs, or free text."))
    elif nun <= 15:
        items.append(("Note", f"Low-cardinality column ({nun} values); suitable for grouping and bar charts."))
    else:
        n80 = int(((vc.cumsum() / k) < 0.8).sum() + 1)
        items.append(("Note", f"{n80} of {nun} values account for 80% of real values."))

    if nun > 5 and nun / k <= 0.5:
        rare = int((vc / n < 0.01).sum())
        if rare:
            items.append(("Note", f"{rare:,} of {nun:,} values each cover under 1% of rows; consider grouping them as 'Other' for charts."))

    return lines + _emit(items, brief)


def _bool_block(s, name, brief):
    n = len(s)
    vc = s.value_counts()
    parts = "; ".join(f"{v} ({_pct(c, n)})" for v, c in vc.items())
    miss = int(s.isna().sum())
    lines = [_head(f"- {name} [boolean]: {n - miss:,} valid values, {_pct(miss, n)} missing")]
    return lines + _emit([("Top values", parts + ".")], brief)


def _empty_block(s, name, flags):
    flags.append((name, "completely empty; candidate to drop"))
    return [_head(
        f"- {name} [empty]: 100.00% missing ({len(s)} of {len(s)}). Nothing to analyze; "
        "drop it unless it is meant to be filled later."
    )]


# ---------------------------------------------------------- dataset checks
def _identical_columns(df):
    seen, dupes = {}, []
    for col in df.columns:
        try:
            key = pd.util.hash_pandas_object(df[col], index=False).values.tobytes()
        except TypeError:
            key = pd.util.hash_pandas_object(df[col].astype(str), index=False).values.tobytes()
        if key in seen:
            dupes.append((seen[key], col))
        else:
            seen[key] = col
    return dupes


def _eligible_numeric(numeric_cols):
    """Columns worth comparing: real numbers, enough data, not constant, not an ID-like counter."""
    keep = {}
    for name, ser in numeric_cols.items():
        v = ser.dropna()
        if len(v) > 20 and v.nunique() > 1 and not (v.nunique() == len(v) and bool((v % 1 == 0).all())):
            keep[name] = ser
    return keep


def _numeric_links(keep, skip, min_r=0.0, max_pairs=8):
    """Rank correlation between numeric columns, strongest first. Returns the list of (a, b, r)."""
    if len(keep) < 2:
        return []
    num = pd.DataFrame({k: v.to_numpy() for k, v in keep.items()})
    if len(num) > 50_000:
        num = num.sample(50_000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = num.corr(method="spearman")
    out = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if (a, b) in skip or (b, a) in skip:
                continue
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= min_r:
                out.append((a, b, float(r)))
    out.sort(key=lambda t: -abs(t[2]))
    return out[:max_pairs]


def _eta_from_ranks(r, codes):
    """How much a category explains a number (0 to 1), on ranks, corrected for many small groups."""
    m = ~np.isnan(r) & (codes >= 0)
    if m.sum() < 10:
        return np.nan
    v, c = r[m], codes[m]
    counts = np.bincount(c)
    nz = counts > 0
    k, N = int(nz.sum()), len(v)
    if k < 2 or N <= k:
        return np.nan
    sums = np.bincount(c, weights=v)
    grand = v.mean()
    ssb = (counts[nz] * (sums[nz] / counts[nz] - grand) ** 2).sum()
    sst = ((v - grand) ** 2).sum()
    if sst <= 0:
        return np.nan
    msw = (sst - ssb) / (N - k)
    om = (ssb - (k - 1) * msw) / (sst + msw)
    return float(np.sqrt(max(om, 0.0)))


def _num_cat_links(df, keep, cats, min_eta=0.0, max_levels=30, max_pairs=5):
    use = [c for c in cats if 2 <= df[c].nunique() <= max_levels]
    if not keep or not use:
        return [], len(keep) * len(use)
    d = df.sample(100_000, random_state=0) if len(df) > 100_000 else df
    ranks = {nm: pd.to_numeric(keep[nm].loc[d.index], errors="coerce").rank().to_numpy(float) for nm in keep}
    codes = {c: pd.factorize(d[c])[0] for c in use}
    out = []
    for nm, r in ranks.items():
        for c in use:
            e = _eta_from_ranks(r, codes[c])
            if pd.notna(e) and e >= min_eta:
                out.append((nm, c, e))
    out.sort(key=lambda t: -t[2])
    return out[:max_pairs], len(keep) * len(use)


def _cramers_v(a, b):
    """Bias-corrected Cramér's V between two categorical columns (0 = unrelated, 1 = fully tied)."""
    t = pd.crosstab(a, b)
    r, c = t.shape
    n = t.values.sum()
    if r < 2 or c < 2 or n < 2:
        return np.nan
    obs = t.values.astype(float)
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    phi2 = ((obs - exp) ** 2 / exp).sum() / n
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    d = min(rc - 1, cc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else np.nan


def _cat_links(df, cats, max_pairs=5, min_v=0.0, max_levels=30):
    use = [c for c in cats if 2 <= df[c].nunique() <= max_levels]
    d = df.sample(50_000, random_state=0) if len(df) > 50_000 else df
    pairs = []
    for i, a in enumerate(use):
        for b in use[i + 1:]:
            sub = d[[a, b]].dropna()
            v = _cramers_v(sub[a], sub[b]) if len(sub) > 1 else np.nan
            if pd.notna(v) and v >= min_v:
                pairs.append((a, b, v))
    pairs.sort(key=lambda t: -t[2])
    return pairs[:max_pairs], len(use)


def _dups_ignoring_ids(df):
    """Rows that repeat once ID-like columns (all values unique, not float/date) are set aside."""
    n = len(df)
    ids = [
        c for c in df.columns
        if n > 1
        and not pd.api.types.is_float_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and df[c].nunique(dropna=False) == n
    ]
    rest = [c for c in df.columns if c not in ids]
    if not ids or not rest:
        return ids, 0
    return ids, int(df.duplicated(subset=rest).sum())


def _label(v, strong, moderate, weak):
    v = abs(v)
    if v >= strong:
        return "strong"
    if v >= moderate:
        return "moderate"
    return "weak" if v >= weak else "very weak or none"


def _relations_section(num_pairs, n_num, converted, nc_pairs, n_nc, cat_pairs, n_cat):
    """'How the columns relate': three short lists, one pair per line."""
    body = []

    def block(title, rows):
        body.append(_wrap(title, 4, "  "))
        for r in rows:
            body.append(_wrap(r, 6, "    "))

    if n_num >= 2 and num_pairs:
        rows = []
        for a, b, r in num_pairs:
            lab = _label(r, 0.7, 0.5, 0.3)
            if abs(r) >= 0.95:
                rows.append(f"{a} ~ {b}: {r:+.2f}, almost the same column")
            elif abs(r) >= 0.3:
                way = "rise together" if r > 0 else "one rises as the other falls"
                rows.append(f"{a} ~ {b}: {r:+.2f}, {lab} ({way})")
            else:
                rows.append(f"{a} ~ {b}: {r:+.2f}, {lab}")
        block(
            "Numbers vs numbers (rank correlation: +1 = rise together, -1 = one rises as the other falls, 0 = unrelated):",
            rows,
        )
        if converted:
            body.append(_wrap(
                "Converted from text just for these comparisons (the columns themselves stay text): "
                + ", ".join(converted) + ".", 4, "  "))
    if nc_pairs:
        block(
            "Numbers vs categories (0 = the category tells you nothing about the number, 1 = it explains it fully):",
            [f"{a} by {c}: {e:.2f}, {_label(e, 0.5, 0.3, 0.1)}" for a, c, e in nc_pairs],
        )
    if cat_pairs:
        block(
            "Categories vs categories (Cramér's V: 0 = unrelated, 1 = knowing one tells you the other):",
            [f"{a} ~ {b}: {v:.2f}, {_label(v, 0.5, 0.3, 0.1)}" for a, b, v in cat_pairs],
        )
    return (["How the columns relate:"] + body + [""]) if body else []


_FLAG_CODES = [
    ("sentinel", r"placeholder code"),
    ("units", r"mixed or missing units"),
    ("text_numbers", r"stored as text"),
    ("placeholders", r"hidden missing"),
    ("useless", r"constant column|only one real value|completely empty|effectively empty"),
    ("redundant", r"identical to column|in lockstep"),
    ("labels", r"inconsistent labels|stray whitespace"),
    ("mixed", r"mixed data types"),
    ("missing", r"^\d+% missing$"),
    ("dates", r"dates after today|uneven|single (day|week|month)|empty (day|week|month)|earlier than|comes before|could not be read as dates"),
    ("extremes", r"outliers by the IQR|strong skew|zero-inflated"),
]


def _flag_code(msg):
    for code, pat in _FLAG_CODES:
        if re.search(pat, msg):
            return code
    return None


def _by_code(flags):
    groups = {}
    for col, msg in flags:
        code = _flag_code(msg)
        if code and col not in groups.setdefault(code, []):
            groups[code].append(col)
    return groups


def _cols(cols, limit=8):
    shown = ", ".join(cols[:limit])
    return shown + (f" and {len(cols) - limit} more" if len(cols) > limit else "")


def _checklist(groups, dups):
    rules = [
        ("text_numbers", "Convert to real numbers (strip units, currency, % signs, commas)"),
        ("placeholders", "Replace placeholder text like '-' or 'NA' with true missing values"),
        ("sentinel", "Replace all-9s placeholder codes with missing values"),
        ("units", "Separate mixed units before comparing values"),
        ("labels", "Standardize labels (spacing and capitalization)"),
        ("mixed", "Fix columns that mix data types"),
        ("useless", "Drop columns that carry no information"),
        ("redundant", "Keep one of each duplicated or lockstep column"),
        ("missing", "Decide how to handle heavy missingness (20% or more)"),
        ("dates", "Review date issues (future dates, gaps, uneven coverage, wrong order)"),
        ("extremes", "Open the raw rows behind extreme values before trimming, capping, or keeping them"),
    ]
    out = [f"{label}: {_cols(groups[code])}." for code, label in rules if code in groups]
    if dups:
        out.append(f"Review {dups:,} duplicate rows and decide whether repeats are real records or errors.")
    return out


# ------------------------------------------------------------- date order
_EARLY_LATE = [
    ("departure", "arrival"), ("depart", "arriv"), ("start", "end"), ("first", "last"),
    ("open", "close"), ("creat", "close"), ("begin", "finish"), ("from", "to"),
]


def _tokens(name):
    return [w for w in re.split(r"[^a-z0-9]+", str(name).lower()) if w]


def _date_pairs(cols):
    """Pairs like (departure_time, arrival_time) where one date should not come before the other."""
    pairs = []
    for a in cols:
        for b in cols:
            if a == b:
                continue
            ta, tb = _tokens(a), _tokens(b)
            for e, l in _EARLY_LATE:
                ea = [w for w in ta if w.startswith(e)]
                lb = [w for w in tb if w.startswith(l)]
                if ea and lb:
                    stem_a = [w for w in ta if w not in ea]
                    stem_b = [w for w in tb if w not in lb]
                    if stem_a == stem_b and (a, b) not in pairs:
                        pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------- main
def diagnose(df, date_cols=None, dayfirst=False, top_n=5, brief=False, return_text=False):
    """
    Print a plain-language diagnosis of a DataFrame.

    date_cols   : columns to force-parse as dates (dates stored as text are also auto-detected).
    dayfirst    : True if text dates are day/month/year (03/04/2026 means 3 April).
    top_n       : how many top values to show for categorical columns.
    brief       : only header stats, warnings, and takeaways for each column.
    return_text : return the report as a string instead of printing it.
    """
    if type(df).__module__.startswith("polars"):  # accept a Polars DataFrame (needs pyarrow installed)
        df = df.to_pandas()
    if not df.index.is_unique:  # e.g. after pd.concat; duplicate labels break alignment-based steps
        df = df.reset_index(drop=True)
    force_dates = set(date_cols or [])
    n, p = df.shape
    out = [f"The data contains {n:,} observations of the following {p} variables:", ""]

    flags, blocks, kinds = [], [], []
    numeric_series, date_series, hidden_map, text_nums = {}, {}, {}, []
    for col in df.columns:
        s = df[col]
        kind, ser, info = _classify(s, col, force_dates, dayfirst)
        kinds.append(kind)
        if kind == "empty":
            blk = _empty_block(s, col, flags)
        elif kind == "bool":
            blk = _bool_block(ser, col, brief)
        elif kind == "numeric":
            blk = _numeric_block(ser, col, flags, info, brief)
            numeric_series[col] = ser.astype(float)
            if info.get("from_text"):
                text_nums.append(col)  # stays text in its own block; converted only for comparisons
            if info.get("placeholders"):
                hidden_map[col] = info["placeholders"]
        elif kind == "date":
            blk = _date_block(ser, col, flags, info, brief)
            date_series[col] = ser
        else:
            blk = _cat_block(ser, col, flags, top_n, brief, hidden_map)
        blocks += blk + [""]  # blank line after every column
        if kind != "empty":
            m = (ser if kind == "numeric" else s).isna().mean()
            if m >= 0.2:
                flags.append((col, f"{m * 100:.0f}% missing"))

    # overview
    kc = Counter(kinds)
    order = ["numeric", "categorical", "date", "bool", "empty"]
    types = ", ".join(f"{kc[k]} {k}" for k in order if kc[k])
    dups = int(df.duplicated().sum())
    cells = n * p
    miss_cells = int(df.isna().sum().sum())
    hid_total = sum(hidden_map.values())
    out.append(_wrap(f"Column types: {types}.", 2))
    out.append(_wrap(f"Duplicate rows: {dups:,} ({_pct(dups, n)}).", 2))
    id_cols, dups_noid = _dups_ignoring_ids(df)
    if id_cols:
        out.append(_wrap(
            f"Duplicate rows ignoring ID-like columns ({', '.join(id_cols)}): {dups_noid:,} ({_pct(dups_noid, n)}).", 2))
    if id_cols:
        out.append(_wrap(f"Possible row identifier (every value unique): {', '.join(id_cols)}.", 2))
    elif n > 1:
        out.append(_wrap("No single column uniquely identifies a row; decide what one row represents before counting.", 2))
    samp = df.sample(200_000, random_state=0) if n > 200_000 else df
    sparse = int((samp.isna().mean(axis=1) > 0.5).sum())
    if sparse:
        out.append(_wrap(f"Rows with more than half their cells missing: about {sparse / len(samp) * 100:.1f}% of rows.", 2))
    line = f"Missing cells: {miss_cells:,} of {cells:,} ({_pct(miss_cells, cells)})"
    if hid_total:
        line += f"; counting placeholders like '-': {miss_cells + hid_total:,} ({_pct(miss_cells + hid_total, cells)})"
    out.append(_wrap(line + ".", 2))

    miss_by_col = df.isna().mean()
    miss_by_col = miss_by_col[(miss_by_col > 0) & (miss_by_col < 1)].sort_values(ascending=False).head(6)
    if len(miss_by_col):
        out.append(_wrap("Most missing: " + ", ".join(f"{c} ({v * 100:.1f}%)" for c, v in miss_by_col.items()) + ".", 2))
    if hidden_map:
        top_h = sorted(hidden_map.items(), key=lambda kv: -kv[1])[:6]
        out.append(_wrap(
            "Placeholder text (counts as missing): " + ", ".join(f"{c} ({v / n * 100:.1f}%)" for c, v in top_h) + ".", 2))

    same = _identical_columns(df)
    for a, b in same:
        flags.append((b, f"identical to column '{a}' (redundant)"))
    if same:
        out.append(_wrap("Identical columns: " + "; ".join(f"{a} = {b}" for a, b in same) + ".", 2))

    keep = _eligible_numeric(numeric_series)
    num_pairs = _numeric_links(keep, {(a, b) for a, b in same})
    for a, b, r in num_pairs:
        if abs(r) >= 0.95:
            flags.append((b, f"moves almost in lockstep with '{a}' ({r:.2f}); may be redundant"))
    cats = [c for c, k in zip(df.columns, kinds) if k == "categorical"]
    nc_pairs, n_nc = _num_cat_links(df, keep, cats)
    cat_pairs, n_cat = _cat_links(df, cats)
    converted = [c for c in text_nums if c in keep]
    rel_lines = _relations_section(num_pairs, len(keep), converted, nc_pairs, n_nc, cat_pairs, n_cat)

    # date order checks (e.g. arrival should not be before departure)
    for a, b in _date_pairs(list(date_series)):
        both = date_series[a].notna() & date_series[b].notna()
        bad = int(((date_series[b] < date_series[a]) & both).sum())
        if bad:
            out.append(_wrap(
                f"Date check: in {bad:,} rows ({_pct(bad, int(both.sum()))}), {b} comes before {a}, which looks wrong.", 2))
            flags.append((b, f"comes before '{a}' in {bad:,} rows"))
        else:
            out.append(_wrap(f"Date check: {b} never comes before {a}. All {int(both.sum()):,} rows are in the right order.", 2))
    # at a glance: a plain-English read of the whole dataset
    groups = _by_code(flags)
    glance = [f"{n:,} rows and {p} columns ({types})."]
    for dc, ds in list(date_series.items())[:2]:
        v = ds.dropna()
        if len(v):
            glance.append(f"{dc} runs from {v.min():%Y-%m-%d} to {v.max():%Y-%m-%d}.")
    problems = []
    names = {  # (one column, several columns)
        "text_numbers": ("holds numbers stored as text", "hold numbers stored as text"),
        "placeholders": ("uses '-' or 'NA' text for missing", "use '-' or 'NA' text for missing"),
        "sentinel": ("contains an all-9s placeholder code", "contain all-9s placeholder codes"),
        "useless": ("carries no information", "carry no information"),
        "redundant": ("duplicates another column", "duplicate another column"),
        "labels": ("has messy labels", "have messy labels"),
        "missing": ("is 20% or more missing", "are 20% or more missing"),
        "dates": ("has date issues", "have date issues"),
        "extremes": ("has extreme values or heavy skew", "have extreme values or heavy skew"),
    }
    for code, (one, many) in names.items():
        if code in groups:
            num = len(groups[code])
            problems.append(f"{_s(num, 'column')} {one if num == 1 else many}")
    if dups:
        problems.append(f"{dups:,} duplicate rows")
    glance.append(("Needs attention: " + "; ".join(problems) + ".") if problems else "No major quality problems detected.")
    out[2:2] = ["At a glance:"] + [_wrap(g, 4, "  - ") for g in glance] + [""]

    out.append("")
    out += blocks
    out += rel_lines

    if flags or dups:
        out.append("Worth a closer look:")
        grouped = {}
        for col, msg in flags:
            grouped.setdefault(col, []).append(msg)
        for col, msgs in grouped.items():
            out.append(_wrap(f"- {col}: " + "; ".join(msgs) + ".", 4, "  "))
        if dups:
            out.append(f"  - dataset: {dups:,} fully duplicated rows ({_pct(dups, n)}).")
    else:
        out.append("Nothing alarming stood out.")

    todo = _checklist(groups, dups)
    if todo:
        out.append("")
        out.append("Cleaning checklist (do these before the analysis):")
        for i, line in enumerate(todo, 1):
            out.append(_wrap(line, 5, f"  {i}. "))

    text = "\n".join(out)
    if return_text:
        return text
    print(text)


# Make it callable as a method: df.diagnose()
pd.DataFrame.diagnose = diagnose


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        diagnose(pd.read_csv(sys.argv[1]))
    else:
        print("Usage: python diagnose.py yourfile.csv")