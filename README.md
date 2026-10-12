## reportpy

**A plain-language report on any pandas DataFrame, in one line.**

```python
import reportpy

df.diagnose()
```

reportpy reads your data and tells you, in plain sentences, what is in each column, what looks wrong, and what to clean first. It is inspired by R's `report` package, with extra checks for dates, distributions and data quality.

### Why reportpy

Getting a first look at a new dataset usually means running a dozen separate snippets: `.info()`, `.describe()`, missing-value counts, value counts, date ranges, duplicate checks, correlations. Profiling tools like ydata-profiling and sweetviz bundle much of this, but they can be slow on large data and they leave the interpretation to you.

reportpy does it in one call and writes the findings as sentences. It also looks for the problems that quietly spoil an analysis:

- placeholder codes such as `9999` or `-999` hiding in numeric columns
- numbers stored as text (`KES 1,200`, `45%`, `12/kg`)
- `-`, `NA` or blank text standing in for missing values
- dates in the future, uneven coverage, long gaps, and arrival dates before departure dates
- the same label written different ways (`Nairobi`, `nairobi `, `NAIROBI`)
- constant columns, ID-like columns, duplicate rows and duplicate columns

The expensive steps run on unique values or on a random sample, so the report stays quick on big tables.

### Install

```bash
pip install reportpy
```

or with uv:

```bash
uv add reportpy
```

Requires Python 3.9 or newer, pandas 2.0 or newer, and numpy.

### Quick start

Importing the package adds a `diagnose()` method to every DataFrame:

```python
import pandas as pd
import reportpy

df = pd.read_csv("flights.csv")
df.diagnose()
```

You can also call it as a function:

```python
from reportpy import diagnose

diagnose(df)
```

### Options

```python
diagnose(df, date_cols=None, dayfirst=False, top_n=5, brief=False, return_text=False)
```

| Option        | Default | What it does                                                                    |
| ------------- | ------- | ------------------------------------------------------------------------------- |
| `date_cols`   | `None`  | Columns to read as dates. Dates stored as text are also detected automatically. |
| `dayfirst`    | `False` | Set to `True` if text dates are day/month/year, so `03/04/2026` means 3 April.  |
| `top_n`       | `5`     | How many top values to list for each categorical column.                        |
| `brief`       | `False` | Short version: header stats, warnings and takeaways only.                       |
| `return_text` | `False` | Return the report as a string instead of printing it.                           |

Examples:

```python
df.diagnose(brief=True)                                  # the short version
diagnose(df, date_cols=["Date"], dayfirst=True)          # text dates, day first
text = diagnose(df, return_text=True)                    # keep the report as a string
```

Polars DataFrames are accepted too. They are converted to pandas first, which needs `pyarrow` installed.

### What the report covers

**At a glance.** Rows, columns, column types, the date range of the first date columns, and a one-line summary of what needs attention.

**The whole dataset.**
- duplicate rows, also counted after setting aside ID-like columns
- a possible row identifier (a column where every value is unique)
- missing cells, also counted with placeholder text such as `-` treated as missing
- rows that are more than half empty, and the columns with the most missing values
- identical columns
- date order checks for pairs like departure/arrival or start/end

**Numeric columns.**
- mean, standard deviation, median, MAD, skewness and kurtosis, with a small histogram
- a plain-terms summary, then readings of the shape, center, tails and spread
- possible mixed groups (bimodality), outliers by the IQR rule, zeros and signs
- numbers stored as text, placeholder codes made of 9s, ID-like counters, constant columns and yes/no flags
- a takeaway telling you which summary to trust

**Date columns.**
- range, span, sort order, future dates and how old the latest record is
- a timeline of rows per day, week or month, with the busiest and quietest periods
- empty periods, how even the coverage is, and the longest gap
- busiest weekday and busiest hour of the day

**Categorical columns.**
- distinct values and top values
- placeholder text counted as missing, stray whitespace and inconsistent labels
- columns that are constant, ID-like, or dominated by one value
- rare values you may want to group as "Other"

**How the columns relate.**
- numbers vs numbers, by rank correlation
- numbers vs categories, by how much the category explains the number
- categories vs categories, by Cramér's V

**Worth a closer look and a cleaning checklist.** Every warning collected in one place, followed by an ordered list of what to fix before analysis.

### Example output

Eight thousand flights, three months (October to December 2026), one departure airport and three destinations. This is the full report reportpy produced for it. Every flight leaves Jomo Kenyatta International Airport and lands at Moi International Airport, Kisumu International Airport or Eldoret International Airport, with four airlines sharing the routes.

```text
The data contains 8,000 observations of the following 10 variables:

At a glance:
  - 8,000 rows and 10 columns (2 numeric, 6 categorical, 2 date).
  - departure_time runs from 2026-10-01 to 2026-12-31.
  - arrival_time runs from 2026-10-01 to 2027-01-01.
  - Needs attention: 1 column carries no information; 1 column has extreme values or heavy skew.

  Column types: 2 numeric, 6 categorical, 2 date.
  Duplicate rows: 0 (0.00%).
  No single column uniquely identifies a row; decide what one row represents before counting.
  Missing cells: 0 of 80,000 (0.00%).
  Date check: arrival_time never comes before departure_time. All 8,000 rows are in the right order.

- departure_airport [categorical]: 8,000 valid values, 0.00% missing
      Distinct values: 1.
      Top values: Jomo Kenyatta International Airport (100.00%).
      Watch: Only one value appears, so this column adds nothing for analysis.

- departure_time [date]: 8,000 valid dates, 0.00% missing
      Range: 2026-10-01 05:45 to 2026-12-31 23:59, a span of 91 days (about 3.0 months), across 92
        distinct days.
      Order: Rows are not sorted by this date.
      Note: 89.9% of the dates are after today, so this looks like a schedule or forecast rather
        than a log of past events.
      Timeline: ▄▆▆▇▇▇▇███▇██▅  (rows per week, left = oldest; a dot means no records)
      Busiest weeks: week of 2026-12-14 (688, 8.60%); week of 2026-11-30 (682, 8.53%); week of
        2026-12-21 (654, 8.18%).
      Quietest weeks: week of 2026-10-05 (465); week of 2026-10-12 (495); week of 2026-10-19 (561).
      Coverage: Fairly even across weeks. Compare periods with rates or percentages, not raw counts.
      Weekday: Busiest is Thursday (15.15%), quietest is Tuesday (13.06%).
      Time of day: Busiest hour is 17:00 (10.85%); records run from 05:00 to 23:59.

- airline [categorical]: 8,000 valid values, 0.00% missing
      Distinct values: 4.
      Top values: Jambojet Limited (49.67%); Kenya Airways (38.74%); Skyward Express (7.99%);
        Safarilink (3.60%).
      Note: Low-cardinality column (4 values); suitable for grouping and bar charts.

- flight_number [categorical]: 8,000 valid values, 0.00% missing
      Distinct values: 44.
      Top values: KQ 602 (4.58%); KQ 670 (4.53%); KQ 624 (4.50%); KQ 612 (4.47%); KQ 606 (4.46%) and
        39 others (77.46%).
      Note: 29 of 44 values account for 80% of real values.
      Note: 4 of 44 values each cover under 1% of rows; consider grouping them as 'Other' for
        charts.

- travel_class [categorical]: 8,000 valid values, 0.00% missing
      Distinct values: 2.
      Top values: Economy (80.70%); Business Class (19.30%).
      Note: Two-value column: 'Economy' (81%) vs 'Business Class' (19%).

- type_of_flight [categorical]: 8,000 valid values, 0.00% missing
      Distinct values: 2.
      Top values: One way (50.36%); Round trip (49.64%).
      Note: Two-value column: 'One way' (50%) vs 'Round trip' (50%).

- total_duration [numeric] ▂▁▁█▁▁▁▄: 8,000 valid values, 0.00% missing
      Key stats: Mean = 62.32, SD = 7.70, Median = 60, MAD = 0, Skewness = 0.58, Kurtosis = -0.53.
      Range: 50 to 75.
      In plain terms: Most common value is 60 (62.21% of rows); values run from 50 to 75.
      Type: Only 4 distinct values, so this is probably a count, a rating scale, or a measurement
        recorded in fixed steps; counts per value will say more than the mean.
      Shape: moderately skewed to the right, with a tail of unusually large values.
      Center: Mean (62.32) is above the median (60), so a few large values are dragging it; the
        median is the better 'typical value'.
      Tails: Tail weight is close to a bell curve.
      Spread: MAD is 0: at least half of the values equal the median exactly, so one repeated value
        dominates. SD is only 12% of the mean, so values are tightly clustered.
      Values: 50 (10.53%); 55 (3.60%); 60 (62.21%); 75 (23.66%).
      Takeaway: mildly off a bell shape; mean and median together tell the story.

- ticket_price [numeric] █▅▆▂▁▁▂▂: 8,000 valid values, 0.00% missing
      Key stats: Mean = 19,644.35, SD = 11,347.26, Median = 17,286, MAD = 9,945.28, Skewness = 1.59,
        Kurtosis = 1.96.
      Range: 7,482 to 55,986.
      In plain terms: A typical value is about 17,286. Half of the rows fall between 10,578 and
        22,446; the full range runs from 7,482 to 55,986, but only 1% of rows go above 51,600, so a
        few very large values stretch the top end.
      Shape: strongly skewed to the right, with a tail of unusually large values.
      Center: Mean (19,644) is slightly above the median (17,286); not a big gap.
      Tails: Kurtosis 1.96: somewhat heavy tails, expect a few extreme values.
      Spread: SD and MAD agree, so spread is consistent across the whole range.
      Typical range: Middle 50% of values: 10,578 to 22,446. 1st to 99th percentile: 8,127 to
        51,600.
      Outliers: By the 1.5 x IQR rule, 782 values (9.8%) above 40,248; none are unusually low. The
        data is skewed, so many of these are probably the natural long tail, not errors.
      Takeaway: right-skewed with a long tail of large values; lean on median and percentiles, and
        consider a log transform.

- arrival_airport [categorical]: 8,000 valid values, 0.00% missing
      Distinct values: 3.
      Top values: Moi International Airport (59.86%); Kisumu International Airport (28.30%); Eldoret
        International Airport (11.84%).
      Note: Low-cardinality column (3 values); suitable for grouping and bar charts.

- arrival_time [date]: 8,000 valid dates, 0.00% missing
      Range: 2026-10-01 06:40 to 2027-01-01 00:59, a span of 92 days (about 3.0 months), across 93
        distinct days.
      Order: Rows are not sorted by this date.
      Note: 89.9% of the dates are after today, so this looks like a schedule or forecast rather
        than a log of past events.
      Timeline: ▄▆▆▇▇▇▇███▇██▅  (rows per week, left = oldest; a dot means no records)
      Busiest weeks: week of 2026-12-14 (688, 8.60%); week of 2026-11-30 (682, 8.53%); week of
        2026-12-21 (654, 8.18%).
      Quietest weeks: week of 2026-10-05 (465); week of 2026-10-12 (495); week of 2026-10-19 (561).
      Coverage: Fairly even across weeks. Compare periods with rates or percentages, not raw counts.
      Weekday: Busiest is Thursday (15.12%), quietest is Tuesday (13.04%).
      Time of day: Busiest hour is 07:00 (11.60%); records run from 00:00 to 22:59.

How the columns relate:
  Numbers vs numbers (rank correlation: +1 = rise together, -1 = one rises as the other falls, 0 =
    unrelated):
    total_duration ~ ticket_price: -0.15, very weak or none
  Numbers vs categories (0 = the category tells you nothing about the number, 1 = it explains it
    fully):
    total_duration by airline: 0.66, strong
    ticket_price by travel_class: 0.65, strong
    total_duration by arrival_airport: 0.64, strong
    ticket_price by type_of_flight: 0.61, strong
    ticket_price by airline: 0.58, strong
  Categories vs categories (Cramér's V: 0 = unrelated, 1 = knowing one tells you the other):
    airline ~ travel_class: 0.61, strong
    airline ~ arrival_airport: 0.37, moderate
    travel_class ~ arrival_airport: 0.19, weak
    airline ~ type_of_flight: 0.00, very weak or none
    travel_class ~ type_of_flight: 0.00, very weak or none

Worth a closer look:
  - departure_airport: constant column.
  - ticket_price: 9.8% of values are outliers by the IQR rule.

Cleaning checklist (do these before the analysis):
  1. Drop columns that carry no information: departure_airport.
  2. Open the raw rows behind extreme values before trimming, capping, or keeping them:
     ticket_price.
```

### Reading the report

- **Watch** marks something that is probably wrong or that changes how you should use the column.
- **Note** is useful context, not a problem.
- **Takeaway** is the short advice for that column.
- **Rank correlation** goes from -1 to +1. Positive means two columns rise together, negative means one rises as the other falls, and near 0 means unrelated.
- **Cramér's V** goes from 0 to 1. Near 0 means two categories are unrelated, near 1 means knowing one tells you the other.

### Good to know

- The report is text only, with no charts. Small bar sparklines show shapes and timelines.
- The thresholds are rules of thumb. They point you at what to check, and the decisions stay with you.
- A strong relationship in the report is not proof that one thing causes another.
- On very large tables, relationships and the half-empty-rows check use a random sample (with a fixed seed, so results repeat).

### Project

- Source code and issues: https://github.com/adrianjuliusaluoch/reportpy
- License: MIT
- Author: Adrian Julius Aluoch
