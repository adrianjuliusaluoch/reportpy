# reportpy

Get a plain-language overview of a pandas DataFrame in one line.

Inspired by R's `report` package. It reads your data and tells you, in plain
sentences, what is in each column, what looks wrong, and what to clean first.

## Install

```bash
pip install reportpy
```

## Use

```python
import pandas as pd
import reportpy

df = pd.read_csv("flights.csv")
df.diagnose()
```

You can also import the function directly:

```python
from reportpy import diagnose

diagnose(df, date_cols=["Date"], dayfirst=True)
diagnose(df, brief=True)
text = diagnose(df, return_text=True)
```

## What you get

- A quick summary at the top: rows, columns, and what needs attention
- For each column: type, missing values, spread, and a plain-language reading
- Date coverage: range, gaps, busiest and quietest periods
- Data-quality checks: placeholder codes like 9999, numbers stored as text,
  messy labels, duplicates, impossible date orders
- How columns relate to each other
- A short cleaning checklist

## Example output

```
At a glance:
  - 8,000 rows and 10 columns (2 numeric, 6 categorical, 2 date).
  - Needs attention: 1 column carries no information; 1 column has extreme values or heavy skew.

- ticket_price [numeric]: 8,000 valid values, 0.00% missing
      In plain terms: A typical value is about 17,286. Half of the rows fall between 10,578 and
        22,446; the full range runs from 7,482 to 55,986.
      Takeaway: right-skewed with a long tail of large values; lean on median and percentiles,
        and consider a log transform.
```

## License

MIT