# Regime Identification

A Streamlit prototype for exploring structural breaks in KPI time series from Excel Analytical Base Tables (ABTs).

Upload a workbook, choose a sheet and KPI, set the analysis window and minimum segment length, and run a piecewise-constant and/or piecewise-linear break analysis. The app displays a diagnostic chart and provides CSV downloads of break-count candidates and selected segments.

## Current method

The app uses exact dynamic programming to find the minimum sum-of-squared-errors segmentation for each feasible number of segments. It then uses BIC to select a segment count.

This version does not use PELT, BEAST, or MOB. It does not yet model marketing spend or other independent variables, test channel-effect changes, or identify causal drivers. Treat the results as exploratory KPI break candidates.

## Features

- Upload `.xlsx` or `.xls` workbooks
- Select a workbook sheet, date column, and numeric KPI
- Analyze all available dates or choose a custom window
- Aggregate observations into weeks ending Sunday, or keep the supplied dates
- Compare piecewise-constant and piecewise-linear fits
- Set a minimum number of observations per segment
- View the BIC-selected result, break-count scan, chart, and segment summary
- Download the chart and CSV results

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
