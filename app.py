from __future__ import annotations

from datetime import date
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Regime Identification", layout="wide")
st.title("Regime Identification")
st.caption("Structural-break analysis for time-series KPIs")


def segment_sse(y: np.ndarray, start: int, end: int, model: str) -> tuple[float, float, float]:
    """Return segment SSE and fitted values at the segment's endpoints."""
    values = y[start:end]
    x = np.arange(len(values), dtype=float)

    if model == "piecewise_constant":
        fitted = np.full(len(values), values.mean())
    else:
        # Each segment gets its own linear trend.
        design = np.column_stack([np.ones(len(values)), x])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ coefficients

    sse = float(np.sum((values - fitted) ** 2))
    return sse, float(fitted[0]), float(fitted[-1])


def scan_breaks(
    dates: pd.Series,
    y: np.ndarray,
    model: str,
    min_segment_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the minimum-SSE segmentation for each break count, then select by BIC."""
    n = len(y)
    min_size = int(min_segment_weeks)
    max_segments = n // min_size

    if max_segments < 1:
        raise ValueError(
            f"Need at least {min_size} observations for this minimum segment size; "
            f"the selected window has {n}."
        )

    # Precompute the SSE for every valid segment.
    costs: dict[tuple[int, int], float] = {}
    for start in range(n):
        for end in range(start + min_size, n + 1):
            costs[(start, end)] = segment_sse(y, start, end, model)[0]

    # dp[k, end] = best SSE for the first `end` observations in `k` segments.
    dp = np.full((max_segments + 1, n + 1), np.inf)
    previous = np.full((max_segments + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, max_segments + 1):
        earliest_end = k * min_size
        for end in range(earliest_end, n + 1):
            earliest_start = (k - 1) * min_size
            latest_start = end - min_size
            for start in range(earliest_start, latest_start + 1):
                prior = dp[k - 1, start]
                if not np.isfinite(prior):
                    continue
                candidate = prior + costs[(start, end)]
                if candidate < dp[k, end]:
                    dp[k, end] = candidate
                    previous[k, end] = start

    rows = []
    best_by_segments = {}

    for segments in range(1, max_segments + 1):
        sse = float(dp[segments, n])
        if not np.isfinite(sse):
            continue

        # Segment coefficients plus one residual-variance parameter.
        parameters = segments * (1 if model == "piecewise_constant" else 2) + 1
        safe_sse = max(sse, np.finfo(float).tiny)
        bic = n * np.log(safe_sse / n) + parameters * np.log(n)

        # Backtrack segment boundaries.
        bounds = []
        end = n
        for k in range(segments, 0, -1):
            start = int(previous[k, end])
            if start < 0:
                raise RuntimeError("Could not reconstruct the selected segmentation.")
            bounds.append((start, end))
            end = start
        bounds.reverse()

        break_indices = [start for start, _ in bounds[1:]]
        break_dates = [pd.Timestamp(dates.iloc[i]) for i in break_indices]

        rows.append({
            "model": model,
            "break_count": segments - 1,
            "segment_count": segments,
            "sse": sse,
            "bic": float(bic),
            "break_dates": "; ".join(d.strftime("%Y-%m-%d") for d in break_dates),
            "break_indices": "; ".join(map(str, break_indices)),
        })
        best_by_segments[segments] = (bounds, break_indices, sse, float(bic))

    scan = pd.DataFrame(rows)
    if scan.empty:
        raise ValueError("No valid segmentation was found.")

    selected_row = scan.loc[scan["bic"].idxmin()]
    selected_segments = int(selected_row["segment_count"])
    bounds, break_indices, _, _ = best_by_segments[selected_segments]

    fitted_all = np.full(n, np.nan)
    segment_rows = []

    for segment_number, (start, end) in enumerate(bounds, start=1):
        sse, fit_start, fit_end = segment_sse(y, start, end, model)

        if model == "piecewise_constant":
            fitted_all[start:end] = fit_start
            slope = 0.0
        else:
            fitted_all[start:end] = np.linspace(fit_start, fit_end, end - start)
            slope = (fit_end - fit_start) / max(end - start - 1, 1)

        segment_rows.append({
            "model": model,
            "segment": segment_number,
            "start_date": pd.Timestamp(dates.iloc[start]).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(dates.iloc[end - 1]).strftime("%Y-%m-%d"),
            "weeks": end - start,
            "mean_kpi": float(np.mean(y[start:end])),
            "slope_per_observation": float(slope),
            "fitted_start": fit_start,
            "fitted_end": fit_end,
            "segment_sse": float(sse),
        })

    scan["selected_by_bic"] = scan["segment_count"].eq(selected_segments)
    segments_df = pd.DataFrame(segment_rows)
    return scan, segments_df, fitted_all, break_indices


uploaded = st.file_uploader("Upload an Excel workbook", type=["xlsx", "xls"])

if uploaded is None:
    st.info("Upload a workbook to choose a sheet, KPI, and analysis window.")
    st.stop()

try:
    workbook = pd.ExcelFile(uploaded)
except Exception as exc:
    st.error(f"Could not open this workbook: {exc}")
    st.stop()

sheet_name = st.selectbox("Workbook sheet", workbook.sheet_names)
raw = pd.read_excel(uploaded, sheet_name=sheet_name)

if raw.empty:
    st.error("The selected sheet is empty.")
    st.stop()

# Suggest a date column by parsing columns and checking common names.
date_options = []
parsed_dates = {}
for column in raw.columns:
    parsed = pd.to_datetime(raw[column], errors="coerce")
    if parsed.notna().sum() >= max(3, int(0.5 * len(raw))):
        date_options.append(column)
        parsed_dates[column] = parsed

if not date_options:
    st.error("No date-like column was detected in this sheet.")
    st.stop()

preferred_dates = [
    c for c in date_options
    if str(c).strip().lower() in {"date", "week_start", "week_start_date", "week"}
]
default_date_index = date_options.index(preferred_dates[0]) if preferred_dates else 0
date_column = st.selectbox("Date column", date_options, index=default_date_index)

numeric_options = [
    c for c in raw.columns
    if c != date_column and pd.to_numeric(raw[c], errors="coerce").notna().sum() >= 3
]
if not numeric_options:
    st.error("No numeric KPI columns were detected.")
    st.stop()

kpi_column = st.selectbox("KPI", numeric_options)

prepared = pd.DataFrame({
    "date": parsed_dates[date_column],
    "kpi": pd.to_numeric(raw[kpi_column], errors="coerce"),
}).dropna()

prepared = prepared.sort_values("date").groupby("date", as_index=False)["kpi"].sum()

if prepared.empty:
    st.error("No usable date and KPI rows remain after parsing.")
    st.stop()

min_date = prepared["date"].min().date()
max_date = prepared["date"].max().date()

st.write(f"Detected date range: **{min_date} to {max_date}**")

range_mode = st.radio(
    "Analysis date range",
    ["All available dates", "Custom date range"],
    horizontal=True,
)

if range_mode == "Custom date range":
    selected_range = st.date_input(
        "Choose the start and end dates",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
        st.warning("Choose both a start date and an end date.")
        st.stop()
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

aggregation = st.selectbox(
    "Time aggregation",
    ["Weekly sum (weeks ending Sunday)", "Keep dates as provided"],
    index=0,
)

method_choice = st.selectbox(
    "Structural-break method",
    [
        "Compare piecewise-constant and piecewise-linear (BIC)",
        "Piecewise-constant (BIC)",
        "Piecewise-linear (BIC)",
    ],
)

min_segment_weeks = st.number_input(
    "Minimum observations per segment",
    min_value=2,
    max_value=52,
    value=11,
    step=1,
    help="For weekly data, 11 means each regime must contain at least 11 weeks.",
)

run_clicked = st.button("Run structural-break analysis", type="primary")

if run_clicked:
    selected = prepared[
        prepared["date"].dt.date.between(start_date, end_date)
    ].copy()

    if aggregation == "Weekly sum (weeks ending Sunday)":
        selected = (
            selected.set_index("date")["kpi"]
            .resample("W-SUN")
            .sum(min_count=1)
            .dropna()
            .rename("kpi")
            .reset_index()
        )

    if len(selected) < 2 * int(min_segment_weeks):
        st.error(
            f"The selected window has {len(selected)} observations. "
            f"At least {2 * int(min_segment_weeks)} are needed to allow one break "
            "with the current minimum segment size."
        )
        st.stop()

    models = {
        "Compare piecewise-constant and piecewise-linear (BIC)": [
            "piecewise_constant", "piecewise_linear"
        ],
        "Piecewise-constant (BIC)": ["piecewise_constant"],
        "Piecewise-linear (BIC)": ["piecewise_linear"],
    }[method_choice]

    all_scans = []
    all_segments = []
    fitted_by_model = {}
    breaks_by_model = {}

    try:
        for model in models:
            scan, segments, fitted, breaks = scan_breaks(
                selected["date"].reset_index(drop=True),
                selected["kpi"].to_numpy(dtype=float),
                model,
                int(min_segment_weeks),
            )
            all_scans.append(scan)
            all_segments.append(segments)
            fitted_by_model[model] = fitted
            breaks_by_model[model] = breaks
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.stop()

    scan_output = pd.concat(all_scans, ignore_index=True)
    segment_output = pd.concat(all_segments, ignore_index=True)

    st.subheader("BIC-selected result")
    selected_results = scan_output[scan_output["selected_by_bic"]]
    st.dataframe(selected_results, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(selected["date"], selected["kpi"], color="#172334", lw=2, label="Actual KPI")

    colors = {
        "piecewise_constant": "#E5A12A",
        "piecewise_linear": "#078F8F",
    }
    labels = {
        "piecewise_constant": "Piecewise-constant BIC fit",
        "piecewise_linear": "Piecewise-linear BIC fit",
    }

    for model, fitted in fitted_by_model.items():
        ax.plot(
            selected["date"],
            fitted,
            color=colors[model],
            lw=2.5,
            label=labels[model],
        )
        for index in breaks_by_model[model]:
            ax.axvline(
                selected["date"].iloc[index],
                color=colors[model],
                linestyle="--",
                alpha=0.75,
            )

    ax.set_title(f"{kpi_column}: structural breaks")
    ax.set_ylabel(kpi_column)
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    csv_scan = scan_output.to_csv(index=False).encode("utf-8")
    csv_segments = segment_output.to_csv(index=False).encode("utf-8")

    chart_buffer = BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=180, bbox_inches="tight")
    chart_buffer.seek(0)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download break-count scan CSV",
        csv_scan,
        file_name="break_count_scan.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download selected segments CSV",
        csv_segments,
        file_name="selected_segments.csv",
        mime="text/csv",
    )
    c3.download_button(
        "Download chart PNG",
        chart_buffer,
        file_name="structural_breaks.png",
        mime="image/png",
    )

    st.subheader("All break-count candidates")
    st.dataframe(
        scan_output.sort_values(["model", "bic"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Selected regime segments")
    st.dataframe(segment_output, use_container_width=True, hide_index=True)
