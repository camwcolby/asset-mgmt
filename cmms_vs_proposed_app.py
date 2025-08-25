
# cmms_vs_proposed_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

st.set_page_config(page_title="CMMS vs Proposed PM Program", layout="wide")

# ==========================================================================================
# Helpers
# ==========================================================================================
def read_csv_safely(path: str, **kw) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, encoding="latin1", on_bad_lines="skip", **kw)

def read_excel_safely(path: str, **kw) -> pd.DataFrame:
    return pd.read_excel(path, dtype=str, **kw)

def clean_id(s):
    s = "" if s is None else str(s).strip()
    s = s.replace(" ", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s

def pick_col(df: pd.DataFrame, *cands, required=True):
    if df is None or df.empty:
        if required: raise KeyError(f"Missing any of columns: {cands}")
        return None
    cl = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in cl:
            return cl[c.lower()]
    for c in df.columns:
        lc = c.lower()
        for cand in cands:
            if cand and cand.lower() in lc:
                return c
    if required:
        raise KeyError(f"Missing any of columns: {cands}")
    return None

def to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce strings like '1,234', '$5,678.90', '(123)' to numeric. Leaves NaN where not parseable."""
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) -> -123
    return pd.to_numeric(s, errors="coerce")

# --- Counts-only helpers (no rounding of hours/costs) -------------------------------------
def as_count_series(s: pd.Series) -> pd.Series:
    """Floor to whole-event counts (nullable Int64)."""
    return np.floor(to_numeric_series(s)).astype("Int64")

def floor_counts(df: pd.DataFrame, count_cols: list | tuple) -> pd.DataFrame:
    """
    Return a copy where only the columns listed in count_cols are floored to whole numbers.
    Leave all other columns (hours, costs, etc.) untouched (decimals preserved).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in count_cols:
        if c in out.columns:
            out[c] = as_count_series(out[c])
    return out

def to_numeric(x):  # legacy single-column guard
    return pd.to_numeric(x, errors="coerce")

def coalesce_date(df: pd.DataFrame, candidates) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index)
    for c in candidates:
        col = pick_col(df, c, required=False)
        if col:
            out = out.fillna(pd.to_datetime(df[col], errors="coerce"))
    return out

def ensure_canonical_id(df: pd.DataFrame, *fallbacks) -> bool:
    """Ensure df has a '__canonical_id' column; attempt to derive from fallbacks."""
    if df is None or df.empty:
        return False
    if "__canonical_id" in df.columns:
        df["__canonical_id"] = df["__canonical_id"].astype(str).map(clean_id)
        return True
    fb_list = fallbacks or ("__id_std_norm", "Asset_ID_STD", "MMSD Asset ID", "Asset ID", "AssetID", "asset_id")
    guess = pick_col(df, *fb_list, required=False)
    if guess:
        df["__canonical_id"] = df[guess].astype(str).map(clean_id)
        return True
    return False

def scale_proposed(proposed_df: pd.DataFrame,
                   selected_start: pd.Timestamp,
                   selected_end: pd.Timestamp,
                   full_start: pd.Timestamp,
                   full_end: pd.Timestamp) -> pd.DataFrame:
    days_full = max(1, int((full_end - full_start).days + 1))
    days_sel  = max(0, int((selected_end - selected_start).days + 1))
    factor = min(10.0, max(0.0, days_sel / days_full))  # clamp
    df = proposed_df.copy()

    # Counts: scale then floor to whole events
    if "expected_events" in df.columns:
        df["expected_events_scaled"] = as_count_series(to_numeric_series(df["expected_events"]) * factor)
    else:
        df["expected_events_scaled"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # Totals: scale as floats (no rounding)
    for base in ["exp_labor_hours_total", "exp_labor_cost_total",
                 "exp_material_cost_total", "exp_total_cost"]:
        if base in df.columns:
            df[base + "_scaled"] = to_numeric_series(df[base]) * factor
        else:
            df[base + "_scaled"] = np.nan

    df["scale_factor"] = factor
    return df

def build_id_alias_from(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame(columns=["ID_norm","__canonical_id"])
    id_cols = [c for c in ["__canonical_id","MMSD Asset ID","Asset ID","AssetID","Asset_ID_STD"] if c in df.columns]
    rows = []
    for c in id_cols:
        tmp = df[[c]].copy()
        tmp["ID_norm"] = tmp[c].astype(str).map(clean_id)
        tmp["__canonical_id"] = tmp["ID_norm"]
        rows.append(tmp[["ID_norm","__canonical_id"]])
    if rows:
        alias = pd.concat(rows, ignore_index=True).drop_duplicates()
        return alias
    return pd.DataFrame(columns=["ID_norm","__canonical_id"])

def pick_any(df: pd.DataFrame, *names):
    return next((c for c in names if c in df.columns), None)

# ==========================================================================================
# Sidebar: file pickers & options
# ==========================================================================================
st.sidebar.header("Files & Options")

default_dir = r"C:\Users\cacolby\Desktop\Asset Management Data Analysis\Merge files\JI"

def pick_first_existing(dirpath, names):
    for n in names:
        p = Path(dirpath) / n
        if p.exists():
            return str(p)
    return str(Path(dirpath) / names[0])

aug_master = Path(default_dir).parent / "ALL_projects_with_pm__augmented_with_cmms_only.csv"
base_master = Path(default_dir).parent / "ALL_projects_with_pm.csv"
default_master = str(aug_master if aug_master.exists() else base_master)

default_prop = pick_first_existing(
    default_dir,
    ["proposed_costed_detail_CODELEVEL.csv",
     "proposed_costed_detail_v2.csv",
     "proposed_costed_detail.csv"]
)
default_comp = pick_first_existing(
    default_dir,
    ["asset_cost_comparison_CODELEVEL.csv",
     "asset_cost_comparison_v2.csv",
     "asset_cost_comparison.csv"]
)

prop_path = st.sidebar.text_input("Proposed Detail CSV", value=default_prop)
comp_path = st.sidebar.text_input("Asset Cost Comparison CSV", value=default_comp)
cmms_xlsx_path = st.sidebar.text_input(
    "CMMS Work History Excel",
    value=r"C:\Users\cacolby\Desktop\Asset Management Data Analysis\CMMS Work History_20220101 to 20250707 (1).xlsx"
)
master_path = st.sidebar.text_input(
    "Master Assets CSV (optional; improves Area & exclusions)",
    value=default_master
)

# Closed-loop optional files
cl_detail_path = st.sidebar.text_input(
    "Closed-loop DETAIL CSV (optional)",
    value=str(Path(default_dir) / "closed_loop_detail.csv")
)
cl_summary_path = st.sidebar.text_input(
    "Closed-loop SUMMARY CSV (optional)",
    value=str(Path(default_dir) / "closed_loop_summary.csv")
)

try:
    dd = Path(default_dir)
    if cl_detail_path and not Path(cl_detail_path).exists():
        cand = dd / "closed_loop_program_by_asset_code.csv"
        if cand.exists(): cl_detail_path = str(cand)
    if cl_summary_path and not Path(cl_summary_path).exists():
        cand = dd / "closed_loop_expected_vs_actual_by_asset.csv"
        if cand.exists(): cl_summary_path = str(cand)
except Exception:
    pass

full_start_str = st.sidebar.text_input("Proposed plan full-window start (YYYY-MM-DD)", value="2022-01-01")
full_end_str   = st.sidebar.text_input("Proposed plan full-window end (YYYY-MM-DD)",   value="2025-07-07")
exclude_cmms_only   = st.sidebar.checkbox("Exclude CMMS-only assets (not in master)", value=True)
enable_area_filter  = st.sidebar.checkbox("Enable Area filter", value=True)
scale_proposed_toggle = st.sidebar.checkbox("Scale proposed to selected date range (linear)", value=True)

# NEW: closed-loop control toggles
use_cl_detail_when_available = st.sidebar.checkbox(
    "Use closed-loop DETAIL to compute metrics (preferred)", value=True
)
scale_cl_summary_when_no_detail = st.sidebar.checkbox(
    "Scale closed-loop SUMMARY to selected date range", value=True
)

show_debug = st.sidebar.checkbox("Show debug captions", value=False)

try:
    full_start = pd.to_datetime(full_start_str).normalize()
    full_end   = pd.to_datetime(full_end_str).normalize()
except Exception:
    st.sidebar.error("Invalid full-window dates. Using defaults.")
    full_start = pd.Timestamp("2022-01-01"); full_end = pd.Timestamp("2025-07-07")

# ==========================================================================================
# Load Data
# ==========================================================================================
@st.cache_data(show_spinner=False)
def load_all(prop_path, comp_path, cmms_xlsx_path, master_path, cl_detail_path, cl_summary_path):
    errors = []
    try:
        prop = read_csv_safely(prop_path)
    except Exception as e:
        prop = pd.DataFrame(); errors.append(f"Proposed read error: {e}")
    try:
        comp = read_csv_safely(comp_path)
    except Exception as e:
        comp = pd.DataFrame(); errors.append(f"Comparison read error: {e}")
    try:
        xls = pd.ExcelFile(cmms_xlsx_path)
        sheet = "CMMS Work History" if "CMMS Work History" in xls.sheet_names else xls.sheet_names[0]
        cmms = read_excel_safely(cmms_xlsx_path, sheet_name=sheet)
    except Exception as e:
        cmms = pd.DataFrame(); errors.append(f"CMMS read error: {e}")
    try:
        master = read_csv_safely(master_path)
    except Exception:
        master = pd.DataFrame()
    try:
        cl_detail = read_csv_safely(cl_detail_path) if cl_detail_path else pd.DataFrame()
    except Exception as e:
        cl_detail = pd.DataFrame(); errors.append(f"Closed-loop DETAIL read error: {e}")
    try:
        cl_summary = read_csv_safely(cl_summary_path) if cl_summary_path else pd.DataFrame()
    except Exception as e:
        cl_summary = pd.DataFrame(); errors.append(f"Closed-loop SUMMARY read error: {e}")

    for df in (prop, comp, cmms, master, cl_detail, cl_summary):
        if "__canonical_id" in df.columns:
            df["__canonical_id"] = df["__canonical_id"].astype(str).map(clean_id)

    if "__canonical_id" not in cmms.columns and not cmms.empty:
        h_id = pick_col(cmms, "MMSD Asset ID","Asset ID","AssetID", required=False)
        if h_id:
            cmms["__canonical_id"] = cmms[h_id].astype(str).map(clean_id)

    area_col = "Area" if ("Area" in comp.columns) else ("Area" if "Area" in master.columns else None)

    return prop, comp, cmms, master, cl_detail, cl_summary, errors, area_col

prop_df, comp_df, cmms_raw, master_df, cl_detail_df, cl_summary_df, load_errors, area_col_hint = load_all(
    prop_path, comp_path, cmms_xlsx_path, master_path, cl_detail_path, cl_summary_path
)
for e in load_errors: st.warning(e)
if comp_df.empty or prop_df.empty or cmms_raw.empty:
    st.error("One or more required files did not load. Check paths on the left.")
    st.stop()

# ==========================================================================================
# Build CMMS baseline with date & Work Type filters
# ==========================================================================================
cmms = cmms_raw.copy()
cmms["__date"] = coalesce_date(cmms, ["Actual Finish Date","Finish Date","Closed Date","Date Completed"])
wt_col = pick_col(cmms, "Work Type","WorkType", required=False)
cmms["__work_type"] = cmms[wt_col].astype(str).str.strip() if wt_col else "PM"

min_date = pd.to_datetime("2000-01-01") if cmms["__date"].isna().all() else pd.to_datetime(cmms["__date"].min())
max_date = pd.to_datetime("today") if cmms["__date"].isna().all() else pd.to_datetime(cmms["__date"].max())

st.sidebar.markdown("---"); st.sidebar.subheader("Filters")
date_from = st.sidebar.date_input("From date", value=min_date.date() if pd.notna(min_date) else date(2022,1,1))
date_to   = st.sidebar.date_input("To date",   value=max_date.date() if pd.notna(max_date) else date(2025,7,7))

work_types_all = sorted([w for w in cmms["__work_type"].dropna().unique().tolist() if w != ""])
default_wt = [w for w in work_types_all if str(w).upper() in {"PM","PREVENTIVE MAINTENANCE"}]
if not default_wt and "PM" in work_types_all: default_wt = ["PM"]
work_types_sel = st.sidebar.multiselect("Work Type(s)", options=work_types_all,
                                        default=default_wt if default_wt else work_types_all[:5])

cmms_f = cmms.copy()
start_sel = pd.Timestamp(date_from); end_sel = pd.Timestamp(date_to)
if cmms_f["__date"].notna().any():
    cmms_f = cmms_f[cmms_f["__date"].between(start_sel, end_sel, inclusive="both")].copy()
if work_types_sel:
    cmms_f = cmms_f[cmms_f["__work_type"].isin(work_types_sel)].copy()

# Numeric baselines
lh = pick_col(cmms_f, "Labor Hours","Total Labor Hours", required=False)
lc = pick_col(cmms_f, "Labor Cost","Total Labor Cost", required=False)
mc = pick_col(cmms_f, "Materials Cost","Material Cost","Parts Cost","Parts/Materials", required=False)
tc = pick_col(cmms_f, "Total Task Cost","Total Cost","Cost", required=False)
for name, col in {"labor_hours": lh, "labor_cost": lc, "material_cost": mc, "total_cost": tc}.items():
    cmms_f[name] = to_numeric_series(cmms_f[col]) if col else np.nan

# Canonical ID
if "__canonical_id" not in cmms_f.columns:
    hid = pick_col(cmms_f, "MMSD Asset ID","Asset ID","AssetID", required=False)
    cmms_f["__canonical_id"] = cmms_f[hid].astype(str).map(clean_id) if hid else pd.NA
cmms_f["__canonical_id"] = cmms_f["__canonical_id"].astype(str).map(clean_id)

# Baseline aggregates
base_asset = (cmms_f.groupby("__canonical_id", dropna=False)
              .agg(labor_hours_sum=("labor_hours","sum"),
                   labor_cost_sum=("labor_cost","sum"),
                   materials_cost_sum=("material_cost","sum"),
                   total_task_cost_sum=("total_cost","sum"),
                   actual_pm_count=("__work_type","count"))
              .reset_index())

# ==========================================================================================
# Proposed detail (CODELEVEL or ACTIVITY-LEVEL agnostic)
# ==========================================================================================
prop_detail = prop_df.copy()
for k in ["expected_events","exp_labor_hours_total","exp_labor_cost_total",
          "exp_material_cost_total","exp_total_cost"]:
    if k in prop_detail.columns:
        prop_detail[k] = to_numeric_series(prop_detail[k])
prop_detail["__canonical_id"] = prop_detail["__canonical_id"].astype(str).map(clean_id)
prop_detail["Work Type"] = "PM"

if scale_proposed_toggle:
    prop_scaled = scale_proposed(prop_detail, start_sel, end_sel, full_start, full_end)
else:
    prop_scaled = prop_detail.copy()
    # floor expected events; keep totals as floats (no rounding)
    if "expected_events" in prop_scaled.columns:
        prop_scaled["expected_events"] = as_count_series(prop_scaled["expected_events"])
    # mirror unscaled totals into *_scaled columns so groupby below works uniformly
    prop_scaled["expected_events_scaled"] = prop_scaled.get("expected_events")
    for base in ["exp_labor_hours_total","exp_labor_cost_total","exp_material_cost_total","exp_total_cost"]:
        if base in prop_scaled.columns:
            prop_scaled[base + "_scaled"] = to_numeric_series(prop_scaled[base])

prop_asset = (prop_scaled.groupby("__canonical_id", dropna=False)
              .agg(expected_events_sum=("expected_events_scaled","sum"),
                   expected_labor_hours_sum=("exp_labor_hours_total_scaled","sum"),
                   expected_labor_cost_sum=("exp_labor_cost_total_scaled","sum"),
                   expected_material_cost_sum=("exp_material_cost_total_scaled","sum"),
                   expected_total_cost_sum=("exp_total_cost_scaled","sum"))
              .reset_index())

# make sums whole numbers ONLY for counts; leave hours/costs as floats
if "expected_events_sum" in prop_asset.columns:
    prop_asset["expected_events_sum"] = as_count_series(prop_asset["expected_events_sum"])

# ==========================================================================================
# Comparison base: scaffold + fresh aggregates
# ==========================================================================================
compare = comp_df.copy()
compare["__canonical_id"] = compare["__canonical_id"].astype(str).map(clean_id)

compare = (compare.drop(columns=[
            c for c in ["expected_events_sum","expected_labor_hours_sum","expected_labor_cost_sum",
                        "expected_material_cost_sum","expected_total_cost_sum",
                        "labor_hours_sum","total_task_cost_sum","materials_cost_sum",
                        "labor_cost_sum","actual_pm_count"] 
            if c in compare.columns
          ], errors="ignore")
          .merge(base_asset, on="__canonical_id", how="left")
          .merge(prop_asset, on="__canonical_id", how="left"))

# --- ADD THIS: include CMMS-only assets into the comparison scaffold ---
# Ensure a __source col exists (so we can mark CMMS stubs)
if "__source" not in compare.columns:
    compare["__source"] = ""

# Find asset IDs that exist in CMMS baseline but not in compare
base_ids = set(base_asset["__canonical_id"].dropna().astype(str))
comp_ids = set(compare["__canonical_id"].dropna().astype(str))
missing_ids = sorted(base_ids - comp_ids)

if missing_ids:
    # Start from CMMS baseline rows for those missing IDs
    cmms_only_rows = base_asset[base_asset["__canonical_id"].isin(missing_ids)].copy()
    cmms_only_rows["__source"] = "CMMS_ONLY_STUB"

    # (optional) bring Area/dims from master if available
    if not master_df.empty:
        key = "Asset_ID_STD" if "Asset_ID_STD" in master_df.columns else \
              pick_col(master_df, "MMSD Asset ID", "Asset ID", "AssetID", required=False)
        if key:
            dims = master_df[[key] + [c for c in ["Area","Asset Class","Asset Type"] if c in master_df.columns]].copy()
            dims["__canonical_id"] = dims[key].astype(str).map(clean_id)
            cmms_only_rows = cmms_only_rows.merge(
                dims.drop(columns=[key]),
                on="__canonical_id", how="left"
            )

    # Append to compare (so toggle can include/exclude them)
    compare = pd.concat([compare, cmms_only_rows], ignore_index=True, sort=False)
# --- END ADD ---

# Gaps (hours/costs float; counts as Int)
compare["labor_hours_gap_total"] = compare["expected_labor_hours_sum"].fillna(0) - compare["labor_hours_sum"].fillna(0)
compare["cost_gap_total"]        = compare["expected_total_cost_sum"].fillna(0) - compare["total_task_cost_sum"].fillna(0)

# floor/clean counts only
if "expected_events_sum" in compare.columns:
    compare["expected_events_sum"] = as_count_series(compare["expected_events_sum"])
if "actual_pm_count" in compare.columns:
    compare["actual_pm_count"] = as_count_series(compare["actual_pm_count"])

# PM count gap as Int64
compare["count_gap_total"] = (compare.get("expected_events_sum", 0).fillna(0).astype("Int64")
                              - compare.get("actual_pm_count", 0).fillna(0).astype("Int64"))

# ==========================================================================================
# Closed-loop merges (OPTIONAL, robust & future-proof) — now filter-aware
# ==========================================================================================
cl_count_col_guess = ("actual_pm_count","closed_loop_count","count","pm_count")
cl_hours_col_guess = ("labor_hours_sum","closed_loop_hours","hours","exp_labor_hours_total","labor_hours")
cl_cost_col_guess  = ("total_task_cost_sum","closed_loop_cost","cost","exp_total_cost","total_cost")

cl_detail_rows_filt = pd.DataFrame()
cl_detail_f = pd.DataFrame()
cl_sum_used = pd.DataFrame()

if not cl_detail_df.empty and use_cl_detail_when_available:
    cl_detail = cl_detail_df.copy()
    ensure_canonical_id(cl_detail, "__id_std_norm","Asset_ID_STD","MMSD Asset ID","Asset ID","AssetID","asset_id")

    wo_date_col = pick_any(cl_detail, "wo_date","work_order_date","date")
    if wo_date_col:
        cl_detail[wo_date_col] = pd.to_datetime(cl_detail[wo_date_col], errors="coerce")
        cl_detail_rows_filt = cl_detail[cl_detail[wo_date_col].between(start_sel, end_sel, inclusive="both")].copy()
    else:
        cl_detail_rows_filt = cl_detail.copy()

    wt_cand = pick_any(cl_detail_rows_filt, "__work_type","work_type","WT","wt")
    if wt_cand and work_types_sel:
        cl_detail_rows_filt[wt_cand] = cl_detail_rows_filt[wt_cand].astype(str).str.strip()
        cl_detail_rows_filt = cl_detail_rows_filt[cl_detail_rows_filt[wt_cand].isin(work_types_sel)]

    ids_filter = None
    if enable_area_filter:
        if "Area" in compare.columns:
            ids_filter = set(compare["__canonical_id"].dropna().unique().tolist())
    if ids_filter:
        cl_detail_rows_filt = cl_detail_rows_filt[cl_detail_rows_filt["__canonical_id"].isin(ids_filter)]

    hrs_col  = pick_any(cl_detail_rows_filt, *cl_hours_col_guess)
    cost_col = pick_any(cl_detail_rows_filt, *cl_cost_col_guess)
    if hrs_col:  cl_detail_rows_filt["__cl_hours"] = to_numeric_series(cl_detail_rows_filt[hrs_col])
    else:        cl_detail_rows_filt["__cl_hours"] = 0.0
    if cost_col: cl_detail_rows_filt["__cl_cost"]  = to_numeric_series(cl_detail_rows_filt[cost_col])
    else:        cl_detail_rows_filt["__cl_cost"]  = 0.0

    if "matched" in cl_detail_rows_filt.columns:
        cl_detail_rows_filt["__cl_count_unit"] = cl_detail_rows_filt["matched"].astype(str).str.upper().isin(
            ["TRUE","1","YES","Y"]).astype(int)
    else:
        cl_detail_rows_filt["__cl_count_unit"] = 1

    cl_detail_f = (cl_detail_rows_filt
        .groupby("__canonical_id", dropna=False)
        .agg(cl_count=("__cl_count_unit", "sum"),
             cl_hours=("__cl_hours", "sum"),
             cl_cost =("__cl_cost",  "sum"))
        .reset_index())

    cl_sum_used = cl_detail_f.copy()

else:
    if not cl_summary_df.empty:
        cl_sum = cl_summary_df.copy()
        has_id = ensure_canonical_id(cl_sum, "__id_std_norm","Asset_ID_STD","MMSD Asset ID","Asset ID","AssetID","asset_id")
        if not has_id:
            st.warning("Closed-loop SUMMARY: couldn’t find any ID column to build __canonical_id.")
        else:
            g = pick_any(cl_sum, *cl_count_col_guess)
            if g and g != "cl_count": cl_sum.rename(columns={g:"cl_count"}, inplace=True)
            g = pick_any(cl_sum, *cl_hours_col_guess)
            if g and g != "cl_hours": cl_sum.rename(columns={g:"cl_hours"}, inplace=True)
            g = pick_any(cl_sum, *cl_cost_col_guess)
            if g and g != "cl_cost":  cl_sum.rename(columns={g:"cl_cost"},  inplace=True)

            for c in ["cl_count","cl_hours","cl_cost"]:
                if c in cl_sum.columns: cl_sum[c] = to_numeric_series(cl_sum[c])

            if enable_area_filter and "Area" in compare.columns:
                ids_filter = set(compare["__canonical_id"].dropna().unique().tolist())
                cl_sum = cl_sum[cl_sum["__canonical_id"].isin(ids_filter)]

            if scale_cl_summary_when_no_detail:
                days_full = max(1, int((full_end - full_start).days + 1))
                days_sel  = max(0, int((end_sel - start_sel).days + 1))
                factor = min(10.0, max(0.0, days_sel / days_full))
                for c in ["cl_count","cl_hours","cl_cost"]:
                    if c in cl_sum.columns:
                        cl_sum[c] = cl_sum[c] * factor
                cl_sum["cl_scale_factor"] = factor

            keep = ["__canonical_id"] + [c for c in ["cl_count","cl_hours","cl_cost","cl_scale_factor"] if c in cl_sum.columns]
            cl_sum_used = cl_sum[keep].drop_duplicates("__canonical_id")
    else:
        cl_sum_used = pd.DataFrame(columns=["__canonical_id","cl_count","cl_hours","cl_cost"])

# Merge the (filtered) closed-loop metrics into compare
if not cl_sum_used.empty:
    compare = compare.merge(cl_sum_used, on="__canonical_id", how="left")

# Ensure closed-loop numbers: floor counts; leave hours/costs as floats
if "cl_count" in compare.columns:
    compare["cl_count"] = as_count_series(compare["cl_count"])

# Execution rate uses floored counts
if "cl_count" in compare.columns:
    denom = to_numeric_series(compare["expected_events_sum"]).fillna(0)
    num   = to_numeric_series(compare["cl_count"]).fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        compare["execution_rate"] = np.where(denom > 0, num / denom, np.nan)

# Debug previews (unchanged)
show_debug = show_debug
if show_debug:
    if not cl_summary_df.empty:
        st.caption(f"CL summary columns (raw): {list(cl_summary_df.columns)}")
        st.dataframe(cl_summary_df.head(10), use_container_width=True, height=220)
    if not cl_sum_used.empty:
        st.caption(f"CL metrics used (post-filter/scale): {list(cl_sum_used.columns)}")
        st.dataframe(cl_sum_used.head(10), use_container_width=True, height=220)
    if not cl_detail_df.empty:
        st.caption(f"CL detail columns (raw): {list(cl_detail_df.columns)}")
        st.dataframe(cl_detail_df.head(10), use_container_width=True, height=220)
    if not cl_detail_rows_filt.empty:
        st.caption("CL detail rows (filtered to current window/work type/area)")
        st.dataframe(cl_detail_rows_filt.head(10), use_container_width=True, height=220)

# ==========================================================================================
# Optional: Exclude CMMS-only assets (not in master)
# ==========================================================================================
if exclude_cmms_only:
    # If comparison already has a __source flag, just drop those rows
    if "__source" in compare.columns:
        compare = compare[compare["__source"].astype(str).str.upper().ne("CMMS_ONLY_STUB")].copy()
    elif not master_df.empty:
        # Fallback: build a set of master IDs and keep only those
        id_cols = [c for c in ["Asset_ID_STD","MMSD Asset ID","Asset ID","AssetID"] if c in master_df.columns]
        ids = set()
        for c in id_cols:
            ids.update(master_df[c].astype(str).map(clean_id).unique().tolist())
        compare = compare[compare["__canonical_id"].isin(ids)].copy()

# Always keep the other frames aligned to whatever IDs are left in compare
ids_sel = set(compare["__canonical_id"].dropna().astype(str).unique().tolist())
if not prop_scaled.empty and "__canonical_id" in prop_scaled.columns:
    prop_scaled = prop_scaled[prop_scaled["__canonical_id"].astype(str).isin(ids_sel)]
if not cmms_f.empty and "__canonical_id" in cmms_f.columns:
    cmms_f = cmms_f[cmms_f["__canonical_id"].astype(str).isin(ids_sel)]
if 'cl_detail_rows_filt' in locals() and not cl_detail_rows_filt.empty and "__canonical_id" in cl_detail_rows_filt.columns:
    cl_detail_rows_filt = cl_detail_rows_filt[cl_detail_rows_filt["__canonical_id"].astype(str).isin(ids_sel)]
if 'cl_sum_used' in locals() and not cl_sum_used.empty and "__canonical_id" in cl_sum_used.columns:
    cl_sum_used = cl_sum_used[cl_sum_used["__canonical_id"].astype(str).isin(ids_sel)]

# ==========================================================================================
# Area filter (if present)
# ==========================================================================================
if enable_area_filter:
    if "Area" in compare.columns:
        areas = sorted([a for a in compare["Area"].dropna().unique().tolist() if a != ""])
        sel = st.multiselect("Filter by Area", options=areas, default=areas[:5] if areas else [])
        if sel:
            compare = compare[compare["Area"].isin(sel)].copy()
            ids_sel = compare["__canonical_id"].dropna().unique()
            prop_scaled = prop_scaled[prop_scaled["__canonical_id"].isin(ids_sel)]
            cmms_f = cmms_f[cmms_f["__canonical_id"].isin(ids_sel)]
            if not cl_detail_rows_filt.empty:
                cl_detail_rows_filt = cl_detail_rows_filt[cl_detail_rows_filt["__canonical_id"].isin(ids_sel)]
            if not cl_sum_used.empty:
                cl_sum_used = cl_sum_used[cl_sum_used["__canonical_id"].isin(ids_sel)]
    elif "Area" in master_df.columns:
        dims = master_df.copy()
        key = "Asset_ID_STD" if "Asset_ID_STD" in dims.columns else pick_col(dims, "MMSD Asset ID","Asset ID","AssetID")
        dims["__canonical_id"] = dims[key].astype(str).map(clean_id)
        compare = compare.merge(dims[["__canonical_id","Area"]], on="__canonical_id", how="left")
        areas = sorted([a for a in compare["Area"].dropna().unique().tolist() if a != ""])
        sel = st.multiselect("Filter by Area", options=areas, default=areas[:5] if areas else [])
        if sel:
            compare = compare[compare["Area"].isin(sel)].copy()
            ids_sel = compare["__canonical_id"].dropna().unique()
            prop_scaled = prop_scaled[prop_scaled["__canonical_id"].isin(ids_sel)]
            cmms_f = cmms_f[cmms_f["__canonical_id"].isin(ids_sel)]
            if not cl_detail_rows_filt.empty:
                cl_detail_rows_filt = cl_detail_rows_filt[cl_detail_rows_filt["__canonical_id"].isin(ids_sel)]
            if not cl_sum_used.empty:
                cl_sum_used = cl_sum_used[cl_sum_used["__canonical_id"].isin(ids_sel)]

# ==========================================================================================
# Metrics (counts, hours, costs, closed-loop exec)
# ==========================================================================================
base_cost   = to_numeric_series(compare['total_task_cost_sum']).fillna(0).sum()
prop_cost   = to_numeric_series(compare['expected_total_cost_sum']).fillna(0).sum()
base_hours  = to_numeric_series(compare['labor_hours_sum']).fillna(0).sum()
prop_hours  = to_numeric_series(compare['expected_labor_hours_sum']).fillna(0).sum()
base_count  = to_numeric_series(compare['actual_pm_count']).fillna(0).sum()
prop_count  = to_numeric_series(compare['expected_events_sum']).fillna(0).sum()

r1c1, r1c2, r1c3 = st.columns(3)
r1c1.metric("Baseline cost (filtered)", f"${base_cost:,.2f}")
r1c2.metric("Proposed cost (filtered)", f"${prop_cost:,.2f}")
r1c3.metric("Δ Cost (Proposed - Baseline)", f"${(prop_cost - base_cost):,.2f}")

r2c1, r2c2, r2c3 = st.columns(3)
r2c1.metric("Baseline hours (filtered)", f"{base_hours:,.2f}")
r2c2.metric("Proposed hours (filtered)", f"{prop_hours:,.2f}")
r2c3.metric("Δ Hours (Proposed - Baseline)", f"{(prop_hours - base_hours):,.2f}")

r3c1, r3c2, r3c3 = st.columns(3)
r3c1.metric("Baseline count (filtered)", f"{int(base_count):,}")
r3c2.metric("Proposed count (filtered)", f"{int(prop_count):,}")
r3c3.metric("Δ Count (Proposed - Baseline)", f"{int(prop_count - base_count):,}")

if "execution_rate" in compare.columns:
    cl_count = to_numeric_series(compare.get("cl_count")).fillna(0).sum()
    cl_hours = to_numeric_series(compare.get("cl_hours")).fillna(0).sum()
    cl_cost  = to_numeric_series(compare.get("cl_cost")).fillna(0).sum()
    exec_rate_pct = (cl_count / prop_count * 100.0) if prop_count and prop_count > 0 else np.nan
    r4c1, r4c2, r4c3, r4c4 = st.columns(4)
    r4c1.metric("Closed-loop count (filtered)", f"{int(cl_count):,}")
    r4c2.metric("Closed-loop hours (filtered)", f"{cl_hours:,.2f}")
    r4c3.metric("Closed-loop cost (filtered)", f"${cl_cost:,.2f}")
    r4c4.metric("Execution rate (CL / Proposed)", f"{(exec_rate_pct if pd.notna(exec_rate_pct) else 0):.1f}%")

# ==========================================================================================
# Tabs
# ==========================================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Compare (Assets)", "Proposed Detail", "CMMS Detail", "Closed-loop Detail", "Downloads"
])

with tab1:
    # choose the count key (scaled or not)
    count_key = "expected_events_sum_scaled" if st.session_state.get("scale_enabled") else "expected_events_sum"

    # ensure we have a local view; floor counts for display if used
    comp_view = compare.copy()
    comp_view = floor_counts(comp_view, [count_key, "actual_pm_count"])

    if "pm_gap_total" in comp_view.columns:
        comp_view["pm_gap_total"] = comp_view[count_key] - comp_view["actual_pm_count"]

    st.subheader("Asset-level comparison (filtered)")
    front = [c for c in [
        "__canonical_id","Area","Asset Description","Asset Class","Asset Type",
        "expected_events_sum","actual_pm_count","count_gap_total",
        "expected_total_cost_sum","total_task_cost_sum","cost_gap_total",
        "expected_labor_hours_sum","labor_hours_sum","labor_hours_gap_total",
        "cl_count","cl_hours","cl_cost","execution_rate"
    ] if c in compare.columns]
    st.dataframe(compare[front + [c for c in compare.columns if c not in front]],
                 use_container_width=True, height=520)

with tab2:
    evt_key = "expected_events_scaled" if st.session_state.get("scale_enabled") else "expected_events"
    df = prop_scaled.copy()
    df = floor_counts(df, [evt_key])

    st.subheader("Proposed detail (scaled to date range)" if scale_proposed_toggle else "Proposed detail")
    label_col = "labels_used" if "labels_used" in prop_scaled.columns else (
        "mapped_activity_norm" if "mapped_activity_norm" in prop_scaled.columns else None
    )
    if scale_proposed_toggle:
        cols = ["__canonical_id","freq_code"]
        if label_col: cols.append(label_col)
        cols += ["expected_events_scaled",
                 "exp_labor_hours_total_scaled","exp_labor_cost_total_scaled",
                 "exp_material_cost_total_scaled","exp_total_cost_scaled",
                 "unit_cost_source","scale_factor"]
    else:
        cols = ["__canonical_id","freq_code"]
        if label_col: cols.append(label_col)
        cols += ["expected_events",
                 "exp_labor_hours_total","exp_labor_cost_total",
                 "exp_material_cost_total","exp_total_cost",
                 "unit_cost_source"]
    cols = [c for c in cols if c in prop_scaled.columns]
    st.dataframe(prop_scaled[cols], use_container_width=True, height=520)

with tab3:
    st.subheader("CMMS (filtered by date & Work Type)")
    show_cols = ["__date","__work_type","__canonical_id"]

    # Prepare display frame (no rounding for hours/costs)
    cmms_display = cmms_f.copy()
    for c in ["Labor Hours","Total Labor Hours","Labor Cost","Total Labor Cost",
              "Materials Cost","Material Cost","Parts Cost","Parts/Materials",
              "Total Task Cost","Total Cost","Cost"]:
        if c in cmms_display.columns:
            cmms_display[c] = to_numeric_series(cmms_display[c])

    for c in ["Labor Hours","Total Labor Hours","Labor Cost","Total Labor Cost","Materials Cost","Material Cost",
              "Total Task Cost","Total Cost","Cost"]:
        if c in cmms_display.columns: show_cols.append(c)

    extra = [c for c in cmms_display.columns if c not in show_cols][:6]
    show_cols += extra

    st.dataframe(cmms_display[show_cols], use_container_width=True, height=520)

with tab4:
    evt_key = "expected_events_scaled" if st.session_state.get("scale_enabled") else "expected_events"
    df = prop_scaled.copy()
    df = floor_counts(df, [evt_key])

    st.subheader("Closed-loop DETAIL" + (" (filtered)" if not cl_detail_rows_filt.empty else ""))
    if not cl_detail_df.empty and not cl_detail_rows_filt.empty:
        show = cl_detail_rows_filt.copy()
        # keep hours/costs numeric; only counts coerced if present
        for raw in ["labor_hours","total_cost","__cl_hours","__cl_cost"]:
            if raw in show.columns:
                show[raw] = to_numeric_series(show[raw])
        if "__cl_count_unit" in show.columns:
            show["__cl_count_unit"] = as_count_series(show["__cl_count_unit"])

        preferred = ["__canonical_id","freq_code","wo_id","wo_date","work_order_date","date",
                     "matched","labor_hours","total_cost","__cl_hours","__cl_cost","__cl_count_unit"]
        front = [c for c in preferred if c in show.columns]
        st.dataframe(show[front + [c for c in show.columns if c not in front]],
                     use_container_width=True, height=520)

with tab5:
    st.subheader("Download filtered outputs")
    st.download_button("Download comparison (CSV)",
        data=compare.to_csv(index=False).encode("utf-8-sig"),
        file_name="asset_cost_comparison_filtered.csv", mime="text/csv")
    st.download_button("Download proposed detail (CSV)",
        data=prop_scaled.to_csv(index=False).encode("utf-8-sig"),
        file_name="proposed_costed_detail_filtered.csv", mime="text/csv")
    st.download_button("Download CMMS detail (CSV)",
        data=cmms_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="cmms_filtered.csv", mime="text/csv")
    if not cl_detail_df.empty:
        data_dl = (cl_detail_rows_filt if not cl_detail_rows_filt.empty else cl_detail_df)
        st.download_button("Download closed-loop DETAIL (CSV)",
            data=data_dl.to_csv(index=False).encode("utf-8-sig"),
            file_name="closed_loop_detail_filtered.csv", mime="text/csv")
    if not cl_sum_used.empty:
        st.download_button("Download closed-loop SUMMARY (CSV)",
            data=cl_sum_used.to_csv(index=False).encode("utf-8-sig"),
            file_name="closed_loop_summary_filtered.csv", mime="text/csv")

with st.expander("Notes on filtering, scaling & closed-loop", expanded=False):
    st.markdown(f"""
- **Date range & Work Type filters** apply directly to the CMMS baseline (from Excel).
- For **Proposed**, if “Scale proposed to selected date range” is ON, totals are proportionally scaled
  from the original window (**{full_start.date()} → {full_end.date()}**).
- **Closed-loop**:
  - If a DETAIL file is provided and **Use closed-loop DETAIL** is ON, closed-loop counts/hours/costs are computed **within the selected date/work-type/area filters**.
  - If only SUMMARY is available, you can optionally scale it linearly to the selected window.
  - **Execution rate** = `cl_count / expected_events_sum` (Proposed uses the scaled totals if scaling is ON).
- Column names are auto-mapped and all math is **type-safe** (strings, commas, and currency handled).
""")







