import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import threading

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
DATA_FORECAST_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_trained_data" / "qn_data_forecast.csv"

cache_lock = threading.RLock()

def quarter_to_index(year, quarter):
    if isinstance(year, pd.Series):
        return year.astype(int) * 4 + quarter.astype(int)
    return int(year) * 4 + int(quarter)

def index_to_quarter(idx):
    year = (idx - 1) // 4
    quarter = (idx - 1) % 4 + 1
    return int(year), int(quarter)

def load_base_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Base data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df["Quarter"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["year"] = df["Date"].dt.year
    df["quarter"] = df["Date"].dt.quarter
    return df

def load_forecast_cache():
    with cache_lock:
        if not DATA_FORECAST_PATH.exists():
            return pd.DataFrame()
        df = pd.read_csv(DATA_FORECAST_PATH)
        if "Quarter" in df.columns:
            if "quarter" not in df.columns:
                df["quarter"] = df["Quarter"]
            else:
                df["quarter"] = df["quarter"].fillna(df["Quarter"])
            df = df.drop(columns=["Quarter"])
            
        if not df.empty and "species" not in df.columns:
            df["species"] = "ALL"
            
        # Drop rows where year or quarter is NaN (dirty data protection)
        df = df.dropna(subset=["year", "quarter"])
        return df

def save_forecast_cache(df_new):
    with cache_lock:
        if df_new.empty:
            return
        df_cache = load_forecast_cache()
        if not df_cache.empty:
            df_combined = pd.concat([df_cache, df_new], ignore_index=True)
        else:
            df_combined = df_new.copy()
    
        if "species" not in df_combined.columns:
            df_combined["species"] = "ALL"
        else:
            df_combined["species"] = df_combined["species"].fillna("ALL")
    
        df_combined = df_combined.drop_duplicates(
            subset=["X", "Y", "year", "quarter", "species"], 
            keep="last"
        )
        
        DATA_FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(DATA_FORECAST_PATH, index=False)
        logger.info("Saved forecast cache")

def get_latest_available_time(df_base, df_cache, x, y, species=None):
    df_base_station = df_base[(df_base["X"] == x) & (df_base["Y"] == y)]
    max_idx = -1
    
    if not df_base_station.empty:
        base_indices = quarter_to_index(df_base_station["year"], df_base_station["quarter"])
        max_idx = max(max_idx, base_indices.max())
        
    if not df_cache.empty:
        cond = (df_cache["X"] == x) & (df_cache["Y"] == y)
        if species:
            cond &= (df_cache["species"] == species)
        else:
            cond &= (df_cache["species"] == "ALL")
            
        df_cache_station = df_cache[cond]
        if not df_cache_station.empty:
            cache_indices = quarter_to_index(df_cache_station["year"], df_cache_station["quarter"])
            max_idx = max(max_idx, cache_indices.max())
            
    if max_idx == -1:
        raise ValueError(f"❌ Không tìm thấy dữ liệu lịch sử cho trạm: {x}, {y}")
        
    year, quarter = index_to_quarter(max_idx)
    logger.info(f"Latest available data for station ({x}, {y}) species={species or 'ALL'}: {year}-Q{quarter}")
    return year, quarter

def _build_history_df(df_base, df_cache, x, y, species=None):
    df_base_station = df_base[(df_base["X"] == x) & (df_base["Y"] == y)].copy()
    
    df_cache_station = pd.DataFrame()
    if not df_cache.empty:
        cond = (df_cache["X"] == x) & (df_cache["Y"] == y)
        if species:
            cond &= (df_cache["species"] == species)
        else:
            cond &= (df_cache["species"] == "ALL")
        df_cache_station = df_cache[cond].copy()
        
    df_combined = pd.concat([df_base_station, df_cache_station], ignore_index=True)
    if not df_combined.empty:
        df_combined["idx"] = quarter_to_index(df_combined["year"], df_combined["quarter"])
        df_combined = df_combined.sort_values("idx").drop_duplicates(subset=["idx"], keep="last")
    return df_combined

def generate_missing_metal_forecasts(x, y, requested_idx):
    df_base = load_base_data()
    df_cache = load_forecast_cache()
    
    latest_year, latest_quarter = get_latest_available_time(df_base, df_cache, x, y, species="ALL")
    latest_idx = quarter_to_index(latest_year, latest_quarter)
    
    if latest_idx >= requested_idx:
        return 
        
    logger.info(f"Missing metal forecast range: {latest_year}-Q{latest_quarter} -> index {requested_idx}")
    
    model_path = PROJECT_DIR / "model" / "output" / "metal_ts_model.pkl"
    model, feature_cols = joblib.load(model_path)
    target_cols = ["CN","As","Cd","Pb","Cu","Hg","Zn","Total_Cr"]
    
    # Lookup Station / Station_Name from base data once
    station_meta = df_base[df_base["X"] == x][["X","Y","Station","Station_Name"]].drop_duplicates(subset=["X","Y"])
    station_val = station_meta.iloc[0]["Station"] if not station_meta.empty else None
    station_name_val = station_meta.iloc[0]["Station_Name"] if not station_meta.empty else None
    
    df_history = _build_history_df(df_base, df_cache, x, y, species="ALL")
    new_results = []
    
    for curr_idx in range(latest_idx + 1, requested_idx + 1):
        curr_year, curr_quarter = index_to_quarter(curr_idx)
        logger.info(f"Forecasting metal quarter: {curr_year}-Q{curr_quarter}")
        
        idx_lag1 = curr_idx - 1
        idx_lag4 = curr_idx - 4
        
        lag1_row = df_history[df_history["idx"] == idx_lag1]
        lag4_row = df_history[df_history["idx"] == idx_lag4]
        
        if lag1_row.empty or lag4_row.empty:
             raise ValueError(f"Không đủ dữ liệu lịch sử lag1/lag4 tại {curr_year}-Q{curr_quarter}")
             
        lag1_row = lag1_row.iloc[-1]
        lag4_row = lag4_row.iloc[-1]
        
        row = {}
        for c in target_cols:
            row[f"{c}_lag1"] = float(lag1_row.get(c, 0))
            row[f"{c}_lag4"] = float(lag4_row.get(c, 0))
            
        row["year"] = int(curr_year)
        row["Quarter"] = int(curr_quarter)
        row["quarter"] = int(curr_quarter)
        
        X_pred = pd.DataFrame([row])[feature_cols].astype(float)
        y_pred = model.predict(X_pred)[0]
        y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
        
        result = {
            "Station": station_val,
            "Station_Name": station_name_val,
            "X": x,
            "Y": y,
            "year": int(curr_year), 
            "quarter": int(curr_quarter),
            "species": "ALL",
            "idx": curr_idx
        }
        result.update(dict(zip(target_cols, y_pred)))
        df_result = pd.DataFrame([result])
        
        df_history = pd.concat([df_history, df_result], ignore_index=True)
        new_results.append(result)

    if new_results:
        df_new = pd.DataFrame(new_results)
        save_forecast_cache(df_new.drop(columns=["idx"], errors="ignore"))

def predict_future_metal_field_for_station(
    start_year,
    start_quarter,
    n_quarters,
    x,
    y
):
    start_idx = quarter_to_index(start_year, start_quarter)
    end_idx = start_idx + n_quarters - 1
    
    generate_missing_metal_forecasts(x, y, end_idx)
    
    df_cache = load_forecast_cache()
    df_cache_station = df_cache[(df_cache["X"] == x) & (df_cache["Y"] == y) & (df_cache["species"] == "ALL")].copy()
    
    if df_cache_station.empty:
        raise RuntimeError("Failed to fetch metal forecast from cache after generation.")
        
    df_cache_station["idx"] = quarter_to_index(df_cache_station["year"], df_cache_station["quarter"])
    
    results = df_cache_station[(df_cache_station["idx"] >= start_idx) & (df_cache_station["idx"] <= end_idx)]
    return results.sort_values("idx").drop(columns=["idx"])


def generate_missing_non_metal_forecasts(x, y, species, requested_idx):
    df_base = load_base_data()
    df_cache = load_forecast_cache()
    
    latest_year, latest_quarter = get_latest_available_time(df_base, df_cache, x, y, species=species)
    latest_idx = quarter_to_index(latest_year, latest_quarter)
    
    if latest_idx >= requested_idx:
        return
        
    logger.info(f"Missing non-metal forecast range ({species}): {latest_year}-Q{latest_quarter} -> index {requested_idx}")
    
    if species == "cobia":
        model_path = PROJECT_DIR / "model" / "output" / "hk_cobia_finetuned.pkl"
    elif species == "oyster":
        model_path = PROJECT_DIR / "model" / "output" / "hk_oyster_finetuned.pkl"
    else:
        raise ValueError(f"Unknown species: {species}")
        
    model = joblib.load(model_path)
    input_cols, features = joblib.load(str(model_path).replace(".pkl", "_features.pkl"))
    
    # Lookup Station / Station_Name from base data once
    station_meta = df_base[df_base["X"] == x][["X","Y","Station","Station_Name"]].drop_duplicates(subset=["X","Y"])
    station_val = station_meta.iloc[0]["Station"] if not station_meta.empty else None
    station_name_val = station_meta.iloc[0]["Station_Name"] if not station_meta.empty else None
    
    df_history = _build_history_df(df_base, df_cache, x, y, species=species)
    new_results = []
    
    for curr_idx in range(latest_idx + 1, requested_idx + 1):
        curr_year, curr_quarter = index_to_quarter(curr_idx)
        logger.info(f"Forecasting non-metal ({species}) quarter: {curr_year}-Q{curr_quarter}")
        
        idx_lag1 = curr_idx - 1
        idx_lag4 = curr_idx - 4
        
        lag1_row = df_history[df_history["idx"] == idx_lag1]
        lag4_row = df_history[df_history["idx"] == idx_lag4]
        
        if lag1_row.empty or lag4_row.empty:
             raise ValueError(f"Không đủ dữ liệu lịch sử lag1/lag4 tại {curr_year}-Q{curr_quarter} (species={species})")
             
        lag1_row = lag1_row.iloc[-1]
        lag4_row = lag4_row.iloc[-1]
        
        row = {}
        for c in features:
            row[f"{c}_lag1"] = float(lag1_row.get(c, 0))
            row[f"{c}_lag4"] = float(lag4_row.get(c, 0))
            
        row["Quarter_Num"] = int(curr_quarter)
        
        X_pred = pd.DataFrame([row])[input_cols].astype(float)
        y_pred = model.predict(X_pred)[0]
        y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))
        
        result = {
            "Station": station_val,
            "Station_Name": station_name_val,
            "X": x,
            "Y": y,
            "year": int(curr_year),
            "quarter": int(curr_quarter),
            "species": species,
            "idx": curr_idx
        }
        result.update(dict(zip(features, y_pred)))
        df_result = pd.DataFrame([result])
        
        df_history = pd.concat([df_history, df_result], ignore_index=True)
        new_results.append(result)
        
    if new_results:
        df_new = pd.DataFrame(new_results)
        save_forecast_cache(df_new.drop(columns=["idx"], errors="ignore"))

def predict_future_non_metal_field_for_station(
    species,
    x,
    y,
    start_year,
    start_quarter,
    n_quarters=4
):
    start_idx = quarter_to_index(start_year, start_quarter)
    end_idx = start_idx + n_quarters - 1
    
    generate_missing_non_metal_forecasts(x, y, species, end_idx)
    
    df_cache = load_forecast_cache()
    df_cache_station = df_cache[(df_cache["X"] == x) & (df_cache["Y"] == y) & (df_cache["species"] == species)].copy()
    
    if df_cache_station.empty:
        raise RuntimeError("Failed to fetch non-metal forecast from cache after generation.")
        
    df_cache_station["idx"] = quarter_to_index(df_cache_station["year"], df_cache_station["quarter"])
    results = df_cache_station[(df_cache_station["idx"] >= start_idx) & (df_cache_station["idx"] <= end_idx)]
    return results.sort_values("idx").drop(columns=["idx"])


METAL_TARGET_COLS = ["CN", "As", "Cd", "Pb", "Cu", "Hg", "Zn", "Total_Cr"]

def predict_for_station(
    species,
    x,
    y,
    start_year,
    start_quarter,
    n_quarters=4
):
    df1 = predict_future_non_metal_field_for_station(
        species=species,
        x=x,
        y=y,
        start_year=start_year,
        start_quarter=start_quarter,
        n_quarters=n_quarters
    )
    df2 = predict_future_metal_field_for_station(
        start_year=start_year,
        start_quarter=start_quarter,
        n_quarters=n_quarters,
        x=x,
        y=y
    )
    
    df1["year"] = df1["year"].astype(int)
    df1["quarter"] = df1["quarter"].astype(int)
    df2["year"] = df2["year"].astype(int)
    df2["quarter"] = df2["quarter"].astype(int)
    
    # From df2 (metal), only keep the actual predicted metal columns + join keys
    metal_available = [c for c in METAL_TARGET_COLS if c in df2.columns]
    df2_clean = df2[["year", "quarter"] + metal_available]
    
    # From df1 (non-metal), drop all-NaN metal columns so df2 values fill them
    all_nan_in_df1 = [
        c for c in df1.columns
        if c in metal_available and df1[c].isna().all()
    ]
    df1_clean = df1.drop(columns=all_nan_in_df1, errors="ignore")
    
    df_merged = pd.merge(
        df1_clean,
        df2_clean,
        on=["year", "quarter"],
        how="inner"
    )
    
    # Fill Station / Station_Name from base data if still NaN (models don't predict labels)
    if df_merged["Station"].isna().any() or "Station" not in df_merged.columns:
        df_base = load_base_data()
        station_lookup = (
            df_base[df_base["X"] == x][["X", "Y", "Station", "Station_Name"]]
            .drop_duplicates(subset=["X", "Y"])
        )
        if not station_lookup.empty:
            row = station_lookup.iloc[0]
            df_merged["Station"] = row["Station"]
            df_merged["Station_Name"] = row["Station_Name"]
    
    return df_merged

if __name__ == "__main__":
    df = predict_for_station(
        species="oyster",
        x=2318587,
        y=428692,
        start_year=2026,
        start_quarter=1,
        n_quarters=4
    )
    print(df)
    df.info()