import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns


def distance_vn2000_km(x1: float, y1: float, x2: float, y2: float) -> float:
    """Khoảng cách không gian cho hệ VN2000 (m → km)"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / 1000.0


def compute_local_R_for_station_quarter(
    df_quarter: pd.DataFrame,
    station_id: str,
    max_dist_km: float = 30.0,
    bin_km: float = 1.0,
    alpha: float = 0.6,
    max_empty_gap_km: float = 3.0
) -> float:
    """
    Tính bán kính hoạt động (r_hsi) cho một trạm trong một quý cụ thể.
    """
    # ---- 1. Lấy trạm trung tâm ----
    center = df_quarter[df_quarter["Station"] == station_id]
    if center.empty:
        return np.nan
    center = center.iloc[0]

    center_hsi = center.get("HSI")
    center_label = center.get("HSI_Level")
    
    if pd.isna(center_hsi) or pd.isna(center_label):
        return np.nan

    # ---- 2. Tính ngưỡng ΔHSI theo toàn quý ----
    sigma_hsi = df_quarter["HSI"].std()
    if pd.isna(sigma_hsi):
        return np.nan
    if sigma_hsi == 0:
        return float(max_dist_km)

    delta_hsi_threshold = alpha * sigma_hsi

    # ---- 3. Thu thập trạm lân cận ----
    records = []
    min_dist_to_any_station = float('inf')
    
    for _, r in df_quarter.iterrows():
        if r["Station"] == station_id:
            continue

        d = distance_vn2000_km(center["X"], center["Y"], r["X"], r["Y"])
        
        if d < min_dist_to_any_station:
            min_dist_to_any_station = d
                
        if d > max_dist_km:
            continue

        records.append({
            "dist_km": d,
            "delta_hsi": abs(center_hsi - r["HSI"]),
            "label": r["HSI_Level"]
        })

    if not records:
        return 0.5

    tmp = pd.DataFrame(records)

    # ---- 4. Chia vòng đồng tâm theo khoảng cách ----
    bins = np.arange(0, max_dist_km + bin_km, bin_km)
    tmp["dist_bin"] = pd.cut(
        tmp["dist_km"],
        bins=bins,
        include_lowest=True
    )

    # ---- 5. Mở rộng R từng vòng, KHÔNG cho lẫn nhãn ----
    last_valid_R = 0.0
    empty_gap_count = 0
    max_empty_bins = max(1, int(max_empty_gap_km / bin_km))

    for dist_bin, g in tmp.groupby("dist_bin", observed=True):
        if g.empty:
            empty_gap_count += 1
            if empty_gap_count >= max_empty_bins:
                break
            continue
            
        empty_gap_count = 0

        # Điều kiện 1: ΔHSI trung bình vượt ngưỡng
        if g["delta_hsi"].mean() >= delta_hsi_threshold:
            break

        # Điều kiện 2 (QUAN TRỌNG): xuất hiện trạm khác nhãn trung tâm
        if (g["label"] != center_label).any():
            break

        last_valid_R = dist_bin.right

    # ---- 6. Ép R trong khoảng hợp lệ ----
    final_R = min(max_dist_km, max(last_valid_R, 0.5))
    
    # Giới hạn R không vượt quá một nửa khoảng cách tới trạm GẦN NHẤT
    # để đảm bảo tuyệt đối không có 2 đường tròn nào bị giao cắt nhau.
    if min_dist_to_any_station != float('inf'):
        final_R = min(final_R, min_dist_to_any_station / 2.0)
        
    return max(final_R, 0.1)


def compute_r_hsi(
    df_hsi: pd.DataFrame,
    max_dist_km: float = 30.0,
    bin_km: float = 1.0,
    alpha: float = 0.6
) -> pd.DataFrame:
    """
    Tính bán kính r_hsi cho toàn bộ DataFrame chứa kết quả dự báo HSI.
    Tự động nhóm theo quý và năm để tính cho từng trạm trong cùng thời điểm.
    """
    df = df_hsi.copy()
    
    # Handle Quarter string if year and quarter are not present
    if "year" not in df.columns or "quarter" not in df.columns:
        if "Quarter" in df.columns:
            dt = pd.to_datetime(df["Quarter"], errors="coerce")
            df["year"] = dt.dt.year
            df["quarter"] = dt.dt.quarter
        else:
            raise ValueError("DataFrame đầu vào phải có cột 'year' và 'quarter', hoặc 'Quarter'")
            
    required_cols = {"Station", "X", "Y", "year", "quarter", "HSI", "HSI_Level"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame đầu vào thiếu các cột bắt buộc: {missing_cols}")

    results = []

    for (year, quarter), g in df.groupby(["year", "quarter"]):
        g = g.reset_index(drop=True)

        for station in g["Station"].unique():
            R = compute_local_R_for_station_quarter(
                df_quarter=g,
                station_id=station,
                max_dist_km=max_dist_km,
                bin_km=bin_km,
                alpha=alpha
            )
            
            results.append({
                "Station": station,
                "year": year,
                "quarter": quarter,
                "r_hsi": R
            })

    if results:
        df_r = pd.DataFrame(results)
        df = df.merge(df_r, on=["Station", "year", "quarter"], how="left")
    else:
        df["r_hsi"] = np.nan
        
    return df


if __name__ == "__main__":
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    MERGED_HSI_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_trained_data" / "hsi_forecast_merged.csv"
    
    if MERGED_HSI_PATH.exists():
        df_merged = pd.read_csv(MERGED_HSI_PATH)
        print(f"Đã tải {len(df_merged)} bản ghi từ {MERGED_HSI_PATH.name}")
        
        if "r_hsi" in df_merged.columns:
            print("\nCột 'r_hsi' đã tồn tại. Đang tạo thống kê và biểu đồ...")
            # Thống kê cơ bản
            print(df_merged[["Station", "year", "quarter", "HSI", "r_hsi"]].head())
            print("\nThống kê mô tả r_hsi:")
            print(df_merged["r_hsi"].describe())
            
            # Vẽ biểu đồ phân phối r_hsi
            plt.figure(figsize=(8, 5))
            r_hsi_values = df_merged["r_hsi"].dropna()
            
            if not r_hsi_values.empty:
                sns.histplot(r_hsi_values, bins=20, kde=True, color="teal")
                plt.title("Distribution of r_hsi (km)")
                plt.xlabel("r_hsi (km)")
                plt.ylabel("Count")
                
                OUT_FIG = PROJECT_DIR / "figure"
                OUT_FIG.mkdir(exist_ok=True)
                fig_path = OUT_FIG / "r_hsi_distribution.png"
                plt.savefig(fig_path, dpi=300)
                plt.close()
                print(f"📊 Đã lưu biểu đồ phân phối r_hsi tại: {fig_path}")
            else:
                print("Tất cả giá trị r_hsi đều là NaN, không thể vẽ biểu đồ.")
        else:
            print("Cột 'r_hsi' chưa được tính toán trong file này.")
    else:
        print(f"Không tìm thấy file: {MERGED_HSI_PATH}")