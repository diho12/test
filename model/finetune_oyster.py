import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error

from basemodel import *

warnings.filterwarnings('ignore')

def finetune_model(base_model_path, new_data_path, output_path, features_list):
    """
    Hàm Fine-tune: Cập nhật mô hình cũ với dữ liệu mới.
    """
    base_model_path = str(base_model_path)
    output_path = str(output_path)
    
    print(f"\n🔧 BẮT ĐẦU FINE-TUNE MÔ HÌNH TỪ: {base_model_path}")
    
    # 1. LOAD MÔ HÌNH GỐC (BASE MODEL)
    if not os.path.exists(base_model_path):
        print(f" Lỗi: Không tìm thấy file model gốc tại {base_model_path}")
        return

    model = joblib.load(base_model_path)
    print(" Đã load xong model gốc.")

    # 2. LOAD METADATA (Để biết ngày xưa train dùng cột nào)
    meta_path = base_model_path.replace('.pkl', '_features.pkl')
    try:
        input_cols_old, features_old = joblib.load(meta_path)
        print(" Đã xác định được cấu trúc input/output cũ.")
    except:
        print(" Lỗi: Không tìm thấy file metadata (_features.pkl). Không thể fine-tune chuẩn.")
        return

    # 3. CHUẨN BỊ DỮ LIỆU MỚI (FINE-TUNE DATA)
    # Lưu ý: Phải dùng logic y hệt như lúc train base model
    print(f" Đang xử lý dữ liệu mới từ: {new_data_path}")
    df_ft, _ = prepare_time_series_data(new_data_path, features_list, lags=[1, 4])
    
    if df_ft is None or len(df_ft) == 0:
        print(" Dữ liệu fine-tune trống hoặc không đủ để tạo lag. Hủy bỏ.")
        return

    # Đảm bảo dữ liệu mới có đủ các cột như dữ liệu cũ
    # (Nếu thiếu cột nào thì điền 0 hoặc báo lỗi, ở đây ta giả định dữ liệu chuẩn)
    X_new = df_ft[input_cols_old]
    y_new = df_ft[features_list]

    print(f" Kích thước dữ liệu Fine-tune: {len(X_new)} mẫu")

    # 4. THỰC HIỆN FINE-TUNE (CẬP NHẬT TRỌNG SỐ)
    # Vì model là MultiOutputRegressor (chứa nhiều model con), ta phải update từng cái
    
    print(" Đang cập nhật kiến thức mới cho mô hình...")
    
    # Duyệt qua từng model con (tương ứng từng cột output: DO, pH, Temp...)
    for i, estimator in enumerate(model.estimators_):
        target_name = features_list[i]
        
        # A. Lấy "bộ não" (booster) của model cũ ra
        old_booster = estimator.get_booster()
        
        # B. Giảm tốc độ học (Learning Rate)
        # Khi fine-tune, ta nên học chậm lại để không "quên" kiến thức cũ quá nhanh
        estimator.set_params(learning_rate=0.005) 
        
        # C. Train tiếp (Incremental Learning)
        # Tham số quan trọng nhất: xgb_model=old_booster
        # Nghĩa là: "Đừng học từ đầu, hãy học tiếp từ cái cũ"
        estimator.fit(X_new, y_new.iloc[:, i], xgb_model=old_booster)
        
    # 5. ĐÁNH GIÁ LẠI TRÊN DỮ LIỆU MỚI
    print("\n KẾT QUẢ SAU KHI FINE-TUNE (TRÊN TẬP DỮ LIỆU MỚI):")
    print("-" * 50)
    y_pred = model.predict(X_new)
    rmse = np.sqrt(mean_squared_error(y_new, y_pred, multioutput='raw_values'))
    
    for i, col_name in enumerate(features_list):
        print(f"   🔹 {col_name:<15} RMSE: {rmse[i]:.4f}")
    
    print("-" * 50)
    print(f" RMSE trung bình: {np.mean(rmse):.4f}")

    # 6. LƯU MÔ HÌNH MỚI (FINETUNED MODEL)
    joblib.dump(model, output_path)
    # Lưu luôn metadata cho model mới (thực ra vẫn y hệt cái cũ)
    joblib.dump((input_cols_old, features_list), output_path.replace('.pkl', '_features.pkl'))
    
    print(f"\n Đã lưu model Fine-tune tại: {output_path}")


if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    
    # Đường dẫn model gốc (Base Model)
    MODEL_DIR = PROJECT_DIR / "model" / "output"
    BASE_OYSTER_MODEL = MODEL_DIR / "hk_oyster_forecast_model.pkl"
    
    # Đường dẫn dữ liệu mới để Fine-tune (Ví dụ: Dữ liệu năm 2024 mới về, hoặc dữ liệu riêng của 1 vùng)
    # Ở đây tôi dùng lại file csv cũ làm ví dụ, thực tế bạn thay bằng file mới
    NEW_DATA_PATH = PROJECT_DIR / "data" / "data_quang_ninh" / "qn_env_clean_ready.csv"
    
    # Đường dẫn lưu model mới
    OUTPUT_FINETUNE = MODEL_DIR / "hk_oyster_finetuned.pkl"

    print(f" Base Model: {BASE_OYSTER_MODEL}")
    
    # Chạy Fine-tune cho HÀU (Ví dụ)
    finetune_model(
        base_model_path = BASE_OYSTER_MODEL,
        new_data_path = NEW_DATA_PATH,
        output_path = OUTPUT_FINETUNE,
        features_list = OYSTER_FEATURES
    )