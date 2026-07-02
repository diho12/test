HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN

1. Phần mềm cần có
- Python 3.10 hoặc 3.11 (khuyến nghị Python 3.11, bản 64-bit).
- IDE đề xuất: Visual Studio Code (có thể dùng PyCharm).

2. Tạo môi trường ảo
Mở Terminal/PowerShell tại thư mục gốc của dự án, nơi có requirements.txt:

Windows:
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1

Nếu PowerShell chặn kích hoạt môi trường ảo, chạy một lần:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

macOS/Linux:
    python3.11 -m venv .venv
    source .venv/bin/activate


3. Cài thư viện
    python -m pip install -r requirements.txt

4. Kiểm tra dữ liệu và mô hình
Trước khi chạy, cần bảo đảm dự án có các thành phần sau:
- data/data_quang_ninh/qn_env_clean_ready.csv
- data/data_quang_ninh/qn_trained_data/
- model/output/metal_ts_model.pkl
- model/output/hk_cobia_finetuned.pkl
- model/output/hk_cobia_finetuned_features.pkl
- model/output/hk_oyster_finetuned.pkl
- model/output/hk_oyster_finetuned_features.pkl

Nếu chưa có các file .pkl, sau khi hoàn thành mục 2 và mục 3, quay lại thư mục
gốc của dự án và chạy lần lượt:
    python model/basemodel.py
    python model/finetune_cobia.py
    python model/finetune_oyster.py
    python model/metal.py
Thứ tự trên sẽ:
- Tạo model gốc và file thông tin đặc trưng cho Cá giò và Hàu.
- Fine-tune hai model bằng dữ liệu Quảng Ninh.
- Tạo model dự báo các chỉ tiêu kim loại.
Sau khi chạy xong, kiểm tra lại thư mục model/output. Cần có tối thiểu:
- metal_ts_model.pkl
- hk_cobia_finetuned.pkl và hk_cobia_finetuned_features.pkl
- hk_oyster_finetuned.pkl và hk_oyster_finetuned_features.pkl
Quá trình huấn luyện có thể mất vài phút tùy cấu hình máy. Chỉ chuyển sang bước
chạy ứng dụng khi các script hoàn tất và không báo lỗi.


5. Chạy ứng dụng
Vẫn đứng tại thư mục gốc của dự án và chạy:
    python -m streamlit run interface/main.py

Sau khi khởi động, trình duyệt thường tự mở tại:
    http://localhost:8501

Nhấn Ctrl+C trong Terminal để dừng ứng dụng.

6. Thiết lập nhanh với Visual Studio Code
- Mở toàn bộ thư mục dự án bằng VS Code.
- Cài extension Python của Microsoft.
- Chọn Python Interpreter là file Python bên trong thư mục .venv.
- Mở Terminal tích hợp và chạy lệnh Streamlit ở mục 5.
