# Tóm tắt thay đổi flow lưu HSI forecast

## 1. Mục tiêu thay đổi

Trước đây HSI được tính trong `utils/hsi.py` bằng `hsi_list`, sau đó chỉ gắn vào dataframe trả về cho app sử dụng tạm thời. Kết quả HSI không có một file lưu trữ ổn định để truy xuất lại.

Mục tiêu mới là lưu kết quả HSI forecast ra CSV, nhưng không tách thành nhiều file rời rạc theo loài. Thay vào đó dùng một file merge có đủ thông tin môi trường, kim loại, HSI và loài.

## 2. Flow hiện tại sau khi sửa

1. `interface/main.py` gọi `predict_for_station(...)` để sinh forecast môi trường.
2. Forecast được đưa vào `compute_hsi(forecast_df, species)` trong `utils/hsi.py`.
3. `compute_hsi()` vẫn chỉ tính toán và trả về dataframe có thêm:
   - `HSI`
   - `HSI_Level`
4. `main.py` gọi `save_hsi_forecast(...)` để lưu dataframe đã có HSI vào file merge.
5. File kết quả được lưu tại:

```text
data/data_quang_ninh/qn_trained_data/hsi_forecast_merged.csv
```

## 3. File CSV mới

File chính hiện là:

```text
hsi_forecast_merged.csv
```

Schema:

```text
Station, Station_Name, Quarter, year, quarter, species, X, Y,
DO, Temperature, pH, Salinity, NH3, PO4, H2S, BOD5, COD, TSS,
Coliform, Alkalinity, Transparency,
CN, As, Cd, Pb, Cu, Hg, Zn, Total_Cr,
HSI, HSI_Level
```

Mỗi dòng đại diện cho:

```text
1 trạm + 1 quý + 1 loài
```

Khóa chống trùng khi lưu:

```text
X, Y, year, quarter, species
```

Nếu tính lại cùng trạm/quý/loài, dòng cũ sẽ được thay bằng kết quả mới nhất.

## 4. Các hàm mới trong `utils/hsi.py`

`compute_hsi(df_forecast, species)`:

- Giữ nguyên vai trò tính HSI.
- Không tự ghi file để tránh side effect.

`prepare_hsi_forecast_for_save(df_hsi, species)`:

- Chuẩn hóa dataframe trước khi lưu.
- Tạo cột `Quarter` dạng ngày giống clean data, ví dụ `2026-01-01`.
- Gắn `species`.
- Sắp xếp cột theo schema thống nhất.

`load_hsi_forecast(path=HSI_FORECAST_PATH)`:

- Đọc file HSI forecast merge.
- Nếu file chưa có hoặc đang rỗng thì trả dataframe rỗng đúng schema.

`save_hsi_forecast(df_hsi, species, path=HSI_FORECAST_PATH)`:

- Append dữ liệu mới vào file merge.
- Deduplicate theo `X, Y, year, quarter, species`.
- Ghi lại CSV.

## 5. Các điểm đã cải thiện

- HSI không còn chỉ tồn tại trong list/session tạm thời.
- Có một file output thống nhất để truy xuất lại kết quả forecast và HSI.
- Không phải join ngược nhiều file khi muốn xem vì sao HSI cao/thấp.
- Có cả chỉ số môi trường, kim loại, `HSI`, `HSI_Level` trong cùng một dòng.
- Dễ lọc theo loài bằng cột `species` thay vì tách `hsi_cobia_forecast.csv` và `hsi_oyster_forecast.csv`.
- `compute_hsi()` vẫn là hàm tính toán thuần, nên ít rủi ro khi app gọi nhiều lần.
- Ghi file có khóa chống trùng, tránh nhân đôi dữ liệu khi tính lại.

## 6. Tình trạng các file HSI cũ

Hai file này hiện không còn là nguồn chính:

```text
data/data_quang_ninh/qn_trained_data/hsi_cobia_forecast.csv
data/data_quang_ninh/qn_trained_data/hsi_oyster_forecast.csv
```

Chúng đang rỗng và chưa được app sử dụng. Flow mới dùng `hsi_forecast_merged.csv`.

## 7. Test đã chạy

Đã chạy kiểm tra cú pháp:

```text
python -m py_compile utils\hsi.py interface\main.py
```

Kết quả: pass.

Có 2 warning cũ trong `main.py` về regex `"\d+"`, không liên quan đến thay đổi HSI.

Đã test helper lưu CSV tạm:

- `compute_hsi()` tính được HSI.
- `save_hsi_forecast()` ghi được CSV.
- Ghi trùng cùng khóa 2 lần vẫn chỉ còn 1 dòng.

Đã smoke test forecast thật cho `NB1`, `cobia`, `2026 Q1`:

```text
Station: NB1
Quarter: 2026-01-01
species: cobia
HSI: 0.82138
```

## 8. Lưu ý

`model/finetune_cobia.py` đang có diff format code, nhưng không liên quan đến flow HSI. `model/finetune_oyster.py` không có thay đổi.
