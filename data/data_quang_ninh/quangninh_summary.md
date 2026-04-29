# Tóm tắt Dữ liệu Môi trường Biển Quảng Ninh (2021 - 2024)

Tệp này cung cấp thông tin tổng quan về bộ dữ liệu quan trắc chất lượng nước và môi trường tại khu vực Quảng Ninh, bao gồm danh sách các tệp, cấu trúc cột, và các hướng phân tích chính.

*(Lưu ý: Theo yêu cầu, các tệp kết quả mô hình liên quan đến môi trường sống của cá giò và hàu như `hsi_cobia.csv`, `hsi_oyster.csv`, `R_cobia.csv`, `R_oyster.csv` không được tóm tắt trong tài liệu này).*

## 1. Cấu trúc thư mục và Các tệp hiện có
- **`qn_env_clean_ready.csv`**: Dữ liệu lõi về môi trường đã qua quá trình làm sạch. Chứa tất cả các chỉ tiêu quan trắc tại từng trạm theo thời gian.
- **`qn_data_21_24.csv`**: Dữ liệu thô/tổng hợp gốc chứa kết quả quan trắc các quý từ 2021 đến 2024. Cột thời gian và các giá trị vẫn ở định dạng nguyên bản.
- **`toa_do_qn.csv`**: Tệp tham chiếu không gian, cung cấp tọa độ `X`, `Y` cho các mã trạm quan trắc (từ NB1, NB2,...).
- **`danh_sach_tra_cuu.txt`**: Từ điển tra cứu ánh xạ giữa mã trạm (NBx) và tên vị trí thực tế, kèm link Google Maps để kiểm chứng trực quan địa điểm.
- **`moddata.py`**: Mã nguồn Python hỗ trợ xử lý dữ liệu. Chứa logic (`pandas`, biểu thức chính quy) để làm sạch và chuyển đổi cột "Quarter" (ví dụ: "Quý 1 2021") sang định dạng thời gian chuẩn `YYYY-MM-DD` ("2021-01-01").
- **`Tong-hop-21-24-gen.xlsx` / `Tong hop NB.2021-2025.IN (3).xlsx`**: Các sổ làm việc Excel chứa dữ liệu tổng hợp tổng quan.

## 2. Từ điển Dữ liệu (Dựa trên tệp `qn_env_clean_ready.csv`)
Dữ liệu quan trắc được cấu trúc dưới dạng bảng time-series theo từng trạm, bao gồm các nhóm thông số sau:

### Trạm và Không gian - Thời gian
- `Station`: Mã định danh trạm quan trắc (Ví dụ: NB1, NB2).
- `Station_Name`: Tên mô tả chi tiết vị trí trạm.
- `Quarter`: Thời gian tiến hành quan trắc (đã chuẩn hóa định dạng thời gian `YYYY-MM-DD`).
- `X`, `Y`: Hệ tọa độ phẳng định vị trạm.

### Hóa - Lý Thủy vực
- `Temperature`: Nhiệt độ (°C).
- `pH`: Tính axit/kiềm của nước.
- `Salinity`: Độ mặn.
- `DO`: Lượng oxy hòa tan.
- `TSS`: Tổng chất rắn lơ lửng.
- `Transparency`: Độ trong.

### Dinh dưỡng và Ô nhiễm Hữu cơ
- `NH3`: Amoni.
- `PO4`: Phốt phát.
- `BOD5`: Nhu cầu oxy sinh hóa (đánh giá ô nhiễm hữu cơ phân hủy sinh học).
- `COD`: Nhu cầu oxy hóa học.
- `Alkalinity`: Độ kiềm.
- `Coliform`: Mật độ vi khuẩn (phân tích mức độ ô nhiễm sinh học).
- Khí và gốc độc: `H2S`, `CN` (Tổng xianua).

### Kim loại nặng
- `As` (Thạch tín), `Cd` (Cadimi), `Pb` (Chì), `Cu` (Đồng), `Hg` (Thủy ngân), `Zn` (Kẽm), `Total_Cr` (Tổng Crom).

## 3. Hướng Phân tích Tiềm năng
1. **Phân tích Biến động theo Thời gian (Time-series):** Sử dụng cột `Quarter` để vẽ biểu đồ line chart, theo dõi tính chu kỳ theo mùa của các chỉ tiêu sinh hóa (như độ mặn, nhiệt độ giảm/tăng theo quý).
2. **Khảo sát Không gian (Spatial Analysis):** Dùng `X`, `Y` kết hợp với thư viện không gian (như `geopandas` hoặc phần mềm QGIS) vẽ bản đồ nhiệt phân bố mức độ ô nhiễm kim loại nặng hay Coliform dọc vùng ven bờ Quảng Ninh.
3. **Tiền xử lý Mô hình hóa:** Tệp `qn_env_clean_ready.csv` đã sẵn sàng cho việc xây dựng các mô hình Machine Learning đánh giá chất lượng nước, đánh giá chỉ số môi trường sống (HSI) cho các loài thủy sản, hay dự báo xu hướng ô nhiễm qua các năm.
