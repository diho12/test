# Tóm tắt dữ liệu chất lượng nước Hong Kong (Hong Kong Water Quality)

**Đường dẫn:** `/prj/data/hk_water_quality`

Đây là bộ dữ liệu quan trắc chất lượng nước biển của Hong Kong. Theo yêu cầu, báo cáo tóm tắt này chỉ tập trung vào dữ liệu quan trắc gốc trong thư mục `water_data` và bỏ qua các tệp dữ liệu phục vụ mô hình sinh học (cobia/oyster modeling như `hk_cobia_quarterly_21vars.csv`, `hk_oyster_quarterly_21vars.csv`).

## 1. Cấu trúc thư mục dữ liệu

Thư mục `water_data` chứa các tệp CSV gốc tương ứng với các vùng nước (Water Control Zone) khác nhau của Hong Kong:
- `marine_water_quality_deep_bay.csv`
- `marine_water_quality_eastern_buffer.csv`
- `marine_water_quality_junk_bay.csv`
- `marine_water_quality_mirs_bay.csv`
- `marine_water_quality_port_shelter.csv`
- `marine_water_quality_southern.csv`
- `marine_water_quality_tolo.csv`
- `marine_water_quality_victoria_harbour.csv`
- `marine_water_quality_western_buffer.csv`

## 2. Thông tin chi tiết trong các tệp dữ liệu gốc (water_data)

Bộ dữ liệu chứa cấu trúc chuỗi thời gian phân chia theo từng trạm quan trắc (Station) và từng tầng nước (Depth). 

### 2.1. Định dạng chung của các bản ghi:
- **Thời gian (Dates):** Từ năm 1986 đến 2024 (chuẩn định dạng YYYY-MM-DD).
- **Vùng nước (Water Control Zone):** Khu vực quản lý (VD: Deep Bay, Tolo Harbour, Victoria Harbour...).
- **Trạm (Station):** ID của trạm quan trắc (VD: DM1, DM2).
- **Độ sâu (Depth):** Tầng nước tiến hành lấy mẫu (Surface Water, Middle Water, Bottom Water).

### 2.2. Các chỉ tiêu lý hóa và vi sinh quan trắc (29 Cột):
Bộ dữ liệu gồm đầy đủ các thông số về chất lượng nước, ví dụ như:
- **Thông số Vật lý:**
  - `Temperature (°C)`: Nhiệt độ nước.
  - `Turbidity (NTU)`: Độ đục.
  - `Salinity (psu)`: Độ mặn.
  - `Secchi Disc Depth (M)`: Độ trong suốt (đĩa Secchi).
  - `Suspended Solids (mg/L)` & `Volatile Suspended Solids (mg/L)`: Chất rắn lơ lửng.
- **Thông số Hóa học và Dinh dưỡng:**
  - `pH`: Độ pH đo được.
  - `Dissolved Oxygen (mg/L)` & `Dissolved Oxygen (%saturation)`: Nhu cầu oxy hòa tan (DO).
  - `5-day Biochemical Oxygen Demand (mg/L)`: Nhu cầu oxy sinh hóa 5 ngày (BOD5).
  - `Unionised Ammonia`, `Ammonia Nitrogen`, `Nitrate Nitrogen`, `Nitrite Nitrogen`, `Total Kjeldahl Nitrogen`, `Total Inorganic Nitrogen`, `Total Nitrogen`: Các hợp chất và dạng thông số về Nitơ.
  - `Total Phosphorus`, `Orthophosphate Phosphorus`: Các hợp chất Photpho.
  - `Silica (mg/L)`: Hàm lượng Silica.
- **Thông số Vi sinh và Sinh học:**
  - `E. coli (cfu/100mL)` & `Faecal Coliforms (cfu/100mL)`: Mật độ vi khuẩn E.coli và Coliform.
  - `Chlorophyll-a (μg/L)` & `Phaeo-pigments (μg/L)`: Nồng độ Diệp lục tố và Phaeo-pigments đánh giá sự phát triển của vi tảo.

*Lưu ý:* Giá trị khuyết thiếu trong bộ dữ liệu thường được biểu diễn bằng `'N/A'`, và một số giá trị đo được dưới ngưỡng phát hiện được đánh dấu nhỏ hơn (ví dụ `'<0.2'`).

## 3. Gợi ý hướng tiến hành phân tích
- **Phân tích biến động theo năm:** Sử dụng cột thời gian (`Dates`) để đánh giá biến động hóa lý theo các quý hoặc theo chu kỳ nhiều năm.
- **Phân tích so sánh đa vùng:** Có thể gom (concatenate) toàn bộ 9 tệp tin của `water_data` để phân biệt và so sánh đặc điểm môi trường giữa các Vùng vịnh (Water Control Zones) tại Hong Kong.
- **Phân tích đa chiều với Cát Bà/Quảng Ninh:** Đối chiếu các yếu tố cốt lõi (pH, DO, BOD5, DO, Salinity...) có giá trị tương đồng trong mô hình hoặc để training Machine Learning kết hợp cùng dữ liệu khu vực Quảng Ninh / Cát Bà đã chuẩn bị trước đó.
