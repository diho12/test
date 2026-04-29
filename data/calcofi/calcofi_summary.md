# Mô tả bộ dữ liệu CalCOFI (Bottle & Cast)

Tệp dữ liệu `bottle_and_cast.csv` là kết quả tổng hợp (merge) từ hai tập dữ liệu chính thuộc chương trình quan trắc hải dương học **CalCOFI** (California Cooperative Oceanic Fisheries Investigations):
1. **Cast data (`cast.csv`)**: Chứa thông tin về không gian và thời gian của từng đợt thả thiết bị lấy mẫu (CTD/rosette) tại các trạm.
2. **Bottle data (`bottle.csv`)**: Chứa các thông số vật lý, hóa học, và sinh học của các mẫu nước được lấy ở nhiều độ sâu khác nhau trong cùng một "cast".

## Từ điển dữ liệu (Data Dictionary)

Dưới đây là ý nghĩa của các cột (features) xuất hiện trong tệp dữ liệu đã tổng hợp:

### 1. Định danh (Identifiers) & Hệ thống trạm (Station Info)
* **`Btl_Cnt`**: Mã định danh duy nhất (ID) cho từng bản ghi mẫu nước (bottle).
* **`Cst_Cnt`**: Mã định danh duy nhất (ID) cho đợt thả thiết bị (cast). Trường này dùng để map giữa bottle và cast.
* **`Cast_ID`**: Mã dịnh danh dạng chuỗi của cast.
* **`Depth_ID`**: ID cụ thể cho bản ghi tại một độ sâu cụ thể.
* **`Sta_ID`**: String ID của trạm quan trắc.
* **`St_Line`**: Chỉ số tuyến (Line) theo hệ tọa độ lưới đo đạc đặc thù của CalCOFI.
* **`Distance`**: Khoảng cách từ trạm đo đến một mốc tham chiếu (thường là bờ biển).

### 2. Thông tin Không gian & Thời gian (Spatio-Temporal)
* **`Date`**: Ngày thực hiện lấy mẫu (định dạng MM/DD/YYYY).
* **`Year`**: Năm lấy mẫu.
* **`Month`**: Tháng lấy mẫu.
* **`Quarter`**: Quý trong năm.
* **`Lat_Dec`**: Vĩ độ của trạm đo (Decimal Degrees).
* **`Lon_Dec`**: Kinh độ của trạm đo (Decimal Degrees).

### 3. Thông số Hải dương học (Physicochemical Parameters)
* **`Depthm`**: Độ sâu thực tế lúc lấy mẫu nước, tính bằng mét (m).
* **`R_Depth`**: Độ sâu được báo cáo/tường minh (Reported Depth).
* **`T_degC`**: Nhiệt độ nước biển tính bằng độ C (°C).
* **`Salnty`**: Độ mặn của nước biển (Salinity).
* **`O2ml_L`**: Lượng oxy hòa tan trong nước tính bằng mililít trên lít (ml/L).
* **`pH1` / `pH2`**: Các giá trị đo độ pH của nước biển.

*(Lưu ý: Cột `Unnamed: 0` sinh ra do quá trình export file từ DataFrame của công cụ như Pandas và có thể bỏ qua).*

## Ứng dụng phổ biến
Dữ liệu này là một trong những bộ dữ liệu time-series về hải dương học lâu đời và có giá trị nhất, thường được dùng cho:
* Xây dựng mô hình Machine Learning hồi quy để **dự đoán nhiệt độ độ sâu** (predict Water Temperature based on Salinity, Depth, ...).
* Trực quan hóa và phân tích sự ấm lên toàn cầu, biến động sinh thái dọc bờ biển California.
