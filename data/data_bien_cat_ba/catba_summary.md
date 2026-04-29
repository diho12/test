# Tóm tắt Dữ liệu Chất lượng Nước biển Cát Bà (2020 - 2024)

Thư mục này chứa bộ dữ liệu quan trắc chất lượng môi trường nước biển tại khu vực Cát Bà, được đo đạc xuyên suốt trong giai đoạn 5 năm từ 2020 đến 2024.

## 1. Cấu trúc thư mục

Các tệp trong thư mục thể hiện luồng làm việc (pipeline) từ dữ liệu thô đến dữ liệu tinh, bao gồm:
- **Các tệp PDF định dạng thô (`*.pdf`)**: Trích xuất từ báo cáo gốc "Tổng hợp kết quả nước biển KV Cát Bà 2020, 2021, 2022, 2023, 2024".
- **Các tệp văn bản DOCX (`sea_2020.docx` - `sea_2024.docx`)**: Dữ liệu thô đang trong quá trình chuyển đổi và bóc tách bảng biểu từ PDF sang Text.
- **Các tệp CSV sạch (`nuoc_bien_2020_clean.csv` - `nuoc_bien_2024_clean.csv`)**: Dữ liệu tabular đã được chuẩn hóa và làm sạch (cleaning), ở trạng thái hoàn thiện để đưa vào phân tích hoặc mô hình học máy.

## 2. Từ điển Dữ liệu (Dựa trên các tệp `_clean.csv`)

Dữ liệu quan trắc được thu thập theo từng trạm và từng đợt, ghi nhận lại qua các trường thuộc tính dưới đây:

### 2.1. Nhóm thông tin thời gian & địa điểm
* **`NB`**: Mã trạm lấy mẫu (Nước Biển). Ví dụ: *NB1, NB2,..., NB28*.
* **`Kí_hiệu`**: Ký hiệu chỉ điểm thời gian lấy mẫu (Tháng/Năm). Ví dụ: *T02/24* là đợt sóng tháng 2 năm 2024.
* **`Tide`**: Hiện trạng thủy triều tại thời điểm đo:
  * `CT`: Chân triều (Triều cạn)
  * `ĐT`: Đỉnh triều (Triều cường)

### 2.2. Nhóm thông số chất lượng nước (Chỉ tiêu Hóa, Lý & Vi sinh)
* **`pH`**: Độ đo tính kiềm/axit của nước biển.
* **`DO`**: Lượng Oxy hòa tan trong nước (Dissolved Oxygen), chỉ báo sinh tồn cho sinh vật phù du.
* **`TSS`**: Tổng lượng chất rắn lơ lửng (Total Suspended Solids), đánh giá độ đục.
* **`NH4`**: Nồng độ Amoni (Ammonium), đánh giá mức độ ô nhiễm hữu cơ.
* **`PO4`**: Nồng độ photphat, chỉ tiêu đánh giá phú dưỡng hóa học.
* **`As` (Asen) & `Pb` (Chì)**: Các kim loại nặng độc hại. Dữ liệu này thường bị khuyến khuyết (Null/Blank) do nồng độ quá nhỏ hoặc không thuộc đợt quan trắc trọn gói.
* **`Dau_mo`**: Hàm lượng dầu mỡ khoáng trong mẫu nước.
* **`Coliform`**: Mức vi khuẩn Coliform, chỉ số ô nhiễm vi sinh.

## 3. Ứng dụng & Phân tích tiềm năng
Do đặc thù chia thep lớp trạm, phân kỳ hàng tháng/năm và bị chi phối bởi thủy triều, bộ dữ liệu cho phép:
- **Đánh giá xu hướng time-series:** Xem xét mức độ suy thoái môi trường hay phục hồi qua 5 cấp độ lịch sử (2020->2024).
- **Phân tích đối ngẫu Thủy triều:** Khảo sát nồng độ tạp chất thay đổi như thế nào giữa khung giờ Triều kiệt (CT) và Triều cường (ĐT).
- **Phân cấp vùng an toàn (Clustering/Classification):** Nhận diện trực tiếp trạm nào ven Cát Bà đang bị gánh nặng ô nhiễm cục bộ từ cảng hoặc du lịch (dựa theo TSS, NH4, Dau_mo).
