# CHU-TUYET-NHI-BTL1-PYTHON
Bài tập lớn 1 môn Lập trình Python của Chu Tuyết Nhi - B23DCCE075.

**Mô tả ngắn:**
Dự án này thực hiện việc thu thập, phân tích và mô hình hóa dữ liệu cầu thủ bóng đá Ngoại hạng Anh mùa giải 2024-2025.

**Nội dung chính:**
* **Thu thập dữ liệu (Chương 1):** Lấy dữ liệu thống kê cầu thủ (> 90 phút thi đấu) từ `fbref.com` bằng Selenium và BeautifulSoup. Dữ liệu được lưu trong `results.csv`.
* **Phân tích & Trực quan hóa (Chương 2):** Phân tích thống kê mô tả (top/bottom 3 cầu thủ, trung bình, trung vị, độ lệch chuẩn), trực quan hóa phân phối dữ liệu bằng histogram. Kết quả lưu trong `results2.csv`  và `top_3.txt`.
* **Phân cụm (Chương 3):** Phân nhóm cầu thủ bằng thuật toán K-Means và trực quan hóa bằng PCA.
* **Ước tính giá trị (Chương 4):** Thu thập giá trị chuyển nhượng ước tính (ETV) từ `footballtransfers.com` cho cầu thủ > 900 phút, xử lý dữ liệu (lưu trong `estimation_data_with_highest_etv.csv`) và xây dựng mô hình Gradient Boosting Regressor để dự đoán giá trị cầu thủ.

**Cấu trúc thư mục:**
* `/SourceCode`: Chứa mã nguồn Python cho từng phần (I, II, III, IV).
* `/Report`: Chứa báo cáo chi tiết (tiếng Việt và tiếng Anh) và các file output (CSV, TXT, plots).
* `/Report/LaTex_Src`: Chứa mã nguồn LaTeX của báo cáo.
