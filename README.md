# Phân tích Lương ngành Khoa học Dữ liệu 2023 (Data Science Salaries 2023)

**CSC17104 – LẬP TRÌNH KHOA HỌC DỮ LIỆU**  
**Đồ án Cuối kỳ**

---

## 👥 Nhóm Thực hiện
**Giảng viên hướng dẫn:** Phạm Trọng Nghĩa - Lê Nhựt Nam - Nguyễn Thanh Tình

**Thành viên:**
- **23122011** - Đoàn Hải Nam
- **23122014** - Hoàng Minh Trung
- **23122036** - Nguyễn Ngọc Khoa

---

## 📊 Tổng quan Dự án
Dự án này phân tích bộ dữ liệu "Data Science Salaries 2023" để tìm hiểu các xu hướng trong thị trường việc làm khoa học dữ liệu toàn cầu. Chúng tôi điều tra các yếu tố như kinh nghiệm, chức danh công việc, quy mô công ty và vị trí địa lý ảnh hưởng như thế nào đến mức lương. Dự án cũng áp dụng các mô hình Học máy (Machine Learning) để dự đoán mức lương dựa trên các thuộc tính này.

### Dữ liệu (Dataset)
- **Nguồn:** [Kaggle - Data Science Salaries 2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023/data)
- **Mô tả:** Bộ dữ liệu chứa thông tin lương của các vị trí Khoa học Dữ liệu từ năm 2020 đến 2023.
- **Các đặc trưng chính:**
  - `work_year`: Năm trả lương.
  - `experience_level`: Entry (EN), Mid (MI), Senior (SE), Executive (EX).
  - `job_title`: Chức danh cụ thể (ví dụ: Data Scientist, ML Engineer).
  - `salary_in_usd`: Lương quy đổi sang USD.
  - `employee_residence` & `company_location`: Thông tin địa lý.
  - `company_size`: Nhỏ (S), Vừa (M), Lớn (L).
  - `remote_ratio`: 0 (Tại văn phòng), 50 (Lai/Hybrid), 100 (Từ xa/Remote).

---

## ❓ Câu hỏi Nghiên cứu & Phát hiện Chính

### 1. Làm Quản lý (Manager) hay Chuyên gia Kỹ thuật (Technical Expert): Hướng nào lương cao hơn?
- **Phát hiện:** Chuyển sang làm Quản lý **không đảm bảo** lương cao hơn ngay lập tức.
- **Chi tiết:** Ở cấp độ **Senior**, các Chuyên gia Kỹ thuật có mức lương trung vị cao hơn (**$164k**) so với Quản lý ($156k). Nhóm Quản lý chỉ vượt lên ở cấp độ **Executive** ($182k so với $167k). Các trường hợp lương "khủng" (>$400k) xuất hiện ở cả hai hướng, chứng tỏ kỹ năng kỹ thuật chuyên sâu được đánh giá cao ngang ngửa kỹ năng lãnh đạo.

### 2. Tại sao công ty quy mô Vừa (Medium) lại trả lương cao hơn công ty Lớn (Large)?
- **Phát hiện:** Đây là một ví dụ điển hình của **Nghịch lý Simpson** gây ra bởi yếu tố địa lý.
- **Chi tiết:** Các công ty Lớn có vẻ trả lương thấp hơn (trung vị toàn cầu $100k) vì họ tuyển dụng tỷ lệ nhân sự lớn ở các thị trường quốc tế/chi phí thấp (46% Non-US). Khi chỉ xét riêng tại **Mỹ**, khoảng cách này gần như biến mất (Medium: $147k so với Large: $142k).

### 3. Mức lương tăng trưởng như thế nào theo kinh nghiệm?
- **Phát hiện:** Tốc độ tăng lương tuân theo **Quy luật Lợi suất giảm dần (Diminishing Returns)**.
- **Chi tiết:** Tốc độ tăng trưởng cao nhất là từ **Entry lên Mid-level (~14%/năm)**. Tốc độ này chậm lại từ Mid lên Senior (~10.6%) và giảm sâu từ Senior lên Executive (~7%).

### 4. Quy mô công ty ảnh hưởng thế nào đến cơ cấu nhân sự?
- **Phát hiện:** Mỗi quy mô công ty có "DNA tuyển dụng" riêng biệt.
- **Chi tiết:**
  - **Công ty Nhỏ:** Cân bằng giữa nhân sự mới (Entry) và trung cấp (Mid-level).
  - **Công ty Vừa:** Tập trung áp đảo vào nhân sự cấp cao (**Senior level chiếm 66%**).
  - **Công ty Lớn:** Cơ cấu cân bằng hơn nhưng vẫn thiên về Senior.

### 5. Những yếu tố nào ảnh hưởng mạnh nhất đến lương?
- **Phát hiện:** **Vị trí địa lý là Vua (Location is King).**
- **Chi tiết:** `employee_residence` (Nơi ở nhân viên) là yếu tố quan trọng nhất, chiếm **>55%** sự biến thiên của lương. Tiếp theo là `experience_level` (#2) và `job_title` (#3). Bất ngờ là `remote_ratio` có tác động trực tiếp rất nhỏ đến mức lương.

### 6. Có thể dự đoán lương bằng Machine Learning không?
- **Phát hiện:** Có thể, nhưng có giới hạn.
- **Mô hình:** Random Forest & XGBoost.
- **Hiệu suất:** **R² Score ~0.44**, **MAE ~$37,000**.
- **Insight:** Mô hình tốt để xác định xu hướng lương "sàn". Tuy nhiên, khoảng 55% sự biến thiên của lương phụ thuộc vào các "biến ẩn" không có trong dữ liệu (ví dụ: tech stack cụ thể, kỹ năng đàm phán, phân khúc công ty).

---

## 📂 Cấu trúc Thư mục

```
.
├── data/
│   ├── raw/            # Dữ liệu gốc (ds_salaries.csv)
│   └── processed/      # Dữ liệu đã qua xử lý
├── src/
│   ├── data_processing.py  # Pipeline làm sạch và xử lý đặc trưng
│   ├── visualization.py    # Các hàm vẽ biểu đồ
│   ├── modeling.py         # Định nghĩa mô hình ML (nếu có)
│   └── __init__.py
├── DataScienceSalaries2023.ipynb  # Notebook phân tích chính (EDA + Modeling)
├── tasks.txt           # Danh sách công việc
├── requirements.txt    # Các thư viện Python cần thiết
└── README.md           # Tài liệu dự án
```

---

## 🚀 Hướng dẫn Chạy

1.  **Clone repository:**
    ```bash
    git clone <repo-url>
    cd CSC17104_2025_Final_Project
    ```

2.  **Cài đặt thư viện:**
    Khuyên dùng môi trường ảo (virtual environment).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Chạy Notebook:**
    Mở `DataScienceSalaries2023.ipynb` bằng Jupyter Notebook hoặc VS Code để xem phân tích và chạy các cell code.
    ```bash
    jupyter notebook DataScienceSalaries2023.ipynb
    ```

---

## 📦 Thư viện Phụ thuộc

- **Cốt lõi:** `numpy`, `pandas`
- **Trực quan hóa:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning:** `scikit-learn`, `xgboost`, `category_encoders`
- **Tiện ích:** `country_converter`, `pycountry`, `tqdm`

Xem chi tiết phiên bản trong `requirements.txt`.
