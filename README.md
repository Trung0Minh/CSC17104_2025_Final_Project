# Data Science Salaries 2023

**CSC17104 – LẬP TRÌNH KHOA HỌC DỮ LIỆU**  
**Đồ án cuối kỳ**

---

## Nhóm thực hiện
**Giảng viên hướng dẫn:** Phạm Trọng Nghĩa - Lê Nhựt Nam - Nguyễn Thanh Tình

**Thành viên:**
- **23122011** - Đoàn Hải Nam
- **23122014** - Hoàng Minh Trung
- **23122036** - Nguyễn Ngọc Khoa

---

## 📊 Tổng quan dự án
Dự án này phân tích bộ dữ liệu "Data Science Salaries 2023" để tìm hiểu các xu hướng trong thị trường việc làm khoa học dữ liệu toàn cầu. Nhóm em điều tra các yếu tố như kinh nghiệm, chức danh công việc, quy mô công ty và vị trí địa lý ảnh hưởng như thế nào đến mức lương. Dự án cũng áp dụng các mô hình Học máy để dự đoán mức lương dựa trên các thuộc tính này.

### Dữ liệu
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

## ❓ Câu hỏi nghiên cứu

### 1. Theo đuổi Manager hay trở thành Technical Expert?
Tại các cấp độ thâm niên cao (Senior & Executive), liệu việc chuyển hướng sang con đường Quản lý (*ví dụ: Manager, Head, Director*) có thực sự đảm bảo mức thu nhập cao hơn so với việc tiếp tục phát triển chuyên sâu theo con đường Kỹ thuật (*ví dụ: Principal, Staff, Architect*) hay không?

### 2. Tại sao công ty Vừa lại trả lương cao hơn công ty Lớn?
Liệu mức chênh lệch này có thực sự phản ánh chế độ đãi ngộ tốt hơn của công ty Vừa, hay nó chỉ là một ảo ảnh thống kê (Simpson's Paradox) gây ra bởi sự khác biệt trong **phân bố địa lý**?

### 3. Mức lương tăng như thế nào qua từng giai đoạn kinh nghiệm làm việc (EN $\rightarrow$ MI $\rightarrow$ SE $\rightarrow$ EX)?
Sự khác biệt về mức lương trung bình giữa các nhóm kinh nghiệm là gì? Mức lương tăng dần như thế nào khi người lao động chuyển từ level thấp lên level cao hơn?

### 4. Quy mô công ty ảnh hưởng như thế nào đến mức độ kinh nghiệm của nhân viên (EN, MI, SE, EX)?
Phân bố các mức độ kinh nghiệm của nhân viên trong từng nhóm quy mô công ty (S, M, L) như thế nào? Có sự khác biệt về cấu trúc nhân sự theo kinh nghiệm giữa công ty nhỏ, vừa và lớn hay không?

### 5. Yếu tố nào ảnh hưởng đến mức lương nhiều nhất?
Yếu tố nào đóng vai trò quan trọng nhất trong việc quyết định mức lương (`adjusted_salary`) của nhân sự ngành Data Science: Kinh nghiệm (`experience_level`), Vị trí địa lý (`employee_residence`), hay Loại hình công việc (`job_category`)?

### 6. Có thể xây dựng một mô hình Machine Learning để dự đoán mức lương?
Có thể xây dựng một mô hình Machine Learning để dự đoán mức lương thực tế (`adjusted_salary`) của một nhân sự dựa trên hồ sơ công việc (Kinh nghiệm, Vị trí, Quy mô công ty...) với độ chính xác bao nhiêu (đo lường bằng $R^2$ và $MAE$)?

---

## 🔍 Tóm tắt phát hiện chính 
Dựa trên quá trình Khám phá Dữ liệu (EDA), nhóm đã rút ra những quan sát cốt lõi:

*   **Sự thống trị của thị trường Mỹ (US):** Dữ liệu bị lệch nghiêm trọng về phía thị trường Mỹ (>50%). Mức lương tại Mỹ cao vượt trội so với phần còn lại của thế giới (thậm chí gấp 2-3 lần so với Châu Âu).
*   **Phân cực trong mô hình làm việc:** Phần lớn nhân sự làm việc hoàn toàn tại văn phòng (*On-site*) hoặc hoàn toàn từ xa (*Remote 100%*). Mô hình Hybrid (50%) chiếm tỷ lệ rất nhỏ (~7.2%) và có mức lương trung vị thấp nhất.
*   **Nghịch lý quy mô công ty:** Các công ty quy mô Vừa (*Medium*) lại có mức lương trung vị cao hơn các công ty Lớn (*Large*) trên bình diện tổng thể. Đây có thể là hệ quả của *Nghịch lý Simpson* do phân bố địa lý (công ty lớn tuyển dụng nhiều ở thị trường quốc tế giá rẻ hơn).
*   **Lương không tăng tuyến tính theo chức danh:** Mặc dù cấp quản lý thường được cho là lương cao hơn, nhưng dữ liệu cho thấy nhiều vị trí Chuyên gia kỹ thuật cấp cao (*Principal/Staff Engineer*) có mức lương ngang ngửa hoặc thậm chí nhỉnh hơn.

---

## 📂 Cấu trúc thư mục

```
.
├── data/
│   ├── raw/            # Dữ liệu gốc (ds_salaries.csv)
│   └── processed/      # Dữ liệu đã qua xử lý
├── src/
│   ├── data_processing.py  # Các hàm làm sạch và xử lý đặc trưng
│   ├── visualization.py    # Các hàm vẽ biểu đồ
│   ├── modeling.py         # Định nghĩa mô hình ML
│   └── __init__.py
├── DataScienceSalaries2023.ipynb   # Notebook phân tích chính
├── requirements.txt                # Các thư viện Python cần thiết
└── README.md                       # Tài liệu dự án
```

---

## 🚀 Hướng dẫn chạy

1.  **Clone repository:**
    ```bash
    git clone https://github.com/Trung0Minh/CSC17104_2025_Final_Project.git
    cd CSC17104_2025_Final_Project
    ```

2.  **Cài đặt thư viện:**
    Tạo môi trường ảo sau đó cài đặt thư viện:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Chạy Notebook:**
    Mở `DataScienceSalaries2023.ipynb` bằng Jupyter Notebook hoặc VS Code để xem phân tích và chạy các cell code.
    ```bash
    jupyter notebook DataScienceSalaries2023.ipynb
    ```

---

## 📦 Thư viện hỗ trợ

- **Cốt lõi:** `numpy`, `pandas`
- **Trực quan hóa:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning:** `scikit-learn`, `xgboost`, `category_encoders`
- **Tiện ích:** `country_converter`, `pycountry`, `tqdm`

Xem chi tiết phiên bản trong `requirements.txt`.
