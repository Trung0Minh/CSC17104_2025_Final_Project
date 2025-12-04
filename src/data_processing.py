import pandas as pd
import numpy as np

def load_data(filepath):
    """Tải tập dữ liệu từ đường dẫn tệp được chỉ định."""
    try:
        df = pd.read_csv(filepath)
        print(f"Đã tải dữ liệu thành công từ {filepath}")
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp tại {filepath}")
        return None

def get_column_info(df):
    """In thông tin về các cột bao gồm kiểu dữ liệu và số lượng giá trị không rỗng."""
    print("\nThông tin cột:")
    print(df.info())
    return df.dtypes

def get_numerical_columns(df):
    """Trả về danh sách tên các cột số."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def analyze_numerical_column_metrics(df, col):
    """
    Phân tích một cột số dựa trên các tiêu chí cụ thể:
    - Phân phối & Xu hướng trung tâm
    - Phạm vi & Giá trị ngoại lai
    - Chất lượng dữ liệu
    """
    print(f"--- Các chỉ số cho: {col} ---")
    
    # 1. Phân phối & Xu hướng trung tâm
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_dev = df[col].std()
    skewness = df[col].skew()
    
    print(f"\n[1] Phân phối & Xu hướng trung tâm:")
    print(f"   - Trung bình: {mean_val:.2f}")
    print(f"   - Trung vị: {median_val:.2f}")
    print(f"   - Độ lệch chuẩn: {std_dev:.2f}")
    print(f"   - Độ lệch: {skewness:.2f} (0 = bình thường, >0 = lệch phải, <0 = lệch trái)")

    # 2. Phạm vi & Giá trị ngoại lai
    min_val = df[col].min()
    max_val = df[col].max()
    
    # Phương pháp IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    num_outliers = outliers.count()
    
    print(f"\n[2] Phạm vi & Giá trị ngoại lai:")
    print(f"   - Min: {min_val}")
    print(f"   - Max: {max_val}")
    print(f"   - IQR: {IQR:.2f} (Q1={Q1:.2f}, Q3={Q3:.2f})")
    print(f"   - Ranh giới giá trị ngoại lai: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   - Số lượng giá trị ngoại lai: {num_outliers} ({num_outliers/len(df)*100:.2f}%)")

    # 3. Chất lượng dữ liệu
    missing_count = df[col].isnull().sum()
    total_rows = len(df)
    # Kiểm tra các giá trị duy nhất để xác định các giá trị có thể không hợp lệ
    unique_values = sorted(df[col].dropna().unique())
    
    print(f"\n[3] Chất lượng dữ liệu:")
    print(f"   - Giá trị bị thiếu: {missing_count} ({missing_count/total_rows*100:.2f}%)")
    if len(unique_values) <= 10:
        print(f"   - Các giá trị duy nhất ({len(unique_values)}): {unique_values}")
    
    print("-" * 40)
