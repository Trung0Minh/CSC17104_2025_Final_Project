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

def analyze_categorical_column(df, col_name, top_n=15):
    """
    Phân tích một cột phân loại:
    1. In số lượng giá trị khác nhau
    2. In value counts
    
    Tham số:
        df: DataFrame
        col_name: tên cột phân loại
        top_n: số lượng nhãn hiển thị trên biểu đồ nếu nhiều nhãn (>20)
    """
    print("="*50)
    print(f"Phân tích cột: {col_name}")
    
    # 1. Số lượng giá trị khác nhau
    n_unique = df[col_name].nunique()
    print(f"Số lượng giá trị khác nhau: {n_unique}\n")
    
    # 2. Value counts
    print("Value counts:")
    print(df[col_name].value_counts(), "\n")

def check_currency_rates(df, threshold_pct=5):
    """
    Kiểm tra tính nhất quán tỉ giá theo từng currency.
    Không thay đổi DataFrame gốc.
    """
    df_copy = df.copy()
    df_copy["actual_rate"] = df_copy["salary_in_usd"] / df_copy["salary"]

    print("=== SUMMARY: Exchange Rate Consistency by Currency ===\n")

    summary = (
        df_copy.groupby("salary_currency")["actual_rate"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    summary["diff_pct"] = (summary["max"] - summary["min"]) / summary["min"] * 100
    summary = summary.sort_values("diff_pct", ascending=False)

    # In summary đẹp
    for _, row in summary.iterrows():
        curr = row["salary_currency"]
        print(
            f"{curr:<4} | "
            f"min_rate={row['min']:.6f} | "
            f"max_rate={row['max']:.6f} | "
            f"diff={row['diff_pct']:.2f}% | "
            f"count={int(row['count'])}"
        )

    # Currency bất thường
    inconsistent = summary[summary["diff_pct"] > threshold_pct]["salary_currency"].tolist()

    print("\n=== Currencies with inconsistent exchange rates (> {}%) ===".format(threshold_pct))
    print(inconsistent)

    return summary, inconsistent

def check_job_title_anomalies(df, col="job_title"):
    titles = df[col].astype(str)

    anomalies = pd.DataFrame({"job_title": titles})

    # 1. Ký tự lạ (ký tự không nằm trong tập cho phép)
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_/")

    def has_weird_char(s):
        return any(ch not in allowed_chars for ch in s)

    anomalies["has_weird_chars"] = titles.apply(has_weird_char)

    # 2. Nhiều khoảng trắng liên tiếp
    anomalies["multiple_spaces"] = titles.apply(lambda s: "  " in s)

    # 3. Toàn chữ hoa / toàn chữ thường
    anomalies["all_caps"] = titles.apply(lambda s: s.isupper())
    anomalies["all_lower"] = titles.apply(lambda s: s.islower())

    # 4. Quá ngắn hoặc quá dài
    anomalies["too_short"] = titles.apply(lambda s: len(s.strip()) < 3)
    anomalies["too_long"] = titles.apply(lambda s: len(s.strip()) > 60)

    # 5. Chứa số
    anomalies["contains_number"] = titles.apply(lambda s: any(ch.isdigit() for ch in s))

    # 6. Lặp từ đơn giản (không dùng regex)
    def has_duplicate_word(s):
        words = s.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:  # ví dụ: Senior Senior
                return True
        return False

    anomalies["duplicate_words"] = titles.apply(has_duplicate_word)

    # Lọc các dòng có bất thường
    flag_cols = [
        "has_weird_chars", "multiple_spaces", "all_caps", "all_lower",
        "too_short", "too_long", "contains_number", "duplicate_words"
    ]

    anomaly_rows = anomalies[anomalies[flag_cols].any(axis=1)]

    print("=== Số dòng có dấu hiệu bất thường:", len(anomaly_rows))

    return anomaly_rows

def check_remote_anomalies(df):
    """
    Kiểm tra trường hợp employee_residence khác company_location
    nhưng remote_ratio = 0.
    
    Đây là dấu hiệu bất thường vì khác quốc gia nhưng không làm remote.
    """
    anomalies = df[
        (df["employee_residence"] != df["company_location"]) &
        (df["remote_ratio"] == 0)
    ]
    print("=== Số dòng bất thường (khác quốc gia nhưng remote_ratio = 0): ", len(anomalies))
    print(anomalies)
    
    return anomalies
