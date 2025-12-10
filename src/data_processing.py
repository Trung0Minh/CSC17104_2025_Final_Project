import pandas as pd
import numpy as np
import country_converter as coco
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """Loads the dataset from the specified filepath."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def get_dataset_overview(df):
    """Returns basic overview statistics of the dataset."""
    print("Dataset Overview:")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Total size: {df.size} elements")
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    empty_rows = df.isnull().all(axis=1).sum()
    print(f"Empty rows: {empty_rows}")
    
    return df.head()


def get_column_info(df):
    """Prints information about columns including data types and non-null counts."""
    print("\nColumn Information:")
    print(df.info())
    return df.dtypes


def get_numerical_columns(df):
    """Returns a list of numerical column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def analyze_numerical_column_metrics(df, col):
    """
    Analyzes a single numerical column based on specific criteria:
    - Distribution & Central Tendency
    - Range & Outliers
    - Data Quality
    """
    print(f"--- Metrics for: {col} ---")
    
    # 1. Distribution & Central Tendency
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_dev = df[col].std()
    skewness = df[col].skew()
    
    print(f"\n[1] Distribution & Central Tendency:")
    print(f"   - Mean: {mean_val:.2f}")
    print(f"   - Median: {median_val:.2f}")
    print(f"   - Std Dev: {std_dev:.2f}")
    print(f"   - Skewness: {skewness:.2f} (0 = normal, >0 = right-skewed, <0 = left-skewed)")

    # 2. Range & Outliers
    min_val = df[col].min()
    max_val = df[col].max()
    
    # IQR Method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    num_outliers = outliers.count()
    
    print(f"\n[2] Range & Outliers:")
    print(f"   - Min: {min_val}")
    print(f"   - Max: {max_val}")
    print(f"   - IQR: {IQR:.2f} (Q1={Q1:.2f}, Q3={Q3:.2f})")
    print(f"   - Outlier Boundaries: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   - Outlier Count: {num_outliers} ({num_outliers/len(df)*100:.2f}%)")

    # 3. Data Quality
    missing_count = df[col].isnull().sum()
    total_rows = len(df)
    # Check for unique values to identify potential impossible values
    unique_values = sorted(df[col].dropna().unique())
    
    print(f"\n[3] Data Quality:")
    print(f"   - Missing Values: {missing_count} ({missing_count/total_rows*100:.2f}%)")
    if len(unique_values) <= 10:
        print(f"   - Unique Values ({len(unique_values)}): {unique_values}")
    
    print("-" * 40)


def group_job_title(job_title):
    """
    Gom nhóm các chức danh công việc thành 6 nhóm chính.
    """
    job_title = str(job_title).lower()
    if any(x in job_title for x in ['manager', 'head', 'lead', 'director', 'principal', 'vp', 'chief']):
        return 'Manager/Lead'
    elif any(x in job_title for x in ['scientist', 'researcher']):
        return 'Data Scientist'
    elif any(x in job_title for x in ['engineer', 'architect']):
        return 'Data Engineer'
    elif 'analyst' in job_title:
        return 'Data Analyst'
    elif any(x in job_title for x in ['machine learning', 'ml', 'ai', 'computer vision', 'nlp']):
        return 'ML/AI Engineer'
    else:
        return 'Other'


def clean_data_pipeline(df):
    """
    Pipeline thực hiện các bước làm sạch cơ bản.
    """
    # 1. Tạo cột job_category
    df['job_category'] = df['job_title'].apply(group_job_title)
    
    # 2. Mapping các giá trị viết tắt
    df['experience_level'] = df['experience_level'].replace({
        'SE': 'Senior Level',
        'EN': 'Entry Level',
        'EX': 'Executive Level',
        'MI': 'Mid Level'
    })
    
    df['employment_type'] = df['employment_type'].replace({
        'FL': 'Freelance',
        'CT': 'Contractor',
        'FT': 'Full-time',
        'PT': 'Part-time'
    })
    
    df['company_size'] = df['company_size'].replace({
        'S': 'Small',
        'M': 'Medium',
        'L': 'Large'
    })

    # 3. Chuẩn hóa cột company_location và employee_residence sang tên quốc gia đầy đủ (ISO 3166 -> tên quốc gia)
    cc = coco.CountryConverter()
    df['company_location'] = cc.convert(names=df['company_location'], to='name_short')
    df['employee_residence'] = cc.convert(names=df['employee_residence'], to='name_short')
    
    return df


def adjust_salary_inflation(df):
    # Tỷ lệ lạm phát tham khảo (US CPI)
    inflation_rates = {2020: 0.0123, 2021: 0.0470, 2022: 0.0650, 2023: 0.0}
    
    def calculate_adjusted(row):
        year = row['work_year']
        salary = row['salary_in_usd']
        if year == 2023: return salary
        
        # Tính gộp lạm phát
        adjusted = salary
        for y in range(year, 2023):
            adjusted *= (1 + inflation_rates.get(y, 0))
        return adjusted

    df['adjusted_salary'] = df.apply(calculate_adjusted, axis=1)
    return df

def group_location(country):
    # 1. Nhóm Mỹ (Dominant)
    if country in ['United States', 'US']:
        return 'US'
    
    # 2. Nhóm các nước phát triển (Tier 2) 
    # Canada, UK, Đức, Pháp, Úc, Sing, Ireland...
    elif country in ['Canada', 'United Kingdom', 'Germany', 'France', 'Australia', 
                     'Netherlands', 'Ireland', 'Singapore', 'Sweden', 'Japan']:
        return 'Other_Developed'
    
    # 3. Nhóm còn lại (India, Brazil, Spain, và các nước mẫu ít lương cao như Israel...)
    else:
        return 'Rest_of_World'
    
def prepare_data_for_model(df):
    """
    Chuẩn bị dữ liệu cho Model:
    1. Chọn các features quan trọng.
    2. Loại bỏ outliers (nếu cần thiết, ở đây ta giữ lại để model học được cả lương cao).
    3. Mã hóa dữ liệu (Encoding).
    """
    # 1. Chọn features
    features = ['experience_level', 'employment_type', 'job_category', 
                'employee_residence', 'remote_ratio', 'company_location', 'company_size']
    
    # SỬ DỤNG 'adjusted_salary' LÀM TARGET
    target = 'adjusted_salary'
    
    df_model = df[features + [target]].copy()
    
    # 2. Encoding
    label_encoders = {}
    for col in features:
        if df_model[col].dtype == 'object' or col == 'remote_ratio': # remote_ratio cũng nên coi là category
            le = LabelEncoder()
            df_model[col] = df_model[col].astype(str)
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le
            
    return df_model, label_encoders

def prepare_data_for_model(df):
    """
    Chuẩn bị dữ liệu cho Model:
    1. Xử lý outliers
    2. Feature Engineering (Gom nhóm Location)
    3. Chọn features quan trọng
    4. Mã hóa dữ liệu (Encoding)
    """
    df_ml = df.copy()
    
    # 1. Xử lý Outliers bằng IQR (Chỉ lọc trên biến mục tiêu adjusted_salary)
    # Lọc bớt các lương quá cao gây nhiễu
    Q1 = df_ml['adjusted_salary'].quantile(0.25)
    Q3 = df_ml['adjusted_salary'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR # Thực tế lương ko âm, nhưng cứ giữ công thức chuẩn
    
    original_len = len(df_ml)
    df_ml = df_ml[(df_ml['adjusted_salary'] >= lower_bound) & (df_ml['adjusted_salary'] <= upper_bound)]
    print(f"Đã loại bỏ {original_len - len(df_ml)} dòng outliers (adjusted_salary > {upper_bound:,.0f})")

    # 2. Feature Engineering: 
    # 2.1 Gom nhóm Location
    # Giảm số lượng category từ 70+ xuống còn 3 nhóm để tránh nhiễu
    df_ml['company_location_group'] = df_ml['company_location'].apply(group_location)
    df_ml['employee_residence_group'] = df_ml['employee_residence'].apply(group_location)
    
    # 3. Chọn features
    # Bỏ 'work_year' (đã chỉnh lạm phát), bỏ 'company_location' gốc (đã có group)
    features = ['experience_level', 'employment_type', 'job_category', 
                'employee_residence_group', 'remote_ratio', 'company_location_group', 'company_size']
    target = 'adjusted_salary' # SỬ DỤNG 'adjusted_salary' LÀM TARGET
    
    df_final = df_ml[features + [target]].copy()
    
    # 4. Encoding
    label_encoders = {}
    for col in features:    
        if df_final[col].dtype == 'object' or col == 'remote_ratio': # remote_ratio cũng nên coi là category
            le = LabelEncoder()
            df_final[col] = df_final[col].astype(str)
            df_final[col] = le.fit_transform(df_final[col])
            label_encoders[col] = le
            
    return df_final, label_encoders