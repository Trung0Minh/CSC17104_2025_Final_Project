import pandas as pd
import numpy as np
import country_converter as coco


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


def assign_job_category(job_title):
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
    df['job_category'] = df['job_title'].apply(assign_job_category)
    
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