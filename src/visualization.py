import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_column_distribution(df, col):
    """
    Plots histogram with KDE and boxplot for a single numerical column.
    """
    # Set style
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histogram + KDE
    sns.histplot(data=df, x=col, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution of {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')
    
    # 2. Boxplot
    sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot of {col}')
    axes[1].set_xlabel(col)
    
    plt.tight_layout()
    plt.show()

def plot_bar_count(df, column_name, top_n=None):
    """
    Vẽ biểu đồ thanh thể hiện tần suất xuất hiện của các giá trị trong cột phân loại.

    Tham số:
        df: DataFrame dữ liệu
        column_name: tên cột cần trực quan hóa
        top_n: nếu muốn chỉ hiển thị top N giá trị phổ biến nhất (thường dùng cho cột nhiều nhãn)
    """
    value_counts = df[column_name].value_counts()

    # Nếu cột có quá nhiều giá trị → cho phép chỉ lấy top_n
    if top_n is not None:
        value_counts = value_counts.head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f"Tần suất xuất hiện của cột: {column_name}")
    plt.xlabel("Giá trị")
    plt.ylabel("Số lượng")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, title="Correlation Matrix (Numerical Columns)"):
    """
    Vẽ heatmap tương quan cho các cột số.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numerical columns found for correlation.")
        return

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool)) # Che một nửa trên để đỡ rối
    sns.heatmap(numeric_df.corr(), mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.show()

def plot_categorical_vs_numerical_box(df, cat_col, num_col, order=None, title=None):
    """
    Vẽ Boxplot để xem phân phối của biến số theo nhóm phân loại.
    Rất quan trọng để xem Salary biến động thế nào theo Experience, Company Size...
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=cat_col, y=num_col, order=order, hue=cat_col, palette='viridis', showfliers=False, legend=False) # showfliers=False để ẩn outliers quá xa cho hình gọn
    sns.stripplot(data=df, x=cat_col, y=num_col, order=order, color='black', alpha=0.3, size=3) # Hiện thêm các điểm dữ liệu mờ
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'Relationship between {cat_col} and {num_col}', fontsize=14, fontweight='bold')
    
    plt.ylabel(num_col)
    plt.xlabel(cat_col)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_categorical_heatmap(df, col1, col2, title=None):
    """
    Vẽ Heatmap tần suất xuất hiện giữa 2 biến phân loại (Crosstab).
    VD: Experience Level vs Company Size.
    """
    crosstab = pd.crosstab(df[col1], df[col2])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu')
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'Heatmap: {col1} vs {col2}', fontsize=14, fontweight='bold')
    plt.show()
