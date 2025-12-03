import matplotlib.pyplot as plt
import seaborn as sns

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
