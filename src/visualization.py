import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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

    # Add mean and median lines with legend
    mean_val = df[col].mean()
    median_val = df[col].median()
    axes[1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[1].axvline(median_val, color='blue', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[1].legend()

    axes[1].set_title(f'Boxplot of {col}')
    axes[1].set_xlabel(col)
    
    plt.tight_layout()
    plt.show()


def plot_bar_count(df, column_name, top_n=None, sort_by_index=False, title=None, figsize=(10, 6), palette="viridis", annotate=True, horizontal=False):
    """
    Vẽ biểu đồ thanh thể hiện tần suất xuất hiện của các giá trị trong cột phân loại.

    Tham số:
        df: DataFrame dữ liệu
        column_name: tên cột cần trực quan hóa
        top_n: nếu muốn chỉ hiển thị top N giá trị phổ biến nhất
        sort_by_index: Sắp xếp theo nhãn (index) thay vì tần suất
        title: Tiêu đề tùy chỉnh
        figsize: Kích thước biểu đồ
        palette: Bảng màu
        annotate: Hiển thị số lượng trên thanh
        horizontal: Vẽ biểu đồ ngang (thích hợp cho nhãn dài)
    """
    value_counts = df[column_name].value_counts()

    if top_n is not None:
        value_counts = value_counts.head(top_n)
    
    if sort_by_index:
        value_counts = value_counts.sort_index()

    plt.figure(figsize=figsize)
    
    if horizontal:
        ax = sns.barplot(y=value_counts.index, x=value_counts.values, hue=value_counts.index, palette=palette, legend=False)
        plt.xlabel("Số lượng")
        plt.ylabel(column_name)
    else:
        ax = sns.barplot(x=value_counts.index, y=value_counts.values, hue=value_counts.index, palette=palette, legend=False)
        plt.xlabel(column_name)
        plt.ylabel("Số lượng")
        plt.xticks(rotation=45, ha='right')

    # Xử lý tiêu đề
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f"Tần suất xuất hiện của cột: {column_name}", fontsize=14, fontweight='bold')

    # Annotate (Hiển thị số liệu trên cột)
    if annotate:
        if horizontal:
            for p in ax.patches:
                width = p.get_width()
                if width > 0:
                    ax.annotate(f'{int(width)}', (width, p.get_y() + p.get_height() / 2),
                                ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')
        else:
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                                ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')

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


def plot_side_by_side_bar(df, col1, col2, top_n=10, title="Comparison"):
    """
    Vẽ 2 biểu đồ cột cạnh nhau để so sánh phân phối (VD: Nơi ở nhân viên vs Vị trí công ty)
    """
    count1 = df[col1].value_counts().head(top_n)
    count2 = df[col2].value_counts().head(top_n)

    fig = go.Figure(data=[
        go.Bar(name='Employee Residence', x=count1.index, y=count1.values, marker_color='indianred'),
        go.Bar(name='Company Location', x=count2.index, y=count2.values, marker_color='lightsalmon')
    ])

    fig.update_layout(barmode='group', title=title, xaxis_tickangle=-45)
    fig.show()

def plot_treemap(df, col, title="Distribution"):
    """
    Vẽ Treemap cho biến phân loại
    """
    temp_df = df[col].value_counts().reset_index()
    temp_df.columns = [col, 'Count']
    
    fig = px.treemap(temp_df, path=[col], values='Count',
                     title=title, color='Count', color_continuous_scale='RdBu')
    fig.show()
