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
