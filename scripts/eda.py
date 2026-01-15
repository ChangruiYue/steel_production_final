# scripts/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def basic_info(df):
    """
    Output basic dataset information
    """
    print("========== Basic Dataset Information ==========")
    print(df.info())
    print("\n========== Descriptive Statistics ==========")
    print(df.describe())


def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def plot_target_distribution(y, save_dir):
    """
    Plot target variable distribution (training set only)
    """
    plt.figure(figsize=(10, 8))
    sns.histplot(y, kde=True)
    plt.title("Target Variable Distribution")
    plt.xlabel("output")
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "target_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_distributions(df, save_dir):
    """
    Plot histograms for all numeric features (excluding target)
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] == 0:
        print("No numeric features found for plotting")
        return

    numeric_df.hist(
        figsize=(12, 8),
        bins=20,
        edgecolor="black"
    )
    plt.suptitle("Feature Distributions")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "feature_distributions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_matrix(df, save_dir):
    """
    Plot feature correlation matrix heatmap
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_boxplots_in_batches(df, columns, save_dir, batch_size=7):
    """
    Plot boxplots in batches for outlier detection
    """
    for i in range(0, len(columns), batch_size):
        subset = columns[i:i + batch_size]

        plt.figure(figsize=(10, 5))
        df[subset].boxplot()
        plt.title(f"Boxplot for Outlier Detection ({i+1}-{i+len(subset)})")
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"boxplot_batch_{i+1}_{i+len(subset)}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_pair_plot(df, columns, save_dir, hue=None, filename="pairplot.png"):
    """
    Plot pairwise feature relationships (pair plot)
    """
    sns.pairplot(
        df[columns + ([hue] if hue else [])],
        hue=hue,
        diag_kind="hist",
        plot_kws={"alpha": 0.4, "s": 18},
    )

    plt.suptitle("Pair Plot of Selected Features", y=1.02)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # ======================
    # Paths and parameters
    # ======================
    data_train_path = "data/normalized_train_data.csv"
    data_test_path = "data/normalized_test_data.csv"

    # ======================
    # Load data
    # ======================
    df_train = pd.read_csv(data_train_path)
    df_test = pd.read_csv(data_test_path)

    data = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    print("Raw data loaded successfully")
    TARGET_COL = "output"
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

     # ======================
    # Training set EDA
    # ======================
    print("\n========== Training Set EDA ==========")
    basic_info(X_train)

    FIG_DIR = "figures/eda"
    ensure_dir(FIG_DIR)

    # 1️⃣ Target variable distribution
    plot_target_distribution(y_train, FIG_DIR)

    # 2️⃣ Feature distributions
    plot_feature_distributions(X_train, FIG_DIR)

    # 3️⃣ Correlation matrix
    train_set = pd.concat([X_train, y_train], axis=1)
    plot_correlation_matrix(train_set, FIG_DIR)

    # 4️⃣ Boxplots (outlier detection)
    input_columns = X_train.columns.tolist()
    plot_boxplots_in_batches(
        X_train,
        input_columns,
        save_dir=FIG_DIR,
        batch_size=7
    )

    # 5️⃣ Pair plot (selected features)
    plot_pair_plot(
        train_set,
        columns=["input1", "input2", "input3", "input4", "output"],
        save_dir=FIG_DIR,
        filename="pairplot_inputs_output.png"
    )