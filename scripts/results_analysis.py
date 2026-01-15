# scripts/05_results_analysis.py
import glob
import time
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from data_preprocessing import remove_duplicates
from data_preprocessing import handle_missing_values
from data_preprocessing import DetectOutliersIQR
from data_preprocessing import encode_categorical_variables

# =========================
# Metric calculation
# =========================

def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics
    Returns RMSE, MAE, R2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# =========================
# Performance summary table
# =========================

def create_performance_table(results):
    """
    Create comparative performance table
    """
    df = pd.DataFrame(results)
    print("\n========== Model Performance Comparison ==========")
    print(df)
    return df


# =========================
# Visualization analysis
# =========================

def plot_model_comparison(results):
    """Create bar plots with error bars"""
    metrics = ['RMSE', 'MAE', 'R2']
    metric_names = {
        "RMSE": "RMSE",
        "MAE": "MAE",
        "R2": r"$R^2$",
    }
    models = list(results.keys())
    x = np.arange(len(models))
    plt.figure(figsize=(10, 8))
    bar_width = 0.15
    for i, metric in enumerate(metrics):
        means = []
        stds = []
        for model in models:
            values = results[model][metric]
            means.append(np.mean(values))
            stds.append(np.std(values))

        plt.bar(
            x + i * bar_width,
            means,
            width=bar_width,
            yerr=stds,
            capsize=4,
            label=metric_names[metric],
            alpha=0.85
        )
    plt.xticks(x + bar_width * (len(metrics) - 1) / 2, models, rotation=30)
    plt.ylabel("Metric Value")
    plt.title("Model Performance Comparison with Error Bars")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()


def plot_predictions_vs_actual(y_true, y_pred, model_name):
    """
    Scatter plot of predictions vs actual values
    """
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--",
        color="red"
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: Prediction vs Actual")
    plt.tight_layout()
 


def plot_residuals(y_true, y_pred, model_name):
    """
    Plot residual analysis
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, residuals)
    plt.axhline(0, linestyle="--", color="r")
    plt.title(f"{model_name}: Residual Distribution")
    plt.xlabel("Residual")
    plt.tight_layout()
 


def plot_learning_curve(
    X_train,
    y_train,
    X_val,
    y_val,
    model,
    mode="auto",
    train_sizes=np.linspace(0.1, 1.0, 5),
    metric="rmse",
    random_state=42,
    save_path=None
):
    """
    Plot learning curves for regression models (data size or iterations)

    Parameters
    ----------
    X_train, y_train : training set
    X_val, y_val     : validation set
    model            : tuned optimal model (sklearn regressor)
    mode             : 'auto' | 'data' | 'iter'
    train_sizes      : proportions of training data
    metric           : 'r2' | 'rmse' | 'mae'
    random_state     : random seed
    """

    def score(y_true, y_pred):
        if metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError("Unsupported metric")

    # ----------------------------
    # Automatically determine learning curve type
    # ----------------------------
    if mode == "auto":
        if hasattr(model, 'loss_curve_') :
            mode = "iter"
        else:
            mode = "data"

    plt.figure(figsize=(8, 6))

    # ======================================================
    # Data size learning curve
    # ======================================================
    if mode == "data":
        train_scores = []
        val_scores = []

        X_train_shuffled, y_train_shuffled = shuffle(
            X_train, y_train, random_state=random_state
        )

        n_samples = X_train.shape[0]

        for frac in train_sizes:
            size = int(n_samples * frac)

            X_sub = X_train_shuffled[:size]
            y_sub = y_train_shuffled[:size]

            m = clone(model)
            m.fit(X_sub, y_sub)

            train_scores.append(score(y_sub, m.predict(X_sub)))
            val_scores.append(score(y_val, m.predict(X_val)))

        plt.plot(train_sizes * n_samples, train_scores, "o-", label="Train")
        plt.plot(train_sizes * n_samples, val_scores, "o-", label="Validation")

        plt.xlabel("Training samples")
        plt.ylabel(metric.upper())
        plt.title("Data Size Learning Curve")

    # ======================================================
    # Iterative learning curve
    # ======================================================
    elif mode == "iter":
        train_scores = []
        val_scores = []

        # Determine iteration parameter
        if hasattr(model, "n_estimators"):
            iter_param = "n_estimators"
            iters = np.linspace(10, model.n_estimators, 5, dtype=int)
        elif hasattr(model, "max_iter"):
            iter_param = "max_iter"
            iters = np.linspace(10, model.max_iter, 5, dtype=int)
        else:
            raise ValueError("Model does not support iterative learning curve")

        for i in iters:
            m = clone(model)
            setattr(m, iter_param, i)
            m.fit(X_train, y_train)

            train_scores.append(score(y_train, m.predict(X_train)))
            val_scores.append(score(y_val, m.predict(X_val)))

        plt.plot(iters, train_scores, "o-", label="Train")
        plt.plot(iters, val_scores, "o-", label="Validation")

        plt.xlabel(iter_param)
        plt.ylabel(metric.upper())
        plt.title("Iterative Learning Curve")

    else:
        raise ValueError("mode must be 'auto', 'data', or 'iter'")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def repeated_test_evaluation(
    best_model,
    X_train,
    y_train,
    X_test,
    y_test,
    n_runs=5,
    random_seed_start=0
):
    """
    Repeatedly train and evaluate a model using cloned instances,
    while keeping the test set fixed.

    Parameters
    ----------
    base_model : sklearn estimator (already instantiated)
        Base model with chosen hyperparameters.
    n_runs : int
        Number of repeated runs with different random seeds.

    Returns
    -------
    dict: metric_name -> list of values
    """

    results = {
        "RMSE": [],
        "MAE": [],
        "R2": [],
        "Training Time": [],
        "Inference Time": []
    }

    for i in range(n_runs):
        seed = random_seed_start + i

        # ---- Clone model to avoid parameter carry-over ----
        model = clone(best_model)

        # ---- Set random state if supported ----
        if "random_state" in model.get_params():
            model.set_params(random_state=seed)

        # ---- Measure training time (TRAIN SET) ----
        model.fit(X_train, y_train)

        # ---- Measure inference time (TEST SET) ----
        y_pred = model.predict(X_test)
    

        # ---- Performance metrics (TEST SET) ----
        rmse, mae, r2 = calculate_metrics(y_test, y_pred)

        results["RMSE"].append(rmse)
        results["MAE"].append(mae)
        results["R2"].append(r2)

    return results

def main():

    TARGET_COL = "output"   # ✅ target column name
    # ======================
    # Data paths
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


    # ======================
    # ✅ Split X / y (very important)
    # ======================
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    # ======================
    # Data split: 70% / 15% / 15%
    # ======================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # ======================
    # 1️⃣ Remove duplicates (✅ only on X, y aligned automatically by index)
    # ======================
    X_train = remove_duplicates(X_train)
    X_val = remove_duplicates(X_val)
    X_test = remove_duplicates(X_test)

    # Synchronize y (avoid index mismatch)
    y_train = y_train.loc[X_train.index]
    y_val = y_val.loc[X_val.index]
    y_test = y_test.loc[X_test.index]

    # ======================
    # 2️⃣ Handle missing values (✅ only on X)
    # ======================
    X_train = handle_missing_values(X_train, strategy="median")
    X_val = handle_missing_values(X_val, strategy="median")
    X_test = handle_missing_values(X_test, strategy="median")

    # ======================
    # 3️⃣ Outlier handling (✅ fit only on X_train)
    # ❌ y is NOT involved in outlier handling
    # ======================
    outlier_transformer = DetectOutliersIQR(factor=1.5)

    X_train = outlier_transformer.fit_transform(X_train)
    X_val = outlier_transformer.transform(X_val)
    X_test = outlier_transformer.transform(X_test)

    print("Outlier handling completed (applied only to features X)")

    # ======================
    # 4️⃣ Categorical variable encoding (✅ only on X)
    # ======================
    combined_X = pd.concat([X_train, X_val, X_test], axis=0)

    combined_X = encode_categorical_variables(combined_X)

    # Split back
    X_train = combined_X.loc[X_train.index]
    X_val = combined_X.loc[X_val.index]
    X_test = combined_X.loc[X_test.index]

    # ======================
    # Paths
    # ======================
    results_table_dir = "results/table"
    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # ======================
    # Load all prediction CSV files
    # ======================
    csv_files = glob.glob(
        os.path.join(results_table_dir, "*_test_predictions.csv")
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No prediction CSV files found in results/table/"
        )

    print("\n===== Loading test prediction results =====")

    results_summary = []

    # ======================
    # Loop over models
    # ======================
    for csv_path in csv_files:
        model_name = os.path.basename(csv_path).replace(
            "_test_predictions.csv", ""
        )

        print(f"\n--- Evaluating model: {model_name} ---")

        df = pd.read_csv(csv_path)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        # ======================
        # Metrics
        # ======================
        rmse, mae, r2 = calculate_metrics(y_true, y_pred)

        results_summary.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R2  : {r2:.4f}")

        # ======================
        # 1️⃣ Prediction vs Actual
        # ======================
        plot_predictions_vs_actual(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name
        )
        plt.savefig(
            os.path.join(figures_dir, f"{model_name}_pred_vs_true.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ======================
        # 2️⃣ Residuals
        # ======================
        plot_residuals(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name
        )
        plt.savefig(
            os.path.join(figures_dir, f"{model_name}_residuals.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    # ======================
    # 3️⃣ Model comparison bar chart
    # ======================
    df_summary = pd.DataFrame(results_summary)

    # Save summary table
    df_summary.to_csv(
        os.path.join(results_table_dir, "test_performance_summary.csv"),
        index=False
    )

    rf_final  = joblib.load("results/models/random_forest.joblib")
    svr_final = joblib.load("results/models/svm.joblib")
    mlp_final = joblib.load("results/models/mlp.joblib")
    gpr_final = joblib.load("results/models/gaussian_process.joblib")

    final_models = {
            "RandomForest": rf_final,
            "SVR": svr_final,
            "MLP": mlp_final,
            "GPR": gpr_final,
        }
    final_results = {
        name: repeated_test_evaluation(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        for name, model in final_models.items()
    }
    plot_model_comparison(final_results)
    plt.savefig("results/figures/Model Performance Comparison.png", dpi=600)

    print("\n✅ Test set evaluation completed.")
    print("Figures saved to:", figures_dir)



if __name__ == "__main__":
    main()