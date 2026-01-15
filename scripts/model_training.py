# scripts/model_training.py
import os
import timeimport numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_preprocessing import remove_duplicates
from data_preprocessing import handle_missing_values
from data_preprocessing import DetectOutliersIQR
from data_preprocessing import encode_categorical_variables
from data_preprocessing import check_data_consistency
from results_analysis import plot_learning_curve


# =========================
# 3.1 Model training functions
# =========================

def get_models(random_state=42):
    """
    Return all models to be trained (without hyperparameters)
    """
    return {
        "random_forest": RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1
        ),
        "svm": SVR(),
        "mlp": MLPRegressor(
            max_iter=500,
            random_state=random_state
        ),
        "gaussian_process": Pipeline([
            ("scaler", StandardScaler()),
            ("gp", GaussianProcessRegressor(
                kernel=RBF(length_scale=1) + WhiteKernel(noise_level_bounds=(1e-2, 1)),
                random_state=42,
            ))
        ])
    }

param_grids = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [15, 20],
        "min_samples_split": [2, 5]
    },

    "svm": {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 0.2],
        "gamma": ["scale", "auto"]
    },

    "mlp": {
        "hidden_layer_sizes": [(64, 32), (128, 64)],
        "activation": ["relu"],
        "alpha": [1e-4, 1e-3]
    },

    "gaussian_process": {
        "gp__alpha": [1e-8],
        "gp__normalize_y":[True]
    }
}

def tune_model(
    model,
    param_grid,
    X_train,
    y_train,
    search_type="grid",
    cv=5,
    n_iter=20,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42
):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
    """

    if search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )

    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state
        )

    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_

# =========================
# 3.2 Model evaluation
# =========================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    Returns: RMSE, MAE, R2
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rmse, mae, r2

def evaluate_and_print(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the model and print results to console (keep 4 decimals)
    """
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)

    print(f"\n===== {model_name} Evaluation =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

    return rmse, mae, r2

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

    check_data_consistency(data)

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
    # 5️⃣ Final check
    # ======================
    print("\n===== Post-preprocessing Data Check =====")
    check_data_consistency(X_train)
    check_data_consistency(X_val)
    check_data_consistency(X_test)

    print("✅ Data preprocessing completed (target y was not modified)")

    # Model training
    best_models = {}        # ✅ store best models
    best_params_all = {}    # (optional) store best parameters
    best_scores_all = {}    # (optional) store best CV scores

    models = get_models()

    for name, model in models.items():
        print(f"\n===== Tuning model: {name} =====")

        best_model, best_params, best_score = tune_model(
            model=model,
            param_grid=param_grids[name],
            X_train=X_train,
            y_train=y_train,
            search_type="grid",   # or "random"
            cv=5
        )

        # ✅ core: save best model
        best_models[name] = best_model

        # ✅ optional: save tuning info
        best_params_all[name] = best_params
        best_scores_all[name] = best_score

        print("Best parameters:", best_params)
        print("Best CV score:", best_score)

        print("Best parameters:", best_params)
        print("Best CV score:", best_score)
        results = {}

    results = {}
    save_dir = "figures/learning_curves"
    os.makedirs(save_dir, exist_ok=True)

    for name, model in best_models.items():
        # -----------------------------
        # Model evaluation
        # -----------------------------
        rmse, mae, r2 = evaluate_and_print(
            model,
            X_val,
            y_val,
            model_name=name
        )

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        # -----------------------------
        # Plot and save learning curves
        # -----------------------------
        save_path = os.path.join(save_dir, f"{name}_learning_curve.png")

        plot_learning_curve(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model=model,
            mode="auto",
            metric="rmse",
            save_path=save_path
        )

        print(f"Learning curve saved to: {save_path}")

            # ======================
    # 6️⃣ Test set prediction & save results
    # ======================
    results_table_dir = "results/table"
    os.makedirs(results_table_dir, exist_ok=True)

    for name, model in best_models.items():
        print(f"\n===== Test prediction: {name} =====")

        # Predict on test set
        y_test_pred = model.predict(X_test)

        # Create DataFrame
        df_pred = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_test_pred
        })

        # Save to CSV
        save_path = os.path.join(
            results_table_dir,
            f"{name}_test_predictions.csv"
        )
        df_pred.to_csv(save_path, index=False)

        print(f"Test predictions saved to: {save_path}")
    
        # ======================
    # 7️⃣ Save best models
    # ======================
    models_save_dir = "results/models"
    os.makedirs(models_save_dir, exist_ok=True)

    for name, model in best_models.items():
        model_path = os.path.join(
            models_save_dir,
            f"{name}.joblib"
        )

        joblib.dump(model, model_path)

        print(f"✅ Model saved: {model_path}")
        # ======================
    # ⏱️ Model training & inference time evaluation
    # ======================
    time_results = {}

    for name, model in best_models.items():
        print(f"\n===== Time evaluation: {name} =====")

        # -------- Training time (fit on full training set) --------
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train

        # -------- Inference time (predict on test set) --------
        start_infer = time.perf_counter()
        _ = model.predict(X_test)
        infer_time = time.perf_counter() - start_infer

        time_results[name] = {
            "Training Time (s)": train_time,
            "Inference Time (s)": infer_time
        }

        print(f"Training time : {train_time:.4f} s")
        print(f"Inference time: {infer_time:.4f} s")

            # ======================
    # Save time evaluation results
    # ======================
    time_df = pd.DataFrame.from_dict(time_results, orient="index")
    time_df.index.name = "Model"

    time_save_path = os.path.join(results_table_dir, "model_time_evaluation.csv")
    time_df.to_csv(time_save_path)

    print(f"Model time evaluation saved to: {time_save_path}")

    # ======================
    # Merge performance metrics with time evaluation
    # ======================
    perf_path = "results/table/test_performance_summary.csv"
    time_path = "results/table/model_time_evaluation.csv"

    # Load CSVs
    df_perf = pd.read_csv(perf_path)
    df_time = pd.read_csv(time_path)

    # Rename time columns to match desired output
    df_time = df_time.rename(columns={
        "Training Time (s)": "Training_Time",
        "Inference Time (s)": "Inference_Time"
    })

    # Merge on model name
    df_merged = pd.merge(
        df_perf,
        df_time,
        on="Model",
        how="inner"
    )

    # (Optional) Pretty model names for presentation
    model_name_map = {
        "random_forest": "RandomForest",
        "svm": "SVR",
        "mlp": "MLP",
        "gaussian_process": "GPR"
    }
    df_merged["Model"] = df_merged["Model"].map(
        lambda x: model_name_map.get(x, x)
    )

    # Save back to the same CSV
    df_merged.to_csv(perf_path, index=False)

    print("test_performance_summary.csv updated with training & inference time")


if __name__ == "__main__":
    main()