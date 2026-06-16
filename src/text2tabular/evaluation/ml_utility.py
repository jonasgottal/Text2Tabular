import pandas as pd
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
)

from .evaluation_utils import apply_mask_to_true_data

# Assuming GROUP_MAPPING might be needed if dataset_name specific logic for target selection becomes complex
# from .test_data_utils import GROUP_MAPPING # If needed in _get_target_and_task_type directly


def _get_target_and_task_type(
    data_df: pd.DataFrame,
    common_cols: List[str],
    mask: Optional[Dict],
    is_masked: bool,
    dataset_name: Optional[str],
    dataset_target_mapping: Optional[Dict[str, str]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Identifies the target variable and task type (classification/regression).
    Returns: (target_column, task_type, error_message)
    """
    target_column = None
    task_type = None
    error_message = None

    # 1. Try dataset_target_mapping first
    if (
        dataset_name
        and dataset_target_mapping
        and dataset_name in dataset_target_mapping
    ):
        target_column_from_map = str(
            dataset_target_mapping[dataset_name]
        ).lower()
        if (
            target_column_from_map in data_df.columns
            and target_column_from_map in common_cols
        ):
            target_column = target_column_from_map
        else:
            error_message = (
                f"Target '{target_column_from_map}' from dataset_target_mapping for '{dataset_name}' "
                f"not found in common/processed columns."
            )
            if target_column is None:
                return None, None, error_message

    # 2. Attempt to get from mask if target not found by mapping
    if (
        not target_column
        and is_masked
        and mask
        and isinstance(mask.get("ml_setup"), dict)
    ):
        target_from_mask = mask["ml_setup"].get("target_variable")
        task_type_from_mask = mask["ml_setup"].get("task_type")
        if target_from_mask:
            target_from_mask = str(target_from_mask).lower()
            if (
                target_from_mask in data_df.columns
                and target_from_mask in common_cols
            ):
                target_column = target_from_mask
                if task_type_from_mask and task_type_from_mask in [
                    "classification",
                    "regression",
                ]:
                    task_type = task_type_from_mask
            else:
                error_message = f"Target variable '{target_from_mask}' from mask not found in processed data columns."
        if task_type_from_mask and task_type_from_mask not in [
            "classification",
            "regression",
        ]:
            error_message = f"Invalid task_type '{task_type_from_mask}' from mask. Must be 'classification' or 'regression'."

    # 3. Infer if not found by mapping or mask
    if not target_column:
        potential_targets = [
            "group",
        ]
        for pt in potential_targets:
            if pt in common_cols:
                target_column = pt
                break

    if not target_column:
        for col in common_cols:
            if (
                data_df[col].dtype == "object"
                or data_df[col].dtype.name == "category"
                or pd.api.types.is_bool_dtype(data_df[col])
            ):
                if (
                    2 <= data_df[col].nunique() < 15
                ):  # Heuristic for categorical target
                    target_column = col
                    break
        if (
            not target_column
        ):  # Fallback to numeric if no clear categorical found
            for col in common_cols:
                if pd.api.types.is_numeric_dtype(data_df[col]):
                    if (
                        not any(
                            sub in col for sub in ["id", "index", "unnamed"]
                        )
                        or len(common_cols) == 1
                    ):
                        target_column = col
                        break

    if not target_column:
        error_message = "Could not identify a suitable target variable via mapping, mask, or inference."
        return None, None, error_message

    if not task_type:  # Determine task_type if not set from mask or mapping
        y_series = data_df[target_column].dropna()
        if y_series.empty:
            error_message = f"Target column '{target_column}' is all NaNs."
            return target_column, None, error_message

        y_type_val = type_of_target(y_series)
        if y_type_val in ["binary", "multiclass"]:
            task_type = "classification"
        elif y_type_val == "continuous":
            if (
                pd.api.types.is_integer_dtype(y_series)
                and y_series.nunique()
                < 15  # Heuristic for integer-coded categories
            ):
                task_type = "classification"
            else:
                task_type = "regression"
        elif y_type_val == "multilabel-indicator":
            task_type = (
                "classification"  # Or handle as a special case if needed
            )
        else:
            error_message = f"Could not determine task type for target '{target_column}' (type: {y_type_val})."
            return target_column, None, error_message

    if target_column and task_type:  # Clear error if successful
        error_message = None
    return target_column, task_type, error_message


def _calculate_feature_importance_comparison(
    true_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    task_type: str,
    random_state: int = 42,
    top_k: int = 5,
) -> Dict:
    results: Dict[str, Any] = {"status": "Skipped"}
    if not features:
        results["message"] = "No features provided for importance calculation."
        return results

    X_true_raw = true_df[features].copy()
    y_true_raw = true_df[target_col].copy()
    true_valid_idx = y_true_raw.dropna().index
    X_true, y_true = (
        X_true_raw.loc[true_valid_idx],
        y_true_raw.loc[true_valid_idx],
    )

    X_recon_raw = recon_df[features].copy()
    y_recon_raw = recon_df[target_col].copy()
    recon_valid_idx = y_recon_raw.dropna().index
    X_recon, y_recon = (
        X_recon_raw.loc[recon_valid_idx],
        y_recon_raw.loc[recon_valid_idx],
    )

    if X_true.empty or X_recon.empty or y_true.empty or y_recon.empty:
        results["message"] = (
            "Not enough data after dropping NaNs in target variable."
        )
        return results
    if task_type == "classification" and (
        len(y_true.unique()) < 2 or len(y_recon.unique()) < 2
    ):
        results["message"] = (
            "Target variable has less than 2 unique classes after NaN drop."
        )
        return results

    numerical_features = X_true.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical_features = X_true.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    try:
        X_combined = pd.concat([X_true, X_recon], axis=0)
        preprocessor.fit(X_combined)
        X_true_processed = preprocessor.transform(X_true)
        X_recon_processed = preprocessor.transform(X_recon)
        feature_names_out = preprocessor.get_feature_names_out()
    except Exception as e:
        results["message"] = f"Preprocessing error: {e}"
        return results

    if (
        X_true_processed.shape[0] == 0
        or X_recon_processed.shape[0] == 0
        or X_true_processed.shape[1] == 0
    ):
        results["message"] = "No data or features left after preprocessing."
        return results

    model_args = {
        "random_state": random_state,
        "n_estimators": 50,
        "max_depth": 10,
    }
    if task_type == "classification":
        model_true, model_recon = RandomForestClassifier(
            **model_args
        ), RandomForestClassifier(**model_args)
        if y_true.nunique() > 1:
            y_true = LabelEncoder().fit_transform(y_true)
        if y_recon.nunique() > 1:
            y_recon = LabelEncoder().fit_transform(y_recon)
    else:
        model_true, model_recon = RandomForestRegressor(
            **model_args
        ), RandomForestRegressor(**model_args)

    try:
        model_true.fit(X_true_processed, y_true)
        importances_true = model_true.feature_importances_
        model_recon.fit(X_recon_processed, y_recon)
        importances_recon = model_recon.feature_importances_
    except Exception as e:
        results["message"] = (
            f"Model training or importance extraction error: {e}"
        )
        return results

    if len(importances_true) != len(feature_names_out) or len(
        importances_recon
    ) != len(feature_names_out):
        results["message"] = (
            "Mismatch between number of importances and feature names."
        )
        if len(importances_true) == X_true_processed.shape[1]:  # Fallback
            feature_names_out = [
                f"feature_{i}" for i in range(X_true_processed.shape[1])
            ]
        else:
            return results

    importances_true_dict = dict(zip(feature_names_out, importances_true))
    importances_recon_dict = dict(zip(feature_names_out, importances_recon))
    sorted_true_features = sorted(
        importances_true_dict.items(), key=lambda item: item[1], reverse=True
    )
    sorted_recon_features = sorted(
        importances_recon_dict.items(), key=lambda item: item[1], reverse=True
    )

    results["true_feature_importances"] = {
        k: v for k, v in sorted_true_features
    }
    results["reconstructed_feature_importances"] = {
        k: v for k, v in sorted_recon_features
    }

    aligned_true_imp = [
        importances_true_dict.get(name, 0) for name in feature_names_out
    ]
    aligned_recon_imp = [
        importances_recon_dict.get(name, 0) for name in feature_names_out
    ]

    if len(aligned_true_imp) > 1 and len(aligned_recon_imp) > 1:
        spearman_corr, spearman_p_value = spearmanr(
            aligned_true_imp, aligned_recon_imp
        )
        results["spearman_rank_correlation"] = spearman_corr
        results["spearman_p_value"] = spearman_p_value
    else:
        results["spearman_rank_correlation"] = np.nan
        results["spearman_p_value"] = np.nan

    top_k_true = set([f[0] for f in sorted_true_features[:top_k]])
    top_k_recon = set([f[0] for f in sorted_recon_features[:top_k]])
    intersection_k = len(top_k_true.intersection(top_k_recon))
    union_k = len(top_k_true.union(top_k_recon))
    results[f"top_{top_k}_jaccard_similarity"] = (
        intersection_k / union_k if union_k > 0 else 0.0
    )
    results[f"top_{top_k}_features_true"] = list(top_k_true)
    results[f"top_{top_k}_features_reconstructed"] = list(top_k_recon)

    results["status"] = "Completed"
    results.pop("message", None)
    return results


def _perform_transfer_learning_evaluation(
    true_df_full: pd.DataFrame,
    recon_df_full: pd.DataFrame,
    features: List[str],
    target_col: str,
    task_type: str,
    random_state: int = 42,
    test_size: float = 0.3,
) -> Dict:
    results: Dict[str, Any] = {"status": "Skipped"}
    if not features:
        results["message"] = "No features provided for transfer learning."
        return results

    X_true_raw, y_true_raw = (
        true_df_full[features].copy(),
        true_df_full[target_col].copy(),
    )
    true_valid_idx = y_true_raw.dropna().index
    X_true_full, y_true_full_target = (
        X_true_raw.loc[true_valid_idx],
        y_true_raw.loc[true_valid_idx],
    )

    X_recon_raw, y_recon_raw = (
        recon_df_full[features].copy(),
        recon_df_full[target_col].copy(),
    )
    recon_valid_idx = y_recon_raw.dropna().index
    X_recon_full, y_recon_full_target = (
        X_recon_raw.loc[recon_valid_idx],
        y_recon_raw.loc[recon_valid_idx],
    )

    if X_true_full.empty or y_true_full_target.empty:
        results["message"] = "True data empty after NaN drop in target."
        return results
    if (
        X_recon_full.empty or y_recon_full_target.empty
    ):  # Recon data might be optional for True-True path
        results["recon_data_message"] = (
            "Reconstructed data empty after NaN drop in target. TrainRecon-TestTrue will be skipped."
        )

    y_true_for_split, y_recon_for_split = (
        y_true_full_target.copy(),
        y_recon_full_target.copy(),
    )
    num_classes = 0

    if task_type == "classification":
        combined_labels = (
            pd.concat([y_true_full_target, y_recon_full_target])
            .dropna()
            .unique()
        )
        num_classes = len(combined_labels)
        if num_classes < 2:
            results["message"] = (
                "Not enough unique classes in combined target for encoding."
            )
            return results
        le = LabelEncoder().fit(combined_labels)
        y_true_for_split = pd.Series(
            le.transform(y_true_full_target.dropna()),
            index=y_true_full_target.dropna().index,
        )
        if not y_recon_full_target.empty:
            y_recon_for_split = pd.Series(
                le.transform(y_recon_full_target.dropna()),
                index=y_recon_full_target.dropna().index,
            )

        if y_true_for_split.nunique() < 2:
            results["message"] = (
                "True target has < 2 unique classes after encoding."
            )
            return results

    try:
        X_true_train, X_true_test, y_true_train, y_true_test = (
            train_test_split(
                X_true_full,
                y_true_for_split,
                test_size=test_size,
                random_state=random_state,
                stratify=(
                    y_true_for_split
                    if task_type == "classification"
                    and y_true_for_split.nunique() > 1
                    else None
                ),
            )
        )
    except ValueError as e:
        results["message"] = f"Error splitting true data: {e}"
        return results

    X_recon_train, y_recon_train = pd.DataFrame(), pd.Series(dtype="float64")
    if (
        not X_recon_full.empty
        and not y_recon_for_split.empty
        and y_recon_for_split.nunique()
        >= (2 if task_type == "classification" else 1)
    ):
        try:
            X_recon_train, _, y_recon_train, _ = train_test_split(
                X_recon_full,
                y_recon_for_split,
                test_size=test_size,
                random_state=random_state,
                stratify=(
                    y_recon_for_split
                    if task_type == "classification"
                    and y_recon_for_split.nunique() > 1
                    else None
                ),
            )
        except ValueError as e:
            results["recon_split_message"] = f"Error splitting recon data: {e}"

    if (
        X_true_train.empty
        or X_true_test.empty
        or (
            task_type == "classification"
            and (y_true_train.nunique() < 2 or y_true_test.nunique() < 2)
        )
    ):
        results["message"] = "True train/test data insufficient after split."
        return results

    numerical_features = X_true_full.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical_features = X_true_full.select_dtypes(
        exclude=np.number
    ).columns.tolist()
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor_base = ColumnTransformer(
        transformers=[], remainder="passthrough"
    )
    if numerical_features:
        preprocessor_base.transformers.append(
            ("num", numerical_transformer, numerical_features)
        )
    if categorical_features:
        preprocessor_base.transformers.append(
            ("cat", categorical_transformer_pipeline, categorical_features)
        )

    if not numerical_features and not categorical_features:
        results["message"] = "No numerical or categorical features found."
        return results

    results_dict: Dict[str, Any] = {"status": "Partial"}
    model_args = {
        "random_state": random_state,
        "n_estimators": 50,
        "max_depth": 10,
    }

    # --- Train True, Test True ---
    try:
        preprocessor_true = copy.deepcopy(preprocessor_base)
        X_true_train_proc = preprocessor_true.fit_transform(X_true_train)
        X_true_test_proc = preprocessor_true.transform(X_true_test)

        if (
            X_true_train_proc.shape[0] == 0
            or X_true_test_proc.shape[0] == 0
            or X_true_train_proc.shape[1] == 0
        ):
            results_dict["train_true_test_true"] = {
                "status": "Skipped",
                "message": "No data after preprocessing for true-true.",
            }
        else:
            model_true = (
                RandomForestClassifier(**model_args)
                if task_type == "classification"
                else RandomForestRegressor(**model_args)
            )
            model_true.fit(X_true_train_proc, y_true_train)
            preds = model_true.predict(X_true_test_proc)
            metrics: Dict[str, Any] = {}
            if task_type == "classification":
                metrics["accuracy"] = accuracy_score(y_true_test, preds)
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true_test, preds, average="weighted", zero_division=0
                )
                metrics.update(
                    {
                        "precision_weighted": p,
                        "recall_weighted": r,
                        "f1_score_weighted": f1,
                    }
                )
                try:
                    proba = model_true.predict_proba(X_true_test_proc)
                    if num_classes == 2:
                        metrics["roc_auc_score"] = roc_auc_score(
                            y_true_test, proba[:, 1]
                        )
                    elif num_classes > 2:
                        metrics["roc_auc_score_ovr_weighted"] = roc_auc_score(
                            y_true_test,
                            proba,
                            multi_class="ovr",
                            average="weighted",
                        )
                except Exception as e_auc:
                    metrics["roc_auc_score_error"] = str(e_auc)
            else:  # Regression
                metrics["r2_score"] = r2_score(y_true_test, preds)
                metrics["mse"] = mean_squared_error(y_true_test, preds)
                metrics["mae"] = mean_absolute_error(y_true_test, preds)
            results_dict["train_true_test_true"] = {
                "status": "Completed",
                "metrics": metrics,
            }
    except Exception as e:
        results_dict["train_true_test_true"] = {
            "status": "Error",
            "message": f"True-True pipeline error: {e}",
        }

    # --- Train Recon, Test True ---
    if (
        X_recon_train.empty
        or y_recon_train.empty
        or (task_type == "classification" and y_recon_train.nunique() < 2)
    ):
        results_dict["train_recon_test_true"] = {
            "status": "Skipped",
            "message": "Recon training data insufficient.",
        }
    else:
        try:
            preprocessor_recon = copy.deepcopy(preprocessor_base)
            X_recon_train_proc = preprocessor_recon.fit_transform(
                X_recon_train
            )
            X_true_test_proc_for_recon_model = preprocessor_recon.transform(
                X_true_test
            )  # Use recon's preprocessor

            if (
                X_recon_train_proc.shape[0] == 0
                or X_true_test_proc_for_recon_model.shape[0] == 0
                or X_recon_train_proc.shape[1] == 0
            ):
                results_dict["train_recon_test_true"] = {
                    "status": "Skipped",
                    "message": "No data after preprocessing for recon-true.",
                }
            else:
                model_recon = (
                    RandomForestClassifier(**model_args)
                    if task_type == "classification"
                    else RandomForestRegressor(**model_args)
                )
                model_recon.fit(X_recon_train_proc, y_recon_train)
                preds = model_recon.predict(X_true_test_proc_for_recon_model)
                metrics_recon: Dict[str, Any] = {}
                if task_type == "classification":
                    metrics_recon["accuracy"] = accuracy_score(
                        y_true_test, preds
                    )
                    p, r, f1, _ = precision_recall_fscore_support(
                        y_true_test, preds, average="weighted", zero_division=0
                    )
                    metrics_recon.update(
                        {
                            "precision_weighted": p,
                            "recall_weighted": r,
                            "f1_score_weighted": f1,
                        }
                    )
                    try:
                        proba = model_recon.predict_proba(
                            X_true_test_proc_for_recon_model
                        )
                        if num_classes == 2:
                            metrics_recon["roc_auc_score"] = roc_auc_score(
                                y_true_test, proba[:, 1]
                            )
                        elif num_classes > 2:
                            metrics_recon["roc_auc_score_ovr_weighted"] = (
                                roc_auc_score(
                                    y_true_test,
                                    proba,
                                    multi_class="ovr",
                                    average="weighted",
                                )
                            )
                    except Exception as e_auc:
                        metrics_recon["roc_auc_score_error"] = str(e_auc)
                else:  # Regression
                    metrics_recon["r2_score"] = r2_score(y_true_test, preds)
                    metrics_recon["mse"] = mean_squared_error(
                        y_true_test, preds
                    )
                    metrics_recon["mae"] = mean_absolute_error(
                        y_true_test, preds
                    )
                results_dict["train_recon_test_true"] = {
                    "status": "Completed",
                    "metrics": metrics_recon,
                }
        except Exception as e:
            results_dict["train_recon_test_true"] = {
                "status": "Error",
                "message": f"Recon-True pipeline error: {e}",
            }

    if (
        results_dict.get("train_true_test_true", {}).get("status")
        == "Completed"
        or results_dict.get("train_recon_test_true", {}).get("status")
        == "Completed"
    ):
        results_dict["status"] = "Completed"
    if (
        results_dict["status"] == "Partial" and "message" in results
    ):  # Carry over initial skip message
        results_dict["initial_skip_message"] = results["message"]
    return results_dict


def evaluate_machine_learning_utility(
    true_data_base: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    mask: Optional[Dict],
    is_masked: bool,
    dataset_name: Optional[str],
    dataset_target_mapping: Optional[Dict[str, str]],
) -> Dict:
    metrics = {}
    current_true_data_eval = apply_mask_to_true_data(
        true_data_base, mask, is_masked
    )

    temp_true_cols = set(current_true_data_eval.columns)
    temp_recon_cols = set(reconstructed_data_eval.columns)
    common_cols = list(temp_true_cols.intersection(temp_recon_cols))
    if dataset_name == "10.1161_JAHA.118.011771":
        # Special case for the "10.1161_JAHA.118.011771" dataset
        # Remove 'jackson (non-malaria site)' rows (group) from recon data because its not in true data
        if "group" in common_cols:
            current_true_data_eval = current_true_data_eval[
                current_true_data_eval["group"] != "jackson (non-malaria site)"
            ]
            reconstructed_data_eval = reconstructed_data_eval[
                reconstructed_data_eval["group"]
                != "jackson (non-malaria site)"
            ]
        common_cols = list(
            set(current_true_data_eval.columns).intersection(
                set(reconstructed_data_eval.columns)
            )
        )
    # remove 'jackson (non-malaria site)' rows (group)

    if not common_cols:
        msg = "No common columns between true and reconstructed data."
        metrics["feature_importance_preservation"] = {
            "status": "Skipped",
            "message": msg,
        }
        metrics["transfer_learning"] = {"status": "Skipped", "message": msg}
        return metrics

    target_column, task_type, error_msg = _get_target_and_task_type(
        current_true_data_eval,  # Use the (potentially masked) true data for target/task identification
        common_cols,
        mask,
        is_masked,
        dataset_name,
        dataset_target_mapping,
    )

    if error_msg or not target_column or not task_type:
        error_msg = (
            error_msg
            or "Target variable or task type could not be determined."
        )
        metrics["feature_importance_preservation"] = {
            "status": "Skipped",
            "message": error_msg,
        }
        metrics["transfer_learning"] = {
            "status": "Skipped",
            "message": error_msg,
        }
    else:
        features_for_ml = [col for col in common_cols if col != target_column]
        if not features_for_ml:
            msg = "No features available after selecting target."
            metrics["feature_importance_preservation"] = {
                "status": "Skipped",
                "message": msg,
            }
            metrics["transfer_learning"] = {
                "status": "Skipped",
                "message": msg,
            }
        elif target_column not in reconstructed_data_eval.columns:
            msg = f"Target column '{target_column}' not found in reconstructed data."
            metrics["feature_importance_preservation"] = {
                "status": "Skipped",
                "message": msg,
            }
            metrics["transfer_learning"] = {
                "status": "Skipped",
                "message": msg,
            }
        else:
            # For feature importance, we use current_true_data_eval (potentially masked)
            # and reconstructed_data_eval
            metrics["feature_importance_preservation"] = (
                _calculate_feature_importance_comparison(
                    current_true_data_eval,
                    reconstructed_data_eval,
                    features_for_ml,
                    target_column,
                    task_type,
                )
            )
            # For transfer learning, the "true_df_full" should be the original unmasked true data
            # if the goal is to see how well a model trained on reconstructed data performs on "real" test data.
            # However, if the mask implies certain features are *never* available, even for the "true" model baseline,
            # then current_true_data_eval (masked) is appropriate.
            # The current implementation of _perform_transfer_learning_evaluation takes true_df_full,
            # which implies it expects the base true data.
            # Let's pass current_true_data_eval for consistency with how other parts are handled,
            # meaning the "true" model in transfer learning is also trained/tested on potentially masked data.
            metrics["transfer_learning"] = (
                _perform_transfer_learning_evaluation(
                    current_true_data_eval,  # True data (potentially masked)
                    reconstructed_data_eval,  # Reconstructed data
                    features_for_ml,
                    target_column,
                    task_type,
                )
            )
    return metrics
