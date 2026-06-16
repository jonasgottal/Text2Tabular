import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import iqr, ks_2samp, entropy, chi2_contingency, spearmanr
from scipy.spatial.distance import jensenshannon, cdist

from text2tabular.evaluation.evaluation_utils import (
    classify_columns,
    apply_mask_to_true_data,
)
from text2tabular.evaluation.visualizations import (
    _calculate_correlation_matrix_for_heatmap,
)
from text2tabular.evaluation.test_data_utils import GROUP_MAPPING


def _calculate_variable_coverage(
    true_data_eval: pd.DataFrame, reconstructed_data_eval: pd.DataFrame
) -> Dict:
    true_cols = set(true_data_eval.columns)
    recon_cols = set(reconstructed_data_eval.columns)

    common_cols = list(true_cols.intersection(recon_cols))
    true_only_cols = list(true_cols.difference(recon_cols))
    recon_only_cols = list(recon_cols.difference(true_cols))

    type_comparison = {}
    for col in common_cols:
        true_type = str(true_data_eval[col].dtype)
        recon_type = str(reconstructed_data_eval[col].dtype)
        type_comparison[col] = {
            "true_type": true_type,
            "reconstructed_type": recon_type,
            "match": true_type == recon_type,
        }

    return {
        "num_true_variables": len(true_cols),
        "num_reconstructed_variables": len(recon_cols),
        "num_common_variables": len(common_cols),
        "common_variables": common_cols,
        "true_only_variables": true_only_cols,
        "reconstructed_only_variables": recon_only_cols,
        "type_comparison": type_comparison,
    }


def _calculate_descriptive_statistics(
    true_data_eval: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    common_cols: list,
    mask: Optional[Dict] = None,
    is_masked: bool = False,
) -> Dict:
    desc_stats = {}
    final_numerical_cols, final_categorical_cols = classify_columns(
        true_data_eval, reconstructed_data_eval, common_cols
    )

    for col in final_numerical_cols:
        true_series = true_data_eval[col].dropna()
        recon_series = reconstructed_data_eval[col].dropna()

        if true_series.empty or recon_series.empty:
            desc_stats[col] = (
                "Skipped: one or both series empty after dropna for numerical comparison"
            )
            continue

        current_col_stats = {}
        relevant_summary_info = None
        if is_masked and mask and isinstance(mask.get("variables"), dict):
            for var_type_key, var_details_dict in mask["variables"].items():
                if (
                    isinstance(var_details_dict, dict)
                    and col in var_details_dict
                ):
                    if (
                        isinstance(var_details_dict[col], dict)
                        and "overall" in var_details_dict[col]
                    ):
                        relevant_summary_info = var_details_dict[col][
                            "overall"
                        ]
                    break

        compute_mean = (not is_masked) or (
            relevant_summary_info and "mean" in relevant_summary_info
        )
        if compute_mean:
            val_true, val_recon = true_series.mean(), recon_series.mean()
            current_col_stats.update(
                {
                    "mean_true": val_true,
                    "mean_recon": val_recon,
                    "mean_diff": val_true - val_recon,
                    "mean_div": (
                        (val_recon - val_true) / val_true
                        if pd.notna(val_true) and val_true != 0
                        else np.nan
                    ),
                }
            )

        compute_median = (not is_masked) or (
            relevant_summary_info and "median" in relevant_summary_info
        )
        if compute_median:
            val_true, val_recon = true_series.median(), recon_series.median()
            current_col_stats.update(
                {
                    "median_true": val_true,
                    "median_recon": val_recon,
                    "median_diff": val_true - val_recon,
                    "median_div": (
                        (val_recon - val_true) / val_true
                        if pd.notna(val_true) and val_true != 0
                        else np.nan
                    ),
                }
            )

        compute_std = (not is_masked) or (
            relevant_summary_info
            and (
                "std" in relevant_summary_info or "sd" in relevant_summary_info
            )
        )
        if compute_std:
            val_true, val_recon = true_series.std(), recon_series.std()
            current_col_stats.update(
                {
                    "std_true": val_true,
                    "std_recon": val_recon,
                    "std_diff": val_true - val_recon,
                    "std_div": (
                        (val_recon - val_true) / val_true
                        if pd.notna(val_true) and val_true != 0
                        else np.nan
                    ),
                }
            )

        compute_iqr = (not is_masked) or (
            relevant_summary_info
            and (
                (
                    "q1" in relevant_summary_info
                    and "q3" in relevant_summary_info
                )
                or "iqr" in relevant_summary_info
            )
        )
        if compute_iqr:
            val_true = iqr(true_series) if not true_series.empty else np.nan
            val_recon = iqr(recon_series) if not recon_series.empty else np.nan
            current_col_stats.update(
                {
                    "iqr_true": val_true,
                    "iqr_recon": val_recon,
                    "iqr_diff": val_true - val_recon,
                    "iqr_div": (
                        (val_recon - val_true) / val_true
                        if pd.notna(val_true) and val_true != 0
                        else np.nan
                    ),
                }
            )

        compute_min = (not is_masked) or (
            relevant_summary_info and "min" in relevant_summary_info
        )
        if compute_min:
            val_true, val_recon = true_series.min(), recon_series.min()
            current_col_stats.update(
                {
                    "min_true": val_true,
                    "min_recon": val_recon,
                    "min_diff": val_true - val_recon,
                    "min_div": (
                        (val_recon - val_true) / val_true
                        if pd.notna(val_true) and val_true != 0
                        else np.nan
                    ),
                }
            )

        compute_max = (not is_masked) or (
            relevant_summary_info and "max" in relevant_summary_info
        )
        if compute_max:
            val_true, val_recon = true_series.max(), recon_series.max()
            current_col_stats.update(
                {
                    "max_true": val_true,
                    "max_recon": val_recon,
                    "max_diff": val_true - val_recon,
                    "max_div": (
                        (val_recon - val_true) / val_true
                        if pd.notna(val_true) and val_true != 0
                        else np.nan
                    ),
                }
            )

        if not current_col_stats:
            desc_stats[col] = (
                "Skipped: No permissible numerical stats to calculate based on mask/summary or data empty"
            )
        else:
            desc_stats[col] = current_col_stats

    for col in final_categorical_cols:
        true_series = true_data_eval[col].dropna()
        recon_series = reconstructed_data_eval[col].dropna()

        if true_series.empty or recon_series.empty:
            desc_stats[col] = (
                "Skipped: one or both series empty after dropna for categorical comparison"
            )
            continue

        relevant_summary_info = None
        if is_masked and mask and isinstance(mask.get("variables"), dict):
            for var_type_key, var_details_dict in mask["variables"].items():
                if (
                    isinstance(var_details_dict, dict)
                    and col in var_details_dict
                ):
                    if (
                        isinstance(var_details_dict[col], dict)
                        and "overall" in var_details_dict[col]
                    ):
                        relevant_summary_info = var_details_dict[col][
                            "overall"
                        ]
                    break

        compute_categorical_stats = (not is_masked) or (
            relevant_summary_info
            and isinstance(relevant_summary_info, dict)
            and relevant_summary_info
        )

        if not compute_categorical_stats:
            desc_stats[col] = (
                "Skipped: Categorical stats not permissible based on mask/summary or data empty"
            )
            continue

        true_counts = true_series.value_counts(normalize=True)
        recon_counts = recon_series.value_counts(normalize=True)
        all_categories = true_counts.index.union(recon_counts.index)
        cat_comparison = {}
        for category in all_categories:
            cat_comparison[str(category)] = {
                "prop_true": true_counts.get(category, 0),
                "prop_recon": recon_counts.get(category, 0),
                "prop_diff": true_counts.get(category, 0)
                - recon_counts.get(category, 0),
                "prop_div": (
                    (
                        recon_counts.get(category, 0)
                        - true_counts.get(category, 0)
                    )
                    / true_counts.get(category, 0)
                    if pd.notna(true_counts.get(category, 0))
                    and true_counts.get(category, 0) != 0
                    else np.nan
                ),
            }
        desc_stats[col] = cat_comparison
    return desc_stats


def _calculate_column_distribution_similarity(
    true_data_eval: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    common_cols: list,
    ks_alpha: float = 0.1,
    num_kl_bins: int = 10,
) -> Dict:
    distribution_similarity_metrics = {}
    # Note: classify_columns could be used here if strict separation is desired,
    # but current logic directly checks dtype which is also fine.
    numerical_cols = [
        col
        for col in common_cols
        if pd.api.types.is_numeric_dtype(true_data_eval[col])
        and col
        in reconstructed_data_eval.columns  # Ensure col exists before dtype check
        and pd.api.types.is_numeric_dtype(reconstructed_data_eval[col])
    ]
    categorical_cols = [
        col
        for col in common_cols
        if not pd.api.types.is_numeric_dtype(true_data_eval[col])
        and col in reconstructed_data_eval.columns  # Ensure col exists
        and not pd.api.types.is_numeric_dtype(reconstructed_data_eval[col])
    ]

    for col in numerical_cols:
        true_series = true_data_eval[col].dropna()
        recon_series = reconstructed_data_eval[col].dropna()
        col_metrics = {}
        if (
            true_series.empty
            or recon_series.empty
            or len(true_series) < 2
            or len(recon_series) < 2
        ):
            distribution_similarity_metrics[col] = (
                "Skipped: one or both series empty or too few samples"
            )
            continue
        try:
            ks_statistic, p_value = ks_2samp(true_series, recon_series)
            col_metrics["ks_statistic"] = ks_statistic
            col_metrics["ks_p_value"] = p_value
            if p_value < ks_alpha:
                try:
                    min_val = min(true_series.min(), recon_series.min())
                    max_val = max(true_series.max(), recon_series.max())
                    if min_val == max_val:
                        col_metrics.update(
                            {
                                "kl_divergence_true_vs_recon": (
                                    0.0
                                    if len(true_series) > 0
                                    and len(recon_series) > 0
                                    else np.nan
                                ),
                                "kl_divergence_recon_vs_true": (
                                    0.0
                                    if len(true_series) > 0
                                    and len(recon_series) > 0
                                    else np.nan
                                ),
                                "kl_divergence_notes": "Single unique value in combined series, KL set to 0 or NaN.",
                            }
                        )
                    else:
                        bins = np.linspace(min_val, max_val, num_kl_bins + 1)
                        true_hist, _ = np.histogram(
                            true_series, bins=bins, density=False
                        )
                        recon_hist, _ = np.histogram(
                            recon_series, bins=bins, density=False
                        )
                        true_pmf = true_hist / true_hist.sum()
                        recon_pmf = recon_hist / recon_hist.sum()
                        true_pmf = np.nan_to_num(true_pmf)
                        recon_pmf = np.nan_to_num(recon_pmf)
                        epsilon = 1e-10
                        true_pmf_smooth = true_pmf + epsilon
                        true_pmf_smooth /= true_pmf_smooth.sum()
                        recon_pmf_smooth = recon_pmf + epsilon
                        recon_pmf_smooth /= recon_pmf_smooth.sum()
                        col_metrics["kl_divergence_true_vs_recon"] = entropy(
                            true_pmf_smooth, recon_pmf_smooth
                        )
                        col_metrics["kl_divergence_recon_vs_true"] = entropy(
                            recon_pmf_smooth, true_pmf_smooth
                        )
                except Exception as e_kl:
                    col_metrics["kl_divergence_error"] = (
                        f"Error during KL divergence: {str(e_kl)}"
                    )
        except Exception as e_ks:
            col_metrics["ks_test_error"] = f"Error during KS test: {str(e_ks)}"
        distribution_similarity_metrics[col] = col_metrics

    for col in categorical_cols:
        true_series = true_data_eval[col].dropna()
        recon_series = reconstructed_data_eval[col].dropna()
        if true_series.empty or recon_series.empty:
            distribution_similarity_metrics[col] = (
                "Skipped: one or both series empty for JSD"
            )
            continue
        true_counts = true_series.value_counts(normalize=True)
        recon_counts = recon_series.value_counts(normalize=True)
        all_categories = true_counts.index.union(recon_counts.index)
        p = true_counts.reindex(all_categories, fill_value=0).values
        q = recon_counts.reindex(all_categories, fill_value=0).values
        if len(p) == 0 or len(q) == 0:
            distribution_similarity_metrics[col] = (
                "Skipped: No categories found after processing for JSD"
            )
            continue
        try:
            jsd = jensenshannon(p, q, base=2)
            distribution_similarity_metrics[col] = {
                "jensen_shannon_divergence": jsd
            }
        except Exception as e:
            distribution_similarity_metrics[col] = (
                f"Error during JSD calculation: {str(e)}"
            )
    return distribution_similarity_metrics


def _calculate_correlation_structure(
    true_data_eval: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    common_cols: List[str],
    mask: Optional[Dict],
    is_masked: bool,
) -> Dict:
    results = {"numerical_correlations": {}, "categorical_associations": {}}

    base_numerical_cols, base_categorical_cols = classify_columns(
        true_data_eval, reconstructed_data_eval, common_cols
    )

    if not is_masked:
        # --- UNMASKED CASE: Calculate full correlation matrix ---
        current_numerical_cols = list(base_numerical_cols)

        binary_categorical_cols = []
        for col in base_categorical_cols:
            if (
                col in true_data_eval.columns
                and col in reconstructed_data_eval.columns
                and true_data_eval[col].dropna().nunique() <= 2
                and reconstructed_data_eval[col].dropna().nunique() <= 2
            ):
                binary_categorical_cols.append(col)

        all_cols_for_heatmap = sorted(
            list(set(current_numerical_cols + binary_categorical_cols))
        )

        if len(all_cols_for_heatmap) >= 2:
            true_corr_matrix = _calculate_correlation_matrix_for_heatmap(
                true_data_eval,
                all_cols_for_heatmap,
                current_numerical_cols,  # Pass all numerical cols considered for the heatmap
                binary_categorical_cols,  # Pass all binary categorical cols for the heatmap
            )
            recon_corr_matrix = _calculate_correlation_matrix_for_heatmap(
                reconstructed_data_eval,
                all_cols_for_heatmap,
                current_numerical_cols,
                binary_categorical_cols,
            )
            results["numerical_correlations"]["all_pairs"] = {
                "true_matrix": true_corr_matrix.to_dict(),
                "reconstructed_matrix": recon_corr_matrix.to_dict(),
                "difference_matrix": (
                    true_corr_matrix - recon_corr_matrix
                ).to_dict(),
                "deviation_matrix": (  # Calculate percentage deviation
                    (recon_corr_matrix - true_corr_matrix) / true_corr_matrix
                    if not true_corr_matrix.empty
                    and not recon_corr_matrix.empty
                    else pd.DataFrame()
                ).to_dict(),
                "evaluated_columns": all_cols_for_heatmap,
                "numerical_columns_used": [
                    col
                    for col in current_numerical_cols
                    if col in all_cols_for_heatmap
                ],
                "binary_categorical_columns_used": binary_categorical_cols,
            }
        else:
            results["numerical_correlations"]["all_pairs"] = {
                "message": "Not enough numerical or binary categorical columns (< 2) for correlation matrix. "
                f"Numerical considered: {len(current_numerical_cols)}, "
                f"Binary categorical considered: {len(binary_categorical_cols)}."
            }
        results["categorical_associations"]["all_pairs"] = {
            "message": "Categorical association (e.g., Chi2) for all_pairs is not calculated by this function."
        }

    else:  # --- MASKED CASE: Calculate specified numerical pair correlations ---
        spec_num_num_results = {}
        variables_in_mask = set()
        if mask:
            for item_type in ["correlations", "statistical_tests"]:
                for item in mask.get(item_type, []):
                    if isinstance(item, dict) and isinstance(
                        item.get("variables"), list
                    ):
                        for var_name in item["variables"]:
                            variables_in_mask.add(str(var_name).lower())

        if not mask or not variables_in_mask:
            results["numerical_correlations"]["specified_pairs"] = {
                "message": "Mask provided, but no specific variables found in mask for correlation analysis."
            }
        else:
            # Filter base_numerical_cols by variables_in_mask
            context_numerical_cols = [
                col
                for col in base_numerical_cols
                if col.lower() in variables_in_mask
            ]
            actual_numerical_col_names_map = {
                col.lower(): col for col in context_numerical_cols
            }

            masked_numerical_pairs_to_evaluate = set()
            if mask and "correlations" in mask:
                for corr_item in mask.get("correlations", []):
                    if (
                        isinstance(corr_item, dict)
                        and isinstance(corr_item.get("variables"), list)
                        and len(corr_item["variables"]) == 2
                    ):
                        var1_name_from_mask = str(
                            corr_item["variables"][0]
                        ).lower()
                        var2_name_from_mask = str(
                            corr_item["variables"][1]
                        ).lower()

                        actual_var1 = actual_numerical_col_names_map.get(
                            var1_name_from_mask
                        )
                        actual_var2 = actual_numerical_col_names_map.get(
                            var2_name_from_mask
                        )

                        if (
                            actual_var1 and actual_var2
                        ):  # Both variables are in our context_numerical_cols
                            masked_numerical_pairs_to_evaluate.add(
                                tuple(sorted((actual_var1, actual_var2)))
                            )

            if not masked_numerical_pairs_to_evaluate:
                results["numerical_correlations"]["specified_pairs"] = {
                    "message": "Mask specified variables, but no numerical-numerical correlation pairs found/defined in the mask's 'correlations' section."
                }
            else:
                for (
                    col1_actual,
                    col2_actual,
                ) in masked_numerical_pairs_to_evaluate:
                    true_series1 = true_data_eval[col1_actual].dropna()
                    true_series2 = true_data_eval[col2_actual].dropna()
                    recon_series1 = reconstructed_data_eval[
                        col1_actual
                    ].dropna()
                    recon_series2 = reconstructed_data_eval[
                        col2_actual
                    ].dropna()

                    true_corr, recon_corr = np.nan, np.nan

                    # Align series for fair comparison if necessary (Spearman does this internally by rank)
                    # For simplicity, calculate on available data, assuming spearmanr handles NaNs appropriately by ranking.
                    if (
                        len(true_series1) >= 2
                        and len(true_series2) >= 2
                        and true_series1.nunique() > 1
                        and true_series2.nunique() > 1
                    ):
                        # Ensure common indices for paired comparison if that's the intent
                        df_true_pair = pd.DataFrame(
                            {
                                col1_actual: true_series1,
                                col2_actual: true_series2,
                            }
                        ).dropna()
                        if (
                            len(df_true_pair) >= 2
                            and df_true_pair[col1_actual].nunique() > 1
                            and df_true_pair[col2_actual].nunique() > 1
                        ):
                            try:
                                true_corr_val_obj = spearmanr(
                                    df_true_pair[col1_actual],
                                    df_true_pair[col2_actual],
                                )
                                true_corr = true_corr_val_obj.correlation
                            except (
                                ValueError,
                                RuntimeWarning,
                                ZeroDivisionError,
                            ):
                                pass

                    if (
                        len(recon_series1) >= 2
                        and len(recon_series2) >= 2
                        and recon_series1.nunique() > 1
                        and recon_series2.nunique() > 1
                    ):
                        df_recon_pair = pd.DataFrame(
                            {
                                col1_actual: recon_series1,
                                col2_actual: recon_series2,
                            }
                        ).dropna()
                        if (
                            len(df_recon_pair) >= 2
                            and df_recon_pair[col1_actual].nunique() > 1
                            and df_recon_pair[col2_actual].nunique() > 1
                        ):
                            try:
                                recon_corr_val_obj = spearmanr(
                                    df_recon_pair[col1_actual],
                                    df_recon_pair[col2_actual],
                                )
                                recon_corr = recon_corr_val_obj.correlation
                            except (
                                ValueError,
                                RuntimeWarning,
                                ZeroDivisionError,
                            ):
                                pass

                    diff = np.nan
                    if pd.notna(true_corr) and pd.notna(recon_corr):
                        diff = true_corr - recon_corr

                    spec_num_num_results[
                        str(tuple(sorted((col1_actual, col2_actual))))
                    ] = {
                        "true_correlation": true_corr,
                        "reconstructed_correlation": recon_corr,
                        "difference": diff,
                    }
                results["numerical_correlations"][
                    "specified_pairs"
                ] = spec_num_num_results

        results["categorical_associations"]["specified_pairs"] = {
            "message": "Categorical association for specified_pairs (masked) is not calculated by this function."
        }

    return results


def _detect_outliers_iqr(series: pd.Series) -> float:
    if series.empty or series.nunique() < 2:
        return np.nan
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr_val = q3 - q1
    if iqr_val == 0:
        return (
            0.0
            if series.nunique() == 1
            else (
                series[series != q1].count() / len(series)
                if len(series) > 0
                else np.nan
            )
        )
    lower, upper = q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val
    num_outliers = series[(series < lower) | (series > upper)].count()
    return num_outliers / len(series) if len(series) > 0 else np.nan


def _calculate_higher_order_moments(
    true_data_eval: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    common_cols: List[str],
    mask: Optional[Dict],
    is_masked: bool,
) -> Dict:
    moments_results = {}
    # Using classify_columns for consistency
    final_numerical_cols, _ = classify_columns(
        true_data_eval, reconstructed_data_eval, common_cols
    )

    for col in final_numerical_cols:
        true_series, recon_series = (
            true_data_eval[col].dropna(),
            reconstructed_data_eval[col].dropna(),
        )
        col_moments = {}
        if true_series.empty or recon_series.empty:
            moments_results[col] = (
                "Skipped: one or both series empty after dropna"
            )
            continue

        relevant_summary_info = None
        if is_masked and mask and isinstance(mask.get("variables"), dict):
            for var_type_key, var_details_dict in mask["variables"].items():
                if (
                    isinstance(var_details_dict, dict)
                    and col in var_details_dict
                ):
                    if (
                        isinstance(var_details_dict[col], dict)
                        and "overall" in var_details_dict[col]
                    ):
                        relevant_summary_info = var_details_dict[col][
                            "overall"
                        ]
                    break

        compute_skew = (not is_masked) or (
            relevant_summary_info and "skewness" in relevant_summary_info
        )
        if compute_skew:
            skew_true, skew_recon = (
                true_series.skew() if len(true_series) >= 3 else np.nan
            ), (recon_series.skew() if len(recon_series) >= 3 else np.nan)
            col_moments.update(
                {
                    "skewness_true": skew_true,
                    "skewness_reconstructed": skew_recon,
                    "skewness_diff": (
                        skew_true - skew_recon
                        if pd.notna(skew_true) and pd.notna(skew_recon)
                        else np.nan
                    ),
                    "skewness_div": (
                        (skew_recon - skew_true) / skew_true
                        if pd.notna(skew_true) and skew_true != 0
                        else np.nan
                    ),
                }
            )

        compute_kurt = (not is_masked) or (
            relevant_summary_info and "kurtosis" in relevant_summary_info
        )
        if compute_kurt:
            kurt_true, kurt_recon = (
                true_series.kurt() if len(true_series) >= 4 else np.nan
            ), (recon_series.kurt() if len(recon_series) >= 4 else np.nan)
            col_moments.update(
                {
                    "kurtosis_true": kurt_true,
                    "kurtosis_reconstructed": kurt_recon,
                    "kurtosis_diff": (
                        kurt_true - kurt_recon
                        if pd.notna(kurt_true) and pd.notna(kurt_recon)
                        else np.nan
                    ),
                    "kurtosis_div": (
                        (kurt_recon - kurt_true) / kurt_true
                        if pd.notna(kurt_true) and kurt_true != 0
                        else np.nan
                    ),
                }
            )

        compute_outliers = (not is_masked) or (
            relevant_summary_info
            and all(
                k in relevant_summary_info for k in ["min", "max", "q1", "q3"]
            )
        )
        if compute_outliers:
            out_true, out_recon = _detect_outliers_iqr(
                true_series
            ), _detect_outliers_iqr(recon_series)
            col_moments.update(
                {
                    "outlier_proportion_iqr_true": out_true,
                    "outlier_proportion_iqr_reconstructed": out_recon,
                    "outlier_proportion_iqr_diff": (
                        out_true - out_recon
                        if pd.notna(out_true) and pd.notna(out_recon)
                        else np.nan
                    ),
                    "outlier_proportion_iqr_div": (
                        (out_recon - out_true) / out_true
                        if pd.notna(out_true) and out_true != 0
                        else np.nan
                    ),
                }
            )

        if not col_moments:
            moments_results[col] = (
                "Skipped: No higher-order moments or outlier stats permissible/calculable."
            )
        else:
            moments_results[col] = col_moments
    return {"per_column_stats": moments_results}


def _df_to_data_dict(df: pd.DataFrame, group_var: str, groups: list) -> dict:
    """
    Convert a DataFrame to a dict of {(variable, group): pd.Series}.
    Returns pd.Series (retaining indices and NaNs) so that downstream
    statistical tests can perform exact pairwise alignment/deletion.
    """
    data = {}
    for group in groups:
        group_df = (
            df[df[group_var] == group] if group_var in df.columns else df
        )
        cols_to_use = [col for col in df.columns if col != group_var]

        for col in cols_to_use:
            # Retain as pd.Series to keep the index for downstream pairwise deletion
            data[(col, group)] = group_df[col]

    return data


def _evaluate_statistical_tests_against_true(
    true_data: pd.DataFrame,
    reconstructed_data: pd.DataFrame,
    parsed_data: dict,
    dataset_name: Optional[str] = None,
) -> dict:
    """
    Evaluate statistical tests (e.g., t-test, chi2, ANOVA) on both true and reconstructed data.
    Returns a dict with test names as keys and both true and reconstructed results.
    """
    from text2tabular.reconstruction.mcmc.mcmc_utils import (
        calculate_statistical_tests,
    )

    test_specs = (
        parsed_data.get("statistical_tests", []) if parsed_data else []
    )
    groups = parsed_data.get("groups", [])

    if dataset_name:
        group_var = GROUP_MAPPING.get(dataset_name, "group")
    else:
        group_var = "group"
    # Convert DataFrames to data dicts
    true_data_dict = _df_to_data_dict(true_data, group_var, groups)
    recon_data_dict = _df_to_data_dict(reconstructed_data, group_var, groups)

    # Evaluate tests
    true_results = calculate_statistical_tests(
        true_data_dict, test_specs, groups
    )
    recon_results = calculate_statistical_tests(
        recon_data_dict, test_specs, groups
    )

    # Aggregate results by test_type/variables/groups for comparison
    results = {}
    for t_spec in test_specs:
        key = (
            t_spec.get("test_type"),
            tuple(t_spec.get("variables", [])),
            tuple(t_spec.get("groups", [])),
        )
        # Find corresponding results
        true_res = next(
            (
                r
                for r in true_results
                if r.get("test_type") == t_spec.get("test_type")
                and r.get("variables") == t_spec.get("variables")
                and tuple(r.get("groups", []))
                == tuple(t_spec.get("groups", []))
            ),
            None,
        )
        recon_res = next(
            (
                r
                for r in recon_results
                if r.get("test_type") == t_spec.get("test_type")
                and r.get("variables") == t_spec.get("variables")
                and tuple(r.get("groups", []))
                == tuple(t_spec.get("groups", []))
            ),
            None,
        )
        results[str(key)] = {
            "input_target": t_spec.get("test_statistic"),
            "true_data": true_res.get("test_statistic") if true_res else None,
            "reconstructed_data": (
                recon_res.get("test_statistic") if recon_res else None
            ),
            "true_result_full": true_res,
            "reconstructed_result_full": recon_res,
        }
    return results


def _calculate_distance_recon_to_true(
    true_data_eval: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    common_cols: list,
    sample_size: int = 1000,
) -> dict:
    """
    Computes the distance from each reconstructed record to its closest true record.
    Returns the average and the full list of distances for histogram plotting.
    """
    numeric_cols = [
        col
        for col in common_cols
        if pd.api.types.is_numeric_dtype(true_data_eval[col])
        and pd.api.types.is_numeric_dtype(reconstructed_data_eval[col])
    ]
    categorical_cols = [
        col
        for col in common_cols
        if not pd.api.types.is_numeric_dtype(true_data_eval[col])
        and not pd.api.types.is_numeric_dtype(reconstructed_data_eval[col])
    ]

    if not numeric_cols and not categorical_cols:
        return {
            "avg_distance_recon_to_true": np.nan,
            "distances_recon_to_true": [],
            "numeric_columns_used": [],
            "categorical_columns_used": [],
            "message": "No common columns for distance calculation.",
        }

    # Drop rows with NaNs in any relevant columns
    all_cols = numeric_cols + categorical_cols
    true_df = true_data_eval[all_cols].dropna()
    recon_df = reconstructed_data_eval[all_cols].dropna()

    if sample_size is not None and len(recon_df) > sample_size:
        recon_df = recon_df.sample(n=sample_size, random_state=42)

    if true_df.shape[0] == 0 or recon_df.shape[0] == 0:
        return {
            "avg_distance_recon_to_true": np.nan,
            "distances_recon_to_true": [],
            "numeric_columns_used": numeric_cols,
            "categorical_columns_used": categorical_cols,
            "message": "No data after dropping NaNs.",
        }

    # Numeric distances (cityblock)
    if numeric_cols:
        # Ensure numeric dtype (avoid object dtype)
        true_numeric = true_df[numeric_cols].astype(float).to_numpy()
        recon_numeric = recon_df[numeric_cols].astype(float).to_numpy()
        numeric_dists = cdist(recon_numeric, true_numeric, metric="cityblock")
    else:
        numeric_dists = np.zeros((len(recon_df), len(true_df)))

    # Categorical distances (0 if match, 1 if not, summed over columns)
    if categorical_cols:
        true_categorical = true_df[categorical_cols].to_numpy(dtype=object)
        recon_categorical = recon_df[categorical_cols].to_numpy(dtype=object)
        # For each reconstructed row, compare to all true rows
        cat_dists = np.zeros((len(recon_df), len(true_df)))
        for i, recon_row in enumerate(recon_categorical):
            # Broadcasting: (n_true, n_cat) != (n_cat,) -> (n_true, n_cat)
            mismatches = (true_categorical != recon_row).astype(int)
            cat_dists[i, :] = mismatches.sum(axis=1)
    else:
        cat_dists = np.zeros((len(recon_df), len(true_df)))

    # Total distance is sum of numeric and categorical distances
    total_dists = numeric_dists + cat_dists
    min_dists = total_dists.min(axis=1)
    avg_dist = min_dists.mean()

    return {
        "avg_distance_recon_to_true": float(avg_dist),
        "distances_recon_to_true": min_dists.tolist(),
        "numeric_columns_used": numeric_cols,
        "categorical_columns_used": categorical_cols,
    }


def evaluate_statistical_similarity(
    true_data_base: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    mask: Optional[Dict],
    is_masked: bool,
    dataset_name: Optional[str] = None,
) -> Dict:
    """
    Evaluate statistical similarity between true and reconstructed data.

    Metrics:
    - Variable coverage:
      Compare the number of variables in both datasets and their types.
    - Descriptive Statistics:
      Compare counts, means, medians, standard deviations, interquartile ranges (IQR), minimums, and maximums
    - Column-Wise Distribution Similarity:
      Automatically handles continuous (e.g., KS test, KL Divergence) and categorical variables (e.g., Jensen-Shannon divergence).
    - Correlation Structure Preservation:
      Compare correlation matrices for continuous variables and contingency tables for categorical variables.
    - Higher-Order Moments and Outlier Detection:
      Compare skewness, kurtosis, and detect outliers in both datasets.

    Returns:
    - A dictionary containing statistical similarity metrics.
    """
    metrics = {}
    current_true_data_eval = apply_mask_to_true_data(
        true_data_base, mask, is_masked
    )

    coverage_info = _calculate_variable_coverage(
        current_true_data_eval, reconstructed_data_eval
    )
    metrics["variable_coverage"] = coverage_info
    common_cols = coverage_info["common_variables"]

    metrics["descriptive_statistics"] = _calculate_descriptive_statistics(
        current_true_data_eval,
        reconstructed_data_eval,
        common_cols,
        mask,
        is_masked,
    )
    metrics["column_distribution_similarity"] = (
        _calculate_column_distribution_similarity(
            current_true_data_eval, reconstructed_data_eval, common_cols
        )
    )
    metrics["correlation_structure"] = _calculate_correlation_structure(
        current_true_data_eval,
        reconstructed_data_eval,
        common_cols,
        mask,
        is_masked,
    )

    metrics["higher_order_moments_and_outliers"] = (
        _calculate_higher_order_moments(
            current_true_data_eval,
            reconstructed_data_eval,
            common_cols,
            mask,
            is_masked,
        )
    )
    metrics["statistical_tests"] = _evaluate_statistical_tests_against_true(
        current_true_data_eval,
        reconstructed_data_eval,
        parsed_data=mask,
        dataset_name=dataset_name,
    )

    # Add distance from reconstructed to true records
    metrics["distance_recon_to_true"] = _calculate_distance_recon_to_true(
        current_true_data_eval,
        reconstructed_data_eval,
        common_cols,
    )

    return metrics
