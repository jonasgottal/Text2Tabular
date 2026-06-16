import pandas as pd
import numpy as np
import copy
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
import matplotlib
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from text2tabular.evaluation.test_data_reconstruction import (
    load_test_datasets,
    test_data_reconstruction,
    test_anova_expansion,
)
from text2tabular.evaluation.test_data_utils import GROUP_MAPPING
from text2tabular.evaluation.statistical_similarity import (
    evaluate_statistical_similarity,
)
from text2tabular.evaluation.ml_utility import (
    evaluate_machine_learning_utility,
)
from text2tabular.evaluation.visualizations import (
    generate_visualizations,
    visualize_anova_results,
)

from text2tabular.reconstruction.main import generate_synthetic_data

import warnings

warnings.filterwarnings("ignore")


class DataReconstructionEvaluation:
    """
    Evaluation metrics for data reconstruction from textual sources.
    If masked, the evaluation will be performed on the data that can be retrieved
    (i.e., the data present in text or defined in the JSON mask).
    """

    def __init__(
        self,
        true_data: pd.DataFrame,
        reconstructed_data: pd.DataFrame,
        mask_param: Optional[Dict] = None,
        masked: bool = False,
        dataset_name: Optional[str] = None,
        dataset_target_mapping: Optional[Dict[str, str]] = None,
    ):

        true_df_processed = true_data.copy()
        recon_df_processed = reconstructed_data.copy()

        true_df_processed.columns = [
            str(col).lower() for col in true_df_processed.columns
        ]
        recon_df_processed.columns = [
            str(col).lower() for col in recon_df_processed.columns
        ]

        for df_proc in [true_df_processed, recon_df_processed]:
            for col_name in df_proc.columns:
                if df_proc[col_name].dtype == "object":
                    try:
                        df_proc[col_name] = df_proc[col_name].apply(
                            lambda x: x.lower() if isinstance(x, str) else x
                        )
                    except Exception:
                        # Consider logging this exception or handling it more specifically
                        pass

        self.true_data_orig: pd.DataFrame = true_df_processed
        self.reconstructed_data_orig: pd.DataFrame = recon_df_processed

        self.mask: Optional[Dict] = None
        self.masked: bool = masked
        self.dataset_name: Optional[str] = dataset_name
        self.dataset_target_mapping: Optional[Dict[str, str]] = (
            dataset_target_mapping
        )

        if mask_param:
            self.mask = copy.deepcopy(mask_param)
            processed_variables_dict = {}
            temp_masked_variable_list = []

            if isinstance(self.mask.get("variables"), dict):
                for var_type, var_details_dict in self.mask[
                    "variables"
                ].items():
                    new_var_details_dict = {}
                    if isinstance(var_details_dict, dict):
                        for var_name, details in var_details_dict.items():
                            lower_var_name = str(var_name).lower()
                            new_var_details_dict[lower_var_name] = details
                            if lower_var_name not in temp_masked_variable_list:
                                temp_masked_variable_list.append(
                                    lower_var_name
                                )
                    processed_variables_dict[str(var_type)] = (
                        new_var_details_dict
                    )
                self.mask["variables"] = processed_variables_dict
                if temp_masked_variable_list:
                    self.mask["_masked_variable_list_internal"] = (
                        temp_masked_variable_list
                    )
                else:
                    self.mask["_masked_variable_list_internal"] = []
            else:
                self.mask["variables"] = {}
                self.mask["_masked_variable_list_internal"] = []

            if self.dataset_name and self.dataset_name in GROUP_MAPPING:
                group_var_name = GROUP_MAPPING[self.dataset_name].lower()

                if group_var_name in self.true_data_orig.columns:
                    if (
                        group_var_name
                        not in self.mask["_masked_variable_list_internal"]
                    ):
                        self.mask["_masked_variable_list_internal"].append(
                            group_var_name
                        )

                    is_group_var_in_mask_variables = False
                    for var_type_details in self.mask["variables"].values():
                        if (
                            isinstance(var_type_details, dict)
                            and group_var_name in var_type_details
                        ):
                            is_group_var_in_mask_variables = True
                            break

                    if not is_group_var_in_mask_variables:
                        if "categorical" not in self.mask["variables"]:
                            self.mask["variables"]["categorical"] = {}

                        if isinstance(
                            self.mask["variables"].get("categorical"), dict
                        ):
                            self.mask["variables"]["categorical"][
                                group_var_name
                            ] = {}
                        # else: Consider logging if 'categorical' is not a dict as expected

        # self.true_data_for_eval is the version that specific evaluation functions
        # will take as a base and then apply masking internally if self.masked is True.
        self.true_data_for_eval: pd.DataFrame = self.true_data_orig.copy()

        temp_recon_for_eval: pd.DataFrame = self.reconstructed_data_orig.copy()
        if "group" in temp_recon_for_eval.columns:
            unique_groups_in_recon: list = (
                temp_recon_for_eval["group"].unique().tolist()
            )

            if unique_groups_in_recon == ["overall"]:
                self.reconstructed_data_for_eval: pd.DataFrame = (
                    temp_recon_for_eval.drop(columns=["group"])
                )
            else:
                self.reconstructed_data_for_eval: pd.DataFrame = (
                    temp_recon_for_eval.copy()
                )
        else:
            self.reconstructed_data_for_eval: pd.DataFrame = (
                temp_recon_for_eval.copy()
            )

    def evaluate_tabular_reconstruction(
        self,
    ) -> Dict:  # This is an instance method
        """
        Evaluate the quality of tabular data reconstruction.
        Uses self.true_data_for_eval (potentially masked inside methods) and self.reconstructed_data_for_eval.

        Returns:
        - A dictionary containing evaluation metrics.
        """
        metrics = {}

        # 1. Statistical Similarity Metrics
        metrics["statistical_similarity"] = evaluate_statistical_similarity(
            self.true_data_for_eval,
            self.reconstructed_data_for_eval,
            self.mask,
            self.masked,
            self.dataset_name,
        )

        # 2. Machine Learning Utility Metrics
        metrics["machine_learning_utility"] = (
            evaluate_machine_learning_utility(
                self.true_data_for_eval,
                self.reconstructed_data_for_eval,
                self.mask,
                self.masked,
                self.dataset_name,
                self.dataset_target_mapping,
            )
        )

        # 3. Visualization-Based Metrics
        metrics["visualizations"] = generate_visualizations(
            self.true_data_for_eval,
            self.reconstructed_data_for_eval,
            self.mask,
            self.masked,
            self.dataset_name,
        )

        return metrics

    def aggregate_results(self, metrics: Dict, IQR=False) -> Dict:
        """
        Aggregates the detailed evaluation metrics into a summary of KPIs.

        Parameters:
        - metrics: The detailed metrics dictionary produced by evaluate_tabular_reconstruction.

        Returns:
        - A dictionary containing aggregated/summary KPIs.
        """
        metric = "div"  # "diff"
        aggregated_summary: Dict[str, Dict[str, Any]] = {
            "statistical_similarity": {},
            "machine_learning_utility": {},
        }

        # Helper for robust averaging
        def _robust_average(values: List[Any]) -> float:
            valid_values = [
                float(v)
                for v in values
                if pd.notna(v) and isinstance(v, (int, float))
            ]
            if not valid_values:
                return np.nan
            return np.mean(valid_values)

        # --- Statistical Similarity ---
        stats_metrics = metrics.get("statistical_similarity", {})

        # 1. Variable Coverage
        coverage_info = stats_metrics.get("variable_coverage", {})
        num_true_vars = coverage_info.get("num_true_variables")
        num_common_vars = coverage_info.get("num_common_variables")
        if num_true_vars is not None and num_common_vars is not None:
            aggregated_summary["statistical_similarity"][
                "variable_coverage"
            ] = f"{num_common_vars}/{num_true_vars}"
        else:
            aggregated_summary["statistical_similarity"][
                "variable_coverage"
            ] = "N/A"

        # 2. Descriptive Statistics
        desc_stats = stats_metrics.get("descriptive_statistics", {})
        mean_diffs, median_diffs, std_diffs, iqr_diffs, prop_diffs = (
            [],
            [],
            [],
            [],
            [],
        )
        if isinstance(desc_stats, dict):
            for col_name, col_stats in desc_stats.items():
                if isinstance(col_stats, dict):
                    if f"mean_{metric}" in col_stats:  # Numeric column
                        mean_diffs.append(abs(col_stats[f"mean_{metric}"]))
                        median_diffs.append(
                            abs(col_stats.get(f"median_{metric}", np.nan))
                        )
                        std_diffs.append(
                            abs(col_stats.get(f"std_{metric}", np.nan))
                        )
                        iqr_diffs.append(
                            abs(col_stats.get(f"iqr_{metric}", np.nan))
                        )
                    else:  # Potentially categorical column
                        for category_name, cat_stats in col_stats.items():
                            if (
                                isinstance(cat_stats, dict)
                                and f"prop_{metric}" in cat_stats
                            ):
                                prop_diffs.append(
                                    abs(cat_stats[f"prop_{metric}"])
                                )
        if IQR:
            aggregated_summary["statistical_similarity"][
                "avg_abs_median_difference_numeric"
            ] = _robust_average(median_diffs)
            aggregated_summary["statistical_similarity"][
                "avg_abs_iqr_difference_numeric"
            ] = _robust_average(iqr_diffs)

        else:
            aggregated_summary["statistical_similarity"][
                "avg_abs_mean_difference_numeric"
            ] = _robust_average(mean_diffs)
            aggregated_summary["statistical_similarity"][
                "avg_abs_std_difference_numeric"
            ] = _robust_average(std_diffs)

        aggregated_summary["statistical_similarity"][
            "avg_abs_proportion_difference_categorical"
        ] = _robust_average(prop_diffs)

        # 3. Column Distribution Similarity
        dist_sim_metrics = stats_metrics.get(
            "column_distribution_similarity", {}
        )
        ks_stats, kl_true_vs_recon, kl_recon_vs_true, jsd_stats = (
            [],
            [],
            [],
            [],
        )
        if isinstance(dist_sim_metrics, dict):
            for col_name, col_metrics in dist_sim_metrics.items():
                if isinstance(col_metrics, dict):
                    if "ks_statistic" in col_metrics:
                        ks_stats.append(col_metrics["ks_statistic"])
                    if "kl_divergence_true_vs_recon" in col_metrics:
                        kl_true_vs_recon.append(
                            col_metrics["kl_divergence_true_vs_recon"]
                        )
                    if "kl_divergence_recon_vs_true" in col_metrics:
                        kl_recon_vs_true.append(
                            col_metrics["kl_divergence_recon_vs_true"]
                        )
                    if "jensen_shannon_divergence" in col_metrics:
                        jsd_stats.append(
                            col_metrics["jensen_shannon_divergence"]
                        )

        aggregated_summary["statistical_similarity"][
            "avg_ks_statistic_numeric"
        ] = _robust_average(ks_stats)
        # aggregated_summary["statistical_similarity"][
        #     "avg_kl_divergence_true_vs_recon_numeric"
        # ] = _robust_average(kl_true_vs_recon)
        # aggregated_summary["statistical_similarity"][
        #     "avg_kl_divergence_recon_vs_true_numeric"
        # ] = _robust_average(kl_recon_vs_true)
        aggregated_summary["statistical_similarity"][
            "avg_jensen_shannon_divergence_categorical"
        ] = _robust_average(jsd_stats)
        # Inverse KL divergence (true vs recon)
        if kl_true_vs_recon:
            inv_kl_vals = [
                1.0 / (1.0 + float(v))
                for v in kl_true_vs_recon
                if pd.notna(v) and isinstance(v, (int, float))
            ]
            aggregated_summary["statistical_similarity"][
                "inverse_kl_divergence_true_vs_recon_numeric"
            ] = (np.mean(inv_kl_vals) if inv_kl_vals else np.nan)
        else:
            aggregated_summary["statistical_similarity"][
                "inverse_kl_divergence_true_vs_recon_numeric"
            ] = np.nan

        # if kl_recon_vs_true:
        #     inv_kl_vals = [
        #         1.0 / (1.0 + float(v))
        #         for v in kl_recon_vs_true
        #         if pd.notna(v) and isinstance(v, (int, float))
        #     ]
        #     aggregated_summary["statistical_similarity"][
        #         "inverse_kl_divergence_recon_vs_true_numeric"
        #     ] = (np.mean(inv_kl_vals) if inv_kl_vals else np.nan)
        # else:
        #     aggregated_summary["statistical_similarity"][
        #         "inverse_kl_divergence_recon_vs_true_numeric"
        #     ] = np.nan

        # 4. Correlation Structure
        corr_structure = stats_metrics.get("correlation_structure", {})
        corr_diffs = []
        if isinstance(corr_structure.get("numerical_correlations"), dict):
            num_corrs = corr_structure["numerical_correlations"]
            if isinstance(num_corrs.get("all_pairs"), dict) and isinstance(
                num_corrs["all_pairs"].get("difference_matrix"), dict
            ):
                diff_matrix = num_corrs["all_pairs"]["difference_matrix"]
                cols = list(diff_matrix.keys())
                for i in range(len(cols)):
                    for j in range(
                        i + 1, len(cols)
                    ):  # Upper triangle, excluding diagonal
                        val = diff_matrix[cols[i]].get(cols[j])
                        if pd.notna(val):
                            corr_diffs.append(abs(val))
            elif isinstance(num_corrs.get("specified_pairs"), dict):
                for pair_key, pair_data in num_corrs[
                    "specified_pairs"
                ].items():
                    if isinstance(pair_data, dict) and pd.notna(
                        pair_data.get("difference")
                    ):
                        corr_diffs.append(abs(pair_data["difference"]))
        aggregated_summary["statistical_similarity"][
            "avg_abs_correlation_difference_numerical"
        ] = _robust_average(corr_diffs)

        # 6. Statistical Tests (average absolute difference between true_data and reconstructed_data)
        stat_tests = stats_metrics.get("statistical_tests", {})
        stat_test_diffs = []
        if isinstance(stat_tests, dict):
            for test_name, test_result in stat_tests.items():
                if (
                    isinstance(test_result, dict)
                    and "true_data" in test_result
                    and "reconstructed_data" in test_result
                ):
                    true_val = test_result["true_data"]
                    recon_val = test_result["reconstructed_data"]
                    if pd.notna(true_val) and pd.notna(recon_val):
                        try:
                            true_val_float = float(true_val)
                            recon_val_float = float(recon_val)
                            if true_val_float != 0:
                                percent_deviation = abs(
                                    true_val_float - recon_val_float
                                ) / abs(true_val_float)
                                stat_test_diffs.append(percent_deviation)
                            else:
                                # If true_val is zero, fallback to absolute difference
                                stat_test_diffs.append(
                                    abs(true_val_float - recon_val_float)
                                )
                        except Exception:
                            # Fallback to absolute difference if conversion fails
                            stat_test_diffs.append(
                                abs(float(true_val) - float(recon_val))
                            )
        aggregated_summary["statistical_similarity"][
            "avg_abs_statistical_test_deviation"
        ] = _robust_average(stat_test_diffs)

        # 5. Higher-Order Moments and Outliers
        higher_order = stats_metrics.get(
            "higher_order_moments_and_outliers", {}
        )
        outlier_prop_diffs = []
        if isinstance(higher_order.get("per_column_stats"), dict):
            for col_name, col_stats in higher_order[
                "per_column_stats"
            ].items():
                if (
                    isinstance(col_stats, dict)
                    and f"outlier_proportion_iqr_diff" in col_stats
                ):
                    outlier_prop_diffs.append(
                        abs(col_stats[f"outlier_proportion_iqr_diff"])
                    )
        aggregated_summary["statistical_similarity"][
            "avg_abs_outlier_proportion_iqr_difference_numeric"
        ] = _robust_average(outlier_prop_diffs)

        # 7. Distance from reconstructed to true records
        distance_metrics = stats_metrics.get("distance_recon_to_true", {})
        avg_dist = distance_metrics.get("avg_distance_recon_to_true", np.nan)
        aggregated_summary["statistical_similarity"][
            "avg_distance_recon_to_true"
        ] = avg_dist

        # plot with dataset name in

        # --- Machine Learning Utility ---
        ml_metrics = metrics.get("machine_learning_utility", {})

        # 1. Feature Importance Preservation
        feat_imp = ml_metrics.get("feature_importance_preservation", {})
        if (
            isinstance(feat_imp, dict)
            and feat_imp.get("status") == "Completed"
        ):
            aggregated_summary["machine_learning_utility"][
                "feature_importance_spearman_rank_correlation"
            ] = feat_imp.get("spearman_rank_correlation", np.nan)
        else:
            aggregated_summary["machine_learning_utility"][
                "feature_importance_spearman_rank_correlation"
            ] = np.nan

        # 2. Transfer Learning
        transfer_learn = ml_metrics.get("transfer_learning", {})
        f1_true_true, f1_recon_true = np.nan, np.nan
        auc_true, auc_recon = 0.5, 0.5
        if (
            isinstance(transfer_learn, dict)
            and transfer_learn.get("status") == "Completed"
        ):
            if (
                isinstance(transfer_learn.get("train_true_test_true"), dict)
                and transfer_learn["train_true_test_true"].get("status")
                == "Completed"
                and isinstance(
                    transfer_learn["train_true_test_true"].get("metrics"), dict
                )
            ):
                f1_true_true = transfer_learn["train_true_test_true"][
                    "metrics"
                ].get("f1_score_weighted", np.nan)
                auc_true = transfer_learn["train_true_test_true"][
                    "metrics"
                ].get(
                    "roc_auc_score",
                    transfer_learn["train_true_test_true"]["metrics"].get(
                        "roc_auc_score_ovr_weighted", np.nan
                    ),
                )

            if (
                isinstance(transfer_learn.get("train_recon_test_true"), dict)
                and transfer_learn["train_recon_test_true"].get("status")
                == "Completed"
                and isinstance(
                    transfer_learn["train_recon_test_true"].get("metrics"),
                    dict,
                )
            ):
                f1_recon_true = transfer_learn["train_recon_test_true"][
                    "metrics"
                ].get("f1_score_weighted", np.nan)
                auc_recon = transfer_learn["train_recon_test_true"][
                    "metrics"
                ].get(
                    "roc_auc_score",
                    transfer_learn["train_recon_test_true"]["metrics"].get(
                        "roc_auc_score_ovr_weighted", np.nan
                    ),
                )

        if pd.notna(f1_true_true) and pd.notna(f1_recon_true):
            aggregated_summary["machine_learning_utility"][
                "transfer_learning_f1_score_weighted_difference"
            ] = (f1_true_true - f1_recon_true)
        else:
            aggregated_summary["machine_learning_utility"][
                "transfer_learning_f1_score_weighted_difference"
            ] = np.nan
        if pd.notna(auc_true) and pd.notna(auc_recon):
            aggregated_summary["machine_learning_utility"][
                "transfer_learning_auc_difference"
            ] = (auc_true - auc_recon)
        else:
            aggregated_summary["machine_learning_utility"][
                "transfer_learning_auc_difference"
            ] = np.nan

        return aggregated_summary


def generate_latex_table(aggregated_summaries, abbrv=False):
    import re

    summary_df = pd.DataFrame(aggregated_summaries)
    if "dataset_name" in summary_df.columns:
        summary_df = summary_df.set_index("dataset_name")

    # Create lists of columns by category
    statistical_cols = [
        col
        for col in summary_df.columns
        if col.startswith("statistical_similarity")
    ]
    ml_cols = [
        col
        for col in summary_df.columns
        if col.startswith("machine_learning_utility")
    ]

    metric_abbreviations = {
        # Statistical Similarity metrics
        # TODO enumerate metrics and consistent width
        "Variable Coverage": "Var Coverage",
        "Avg Abs Mean Difference Numeric": "Mean Dev",
        "Avg Abs Std Difference Numeric": "Std Dev",
        "Avg Abs Proportion Difference Categorical": "Cat Prop Dev",
        "Avg Ks Statistic Numeric": "KS Stat",
        "Inverse Kl Divergence True Vs Recon Numeric": "Inv KL Div",
        "Avg Jensen Shannon Divergence Categorical": "JS Div",
        "Avg Abs Correlation Difference Numerical": "Corr Dev",
        "Avg Abs Outlier Proportion Iqr Difference Numeric": "Outlier Diff",
        "Avg Distance Recon To True": "Mean Dist R→T",
        "Avg Abs Statistical Test Deviation": "Stat Dev",
        # Machine Learning Utility metrics
        "Feature Importance Spearman Rank Correlation": "Feat Imp Corr",
        "Transfer Learning F1 Score Weighted Difference": "F1 Diff",
        "Transfer Learning Auc Difference": "AUC Diff",
    }

    def clean_metric_name(name):
        if name.startswith("statistical_similarity_"):
            name = name[len("statistical_similarity_") :]
        elif name.startswith("machine_learning_utility_"):
            name = name[len("machine_learning_utility_") :]
        return " ".join(word.capitalize() for word in name.split("_"))

    formatted_index = [name.replace("_", " ") for name in summary_df.index]

    if abbrv:

        def shorten(name):
            import re

            # Remove all non-alphanumeric characters
            alphanum = re.sub(r"[^A-Za-z0-9]", "", name)
            # Take the last 8, but at least 4 characters
            short = alphanum[-7:]
            if len(short) < 4:
                short = alphanum[-4:]
            return f"...{short}"

        formatted_index = [shorten(name) for name in summary_df.index]

    formatted_df = pd.DataFrame(
        summary_df[statistical_cols + ml_cols].values,
        index=formatted_index,
        columns=[clean_metric_name(col) for col in statistical_cols + ml_cols],
    )

    transposed_df = formatted_df.transpose()
    new_index = []
    stat_metrics = [clean_metric_name(col) for col in statistical_cols]
    ml_metrics = [clean_metric_name(col) for col in ml_cols]
    for metric in stat_metrics:
        new_index.append(metric)
    for metric in ml_metrics:
        new_index.append(metric)
    final_df = pd.DataFrame(index=new_index, columns=transposed_df.columns)
    for metric in stat_metrics:
        final_df.loc[metric] = transposed_df.loc[metric]
    for metric in ml_metrics:
        final_df.loc[metric] = transposed_df.loc[metric]
    final_df_compact = final_df.copy()
    final_df_compact.index = [
        metric_abbreviations.get(idx, idx) for idx in final_df.index
    ]
    latex_table = final_df_compact.to_latex(
        float_format="%.3f",
        bold_rows=False,
        caption="Data Reconstruction Evaluation Metrics Across Datasets",
        label="tab:data_reconstruction",
        na_rep="",
    )
    latex_table = latex_table.replace(
        "\\begin{table}", "\\begin{table}\n\\centering"
    )

    latex_table = latex_table.replace("\\toprule", "\\hline")
    latex_table = latex_table.replace("\\midrule", "\\hline")
    latex_table = latex_table.replace("\\bottomrule", "\\hline")
    table_format = re.search(r"\\begin{tabular}{([^}]+)}", latex_table)
    if table_format:
        format_str = table_format.group(1)
        col_count = len(format_str.strip())
        column_formats = ["l"] + ["c"] * (col_count - 1)
        new_format = "|" + "|".join(column_formats) + "|"
        latex_table = latex_table.replace(
            f"\\begin{{tabular}}{{{format_str}}}",
            f"\\begin{{tabular}}{{{new_format}}}",
        )
    # Section headers
    if stat_metrics:
        first_stat_abbr = metric_abbreviations.get(
            stat_metrics[0], stat_metrics[0]
        )
        pattern = re.compile(rf"{re.escape(first_stat_abbr)}.*")
        match = pattern.search(latex_table)
        if match:
            insert_pos = match.start()
            header_line = (
                "\\multicolumn{"
                + str(col_count)
                + "}{|c|}{\\textbf{Statistical Similarity}} \\\\\n"
            )
            latex_table = (
                latex_table[:insert_pos]
                + header_line
                + latex_table[insert_pos:]
            )
    if ml_metrics:
        first_ml_abbr = metric_abbreviations.get(ml_metrics[0], ml_metrics[0])
        pattern = re.compile(rf"{re.escape(first_ml_abbr)}.*")
        match = pattern.search(latex_table)
        if match:
            insert_pos = match.start()
            header_line = (
                "\\multicolumn{"
                + str(col_count)
                + "}{|c|}{\\textbf{Machine Learning Utility}} \\\\\n"
            )
            latex_table = (
                latex_table[:insert_pos]
                + header_line
                + latex_table[insert_pos:]
            )
    return latex_table


def save_visualizations(all_metrics, dataset_name):
    # Create the images directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    images_dir = os.path.join(project_root, "paper", "images")

    # Create directory with proper error handling
    try:
        Path(images_dir).mkdir(parents=True, exist_ok=True)
        print(f"Images will be saved to: {images_dir}")
    except PermissionError:
        print(f"Warning: Permission denied when creating {images_dir}")
        print("Images will not be saved to disk")

    # Check if visualizations exist in all_metrics
    if "visualizations" not in all_metrics:
        print(f"No visualizations found for {dataset_name}")
        return

    visualizations = all_metrics["visualizations"]

    # Iterate through different visualization types (e.g., heatmaps, distribution_plots, etc.)
    for viz_type, viz_objects in visualizations.items():
        if isinstance(viz_objects, list):
            # Multiple plots of the same type
            for i, fig in enumerate(viz_objects, 1):
                filename = f"{dataset_name}_{viz_type}_{i}.pdf"
                filepath = os.path.join(images_dir, filename)
                fig.savefig(filepath, bbox_inches="tight")
                plt.close(fig)  # Close the figure to free memory
                print(f"Saved: {filepath}")

        elif isinstance(viz_objects, dict):
            # Dictionary of plots (like distribution_plots with one per variable)
            for var_name, fig in viz_objects.items():
                # strip varname from slashes etc
                var_name = str(var_name).replace("/", "_").replace("\\", "_")
                filename = f"{dataset_name}_{viz_type}_{var_name}.pdf"
                filepath = os.path.join(images_dir, filename)
                fig.savefig(filepath, bbox_inches="tight")
                plt.close(fig)  # Close the figure to free memory
                print(f"Saved: {filepath}")
        else:
            # Single plot
            fig = viz_objects
            filename = f"{dataset_name}_{viz_type}.pdf"
            filepath = os.path.join(images_dir, filename)
            fig.savefig(filepath, bbox_inches="tight")
            plt.close(fig)  # Close the figure to free memory
            print(f"Saved: {filepath}")
    # --- Add histogram for distance_recon_to_true if present ---
    stats_metrics = all_metrics.get("statistical_similarity", {})
    distance_metrics = stats_metrics.get("distance_recon_to_true", {})
    distances = distance_metrics.get("distances_recon_to_true", None)
    if distances and isinstance(distances, list) and len(distances) > 0:
        # Use percentiles to limit the axis to the majority of data (e.g., 1st to 99th percentile)
        _, upper = np.percentile(distances, [1, 95])
        plt.figure(figsize=(6, 4))
        plt.hist(
            [d for d in distances if d <= upper],
            bins=30,
            color="royalblue",
            alpha=0.8,
        )
        plt.xlabel("Distance to Closest True Record")
        plt.ylabel("Count")
        plt.title(f"Histogram of Distances (Recon → True)\n{dataset_name}")
        plt.xlim(0, upper)
        plt.tight_layout()
        filename = f"{dataset_name}_distance_recon_to_true_hist.pdf"
        filepath = os.path.join(images_dir, filename)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filepath}")


def main():
    import os

    # Ablation study: marginal only, + copula, + MCMC, gold summaries)
    # seed
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    COMPLETE = True

    # Load datasets if not already loaded
    if "datasets" not in locals():
        datasets = load_test_datasets()

    all_aggregated_summaries = []
    dataset_names_for_index = []

    # Configuration for test_data_reconstruction (can be adjusted)
    MAX_ITERATIONS_RECONSTRUCTION = 1500  # As used in your example
    N_SAMPLES_RECONSTRUCTION = 10
    IQR_FOR_RECONSTRUCTION_INPUT = (
        False  # For generating the input summary for reconstruction
    )
    IGNORE_COPULA = False  # Whether to ignore copula in reconstruction
    # which abbrev to add to latex table name
    GOLD = False
    ending = "_baseline"
    if IGNORE_COPULA:
        ending = "_no_copula"
    elif MAX_ITERATIONS_RECONSTRUCTION == 0:
        ending = "_no_mcmc"
    elif GOLD:
        ending = "_gold_summary"
    # Configuration for aggregate_results
    IQR_FOR_AGGREGATION = IQR_FOR_RECONSTRUCTION_INPUT

    if COMPLETE:

        print(f"Starting batch evaluation for {list(datasets.keys())}...")

        original_df, reconstructed_df, summary, cat_var, cont_var = (
            test_anova_expansion(
                "diamonds",
                n_samples=N_SAMPLES_RECONSTRUCTION,
                max_iterations=int(MAX_ITERATIONS_RECONSTRUCTION / 6),
                # Grouped=True,
            )
        )

        visualize_anova_results(
            original_df,
            reconstructed_df,
            cat_var,
            cont_var,
            save_pdfs=True,
            original_color="gold",
            reconstructed_color="red",
            original_name="original_dist_mean",
            reconstructed_name="reconstructed_dist_mean",
        )

        original_df, reconstructed_df, summary, cat_var, cont_var = (
            test_anova_expansion(
                "diamonds",
                IQR=True,
                n_samples=N_SAMPLES_RECONSTRUCTION,
                max_iterations=int(MAX_ITERATIONS_RECONSTRUCTION / 6),
                # Grouped=True,
            )
        )

        visualize_anova_results(
            original_df,
            reconstructed_df,
            cat_var,
            cont_var,
            save_pdfs=True,
            original_color="gold",
            reconstructed_color="green",
            original_name="original_dist_iqr",
            reconstructed_name="reconstructed_dist_iqr",
        )

    if COMPLETE and not GOLD:
        for dataset_name, dataset_dict in datasets.items():
            print(f"\nProcessing dataset: {dataset_name}...")

            try:
                # 1. Run data reconstruction
                # Ensure 'data' key exists
                if "data" not in dataset_dict:
                    print(
                        f"Skipping {dataset_name}: 'data' key not found in dataset_dict."
                    )
                    continue

                original_df, reconstructed_df, summary = (
                    test_data_reconstruction(
                        dataset_name=dataset_name,
                        dataset_info=dataset_dict,
                        max_iterations=MAX_ITERATIONS_RECONSTRUCTION,
                        n_samples=N_SAMPLES_RECONSTRUCTION,
                        IQR=IQR_FOR_RECONSTRUCTION_INPUT,  # This IQR flag is for the input summary generation
                        ignore_copula=IGNORE_COPULA,
                    )
                )

                if reconstructed_df is None or reconstructed_df.empty:
                    print(
                        f"Skipping {dataset_name} due to empty reconstructed_df."
                    )
                    all_aggregated_summaries.append(
                        {
                            "dataset_name": dataset_name,
                            "error": "Reconstruction failed or produced empty data",
                        }
                    )
                    dataset_names_for_index.append(dataset_name)
                    continue

                # 2. Instantiate Evaluation Suite
                current_target = GROUP_MAPPING.get(dataset_name)
                if not current_target:
                    print(
                        f"Warning: No target variable defined in dataset_target_mapping for {dataset_name}. ML utility might be limited."
                    )

                evaluation_suite = DataReconstructionEvaluation(
                    true_data=original_df,
                    reconstructed_data=reconstructed_df,
                    mask_param=summary,  # Assuming no mask for this batch run, can be parameterized
                    dataset_name=dataset_name,
                    dataset_target_mapping=(
                        GROUP_MAPPING if current_target else None
                    ),
                )

                # 3. Perform the full evaluation
                all_metrics = (
                    evaluation_suite.evaluate_tabular_reconstruction()
                )

                save_visualizations(all_metrics, dataset_name)

                # 4. Aggregate the results
                # The IQR flag here determines if median/IQR or mean/std are reported in the aggregate
                metric_summary = evaluation_suite.aggregate_results(
                    all_metrics, IQR=IQR_FOR_AGGREGATION
                )

                # Store the summary
                flattened_summary = {"dataset_name": dataset_name}
                for category, kpis in metric_summary.items():
                    for kpi_name, value in kpis.items():
                        flattened_summary[f"{category}_{kpi_name}"] = value
                all_aggregated_summaries.append(flattened_summary)
                dataset_names_for_index.append(dataset_name)
                print(f"Finished processing {dataset_name}.")

            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")
                all_aggregated_summaries.append(
                    {"dataset_name": dataset_name, "error": str(e)}
                )
                if dataset_name not in dataset_names_for_index:
                    dataset_names_for_index.append(dataset_name)

        # Create a DataFrame from the list of aggregated summaries
        if all_aggregated_summaries:

            latex_table = generate_latex_table(all_aggregated_summaries)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(
                os.path.join(script_dir, "..", "..", "..")
            )
            paper_dir = os.path.join(project_root, "paper")

            # Create directory with proper error handling
            try:
                Path(paper_dir).mkdir(parents=True, exist_ok=True)
                print(f"Table will be saved to: {paper_dir}")
                output_path = os.path.join(
                    paper_dir, f"data_reconstruction_table{ending}.tex"
                )
                with open(output_path, "w") as f:
                    f.write(latex_table)
            except PermissionError:
                print(f"Warning: Permission denied when creating {paper_dir}")
                print("Images will not be saved to disk")
        else:
            print("No summaries were generated.")

    import json

    all_aggregated_summaries = []
    all_aggregated_summaries_clean = []
    dataset_names_for_index = []
    doi_nums = [
        "10.1001_jamanetworkopen.2019.20511",
        "10.1161_JAHA.118.011771",
        "10.1186_s13104-019-4632-2",
        "10.1371_journal.pmed.1002015",
        "10.1371_journal.pmed.1002785",
        "10.1371_journal.pmed.1003621",
        "10.30802_AALAS-JAALAS-23-000028",
    ]
    for doi_num in doi_nums:

        json_path = (
            f"src/text2tabular/data/real/replication/{doi_num}/{doi_num}.json"
        )
        if GOLD:
            json_path = f"src/text2tabular/data/real/replication/{doi_num}/{doi_num}_gold.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        csv_path_raw = (
            f"src/text2tabular/data/real/replication/{doi_num}/{doi_num}.csv"
        )

        csv_path_clean = f"src/text2tabular/data/real/replication/{doi_num}/{doi_num}_clean.csv"

        original_df = pd.read_csv(
            csv_path_raw,
            # dtype={col: "Int64" for col in int_columns},
            na_values=[" ", "."],  # Treat single space as NaN
            keep_default_na=True,
        )

        clean_df = pd.read_csv(
            csv_path_clean,
            # dtype={col: "Int64" for col in int_columns},
            na_values=[" ", "."],  # Treat single space as NaN
            keep_default_na=True,
        )

        parsed_data, reconstructed_df = generate_synthetic_data(
            data,
            n_samples=N_SAMPLES_RECONSTRUCTION,
            seed=seed,
            max_iterations=MAX_ITERATIONS_RECONSTRUCTION,
            ignore_copula=IGNORE_COPULA,
        )
        evaluation_suite = DataReconstructionEvaluation(
            true_data=original_df,
            reconstructed_data=reconstructed_df,
            mask_param=parsed_data,
            dataset_name=f"{doi_num}",
        )

        parsed_data_clean, reconstructed_df_clean = generate_synthetic_data(
            data,
            n_samples=N_SAMPLES_RECONSTRUCTION,
            seed=seed,
            max_iterations=MAX_ITERATIONS_RECONSTRUCTION,
            ignore_copula=IGNORE_COPULA,
        )
        evaluation_suite_clean = DataReconstructionEvaluation(
            true_data=clean_df,
            reconstructed_data=reconstructed_df_clean,
            mask_param=parsed_data_clean,
            dataset_name=f"{doi_num}",
        )

        all_metrics = evaluation_suite.evaluate_tabular_reconstruction()
        metric_summary = evaluation_suite.aggregate_results(all_metrics)
        save_visualizations(all_metrics, doi_num)

        flattened_summary = {"dataset_name": doi_num}
        for category, kpis in metric_summary.items():
            for kpi_name, value in kpis.items():
                flattened_summary[f"{category}_{kpi_name}"] = value
        all_aggregated_summaries.append(flattened_summary)

        all_metrics = evaluation_suite_clean.evaluate_tabular_reconstruction()
        metric_summary = evaluation_suite_clean.aggregate_results(all_metrics)
        save_visualizations(all_metrics, doi_num)

        flattened_summary = {"dataset_name": doi_num}
        for category, kpis in metric_summary.items():
            for kpi_name, value in kpis.items():
                flattened_summary[f"{category}_{kpi_name}"] = value
        all_aggregated_summaries_clean.append(flattened_summary)

    # Generate and save LaTeX table
    latex_table = generate_latex_table(all_aggregated_summaries, abbrv=True)
    # find \label{tab:data_reconstruction} and replace it with \label{tab:real_data_reconstruction}
    latex_table = latex_table.replace(
        "\\label{tab:data_reconstruction}",
        "\\label{tab:real_data_reconstruction}",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    paper_dir = os.path.join(project_root, "paper")
    Path(paper_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        paper_dir, f"real_data_reconstruction_table_raw{ending}.tex"
    )
    with open(output_path, "w") as f:
        f.write(latex_table)

    latex_table = generate_latex_table(
        all_aggregated_summaries_clean, abbrv=True
    )
    # find \label{tab:data_reconstruction} and replace it with \label{tab:real_data_reconstruction}
    latex_table = latex_table.replace(
        "\\label{tab:data_reconstruction}",
        "\\label{tab:real_data_reconstruction_clean}",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    paper_dir = os.path.join(project_root, "paper")
    Path(paper_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        paper_dir, f"real_data_reconstruction_table_clean{ending}.tex"
    )
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to: {output_path}")

    print("Aggregated metrics summary:")
    print(metric_summary)


if __name__ == "__main__":
    main()
