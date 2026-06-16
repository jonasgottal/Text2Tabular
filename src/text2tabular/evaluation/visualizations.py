import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import matthews_corrcoef
from typing import Dict, Optional, Any, List


from text2tabular.evaluation.evaluation_utils import apply_mask_to_true_data
from text2tabular.evaluation.test_data_utils import GROUP_MAPPING

from pathlib import Path
import os


plt.rcParams.update(
    {
        "font.size": 26,  # Default text size
        "axes.titlesize": 26,  # Title font size
        "axes.labelsize": 24,  # Axis label font size
        "xtick.labelsize": 22,  # X tick label size
        "ytick.labelsize": 22,  # Y tick label size
        "legend.fontsize": 20,  # Legend font size
    }
)


def visualize_anova_results(
    original_df,
    reconstructed_df,
    cat_var,
    cont_var,
    save_pdfs=False,
    original_color=None,
    reconstructed_color=None,
    original_name="original_distribution",
    reconstructed_name="reconstructed_distribution",
):
    """
    Visualize the ANOVA relationship in both original and reconstructed data.
    Handles case differences between original and reconstructed labels.

    Args:
        original_df: Original dataframe
        reconstructed_df: Reconstructed dataframe
        cat_var: Categorical variable name
        cont_var: Continuous variable name
    """
    if reconstructed_df is None:
        print("Cannot visualize: Reconstruction failed")
        return
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
        save_pdfs = False
    # Get reconstructed data with overall group
    recon_df = reconstructed_df.copy()

    # Create lowercase version of categorical variable in original data for comparison
    original_df = original_df.copy()
    original_df[f"{cat_var}_lower"] = (
        original_df[cat_var].astype(str).str.lower()
    )

    # Determine category order from original dataset
    category_order = list(original_df[cat_var].unique())
    category_order_lower = [str(c).lower() for c in category_order]
    print(f"Original categories: {category_order}")
    print(f"Lowercase categories: {category_order_lower}")

    # Check if reconstructed data has the lowercase categories
    recon_categories = recon_df[cat_var].unique()
    print(f"Reconstructed categories: {recon_categories}")

    # Create mapping from lowercase to original case
    case_mapping = {str(c).lower(): c for c in category_order}

    # For plotting reconstructed data, we need to convert to original case if possible
    try:
        recon_df[f"{cat_var}_orig_case"] = recon_df[cat_var].map(case_mapping)
        # If mapping fails (NaN values), fall back to original lowercase values
        recon_df[f"{cat_var}_orig_case"].fillna(
            recon_df[cat_var], inplace=True
        )
        recon_cat_var = f"{cat_var}_orig_case"
    except:
        print(
            "Warning: Could not map reconstructed categories back to original case"
        )
        recon_cat_var = cat_var

    # Extract data from both dataframes
    plt.figure(figsize=(12, 7))

    # 1. Box plots to compare distributions by group
    plt.subplot(2, 2, 1)
    sns.boxplot(data=original_df, x=cat_var, y=cont_var, order=category_order)
    plt.title(f"Original: {cont_var} by {cat_var}")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    sns.boxplot(
        data=recon_df,
        x=recon_cat_var,
        y=cont_var,
        order=category_order,  # Use original order
    )
    plt.title(f"Reconstructed: {cont_var} by {cat_var}")
    plt.xticks(rotation=45)

    # 2. Violin plots for more detailed distribution view
    plt.subplot(2, 2, 3)
    # Create a separate figure for saving
    if save_pdfs:
        fig3 = plt.figure(figsize=(12, 7))
    sns.violinplot(
        data=original_df,
        x=cat_var,
        y=cont_var,
        order=category_order,
        color=original_color if original_color else None,
    )
    plt.title(f"Original Distribution: {cont_var} by {cat_var}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save plot 3 as PDF
    if save_pdfs:
        filename = f"{original_name}_{cat_var}_{cont_var}.pdf"
        filepath = os.path.join(images_dir, filename)
        fig3.savefig(filepath, bbox_inches="tight")
        plt.close(fig3)

    plt.subplot(2, 2, 4)
    # Create a separate figure for saving
    if save_pdfs:
        fig4 = plt.figure(figsize=(12, 7))
    sns.violinplot(
        data=recon_df,
        x=recon_cat_var,
        y=cont_var,
        order=category_order,
        color=reconstructed_color if reconstructed_color else None,
    )
    plt.title(f"Reconstructed Distribution: {cont_var} by {cat_var}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save plot 4 as PDF
    if save_pdfs:
        filename = f"{reconstructed_name}_{cat_var}_{cont_var}.pdf"
        filepath = os.path.join(images_dir, filename)
        fig4.savefig(filepath, bbox_inches="tight")
        plt.close(fig4)

    plt.tight_layout()
    # plt.show()

    # 3. Group means comparison
    plt.figure(figsize=(10, 6))

    # For means comparison, we need case-insensitive matching
    # Calculate group means for original data
    orig_means = original_df.groupby(cat_var)[cont_var].mean()

    # For reconstructed data, first group by lowercase categories
    # Then map results back to original case for display
    recon_means_dict = {}
    for orig_cat, lower_cat in zip(category_order, category_order_lower):
        cat_data = recon_df[
            recon_df[cat_var].str.lower() == lower_cat.lower()
        ][cont_var]
        if len(cat_data) > 0:
            recon_means_dict[orig_cat] = cat_data.mean()
        else:
            print(
                f"Warning: No data for category '{orig_cat}' in reconstructed data"
            )
            recon_means_dict[orig_cat] = np.nan

    recon_means = pd.Series(recon_means_dict)

    # Ensure both have same categories by reindexing with the fixed order
    orig_means = orig_means.reindex(category_order)
    recon_means = recon_means.reindex(category_order)

    # Plot means comparison
    comparison_df = pd.DataFrame(
        {"Original": orig_means, "Reconstructed": recon_means}
    ).reset_index()
    comparison_df.columns = [
        cat_var,
        "Original",
        "Reconstructed",
    ]  # Rename index col

    comparison_df_melted = pd.melt(
        comparison_df,
        id_vars=[cat_var],
        value_vars=["Original", "Reconstructed"],
        var_name="Source",
        value_name=cont_var,
    )

    sns.barplot(
        data=comparison_df_melted,
        x=cat_var,
        y=cont_var,
        hue="Source",
        order=category_order,
    )
    plt.title(f"Comparison of Mean {cont_var} by {cat_var}")
    plt.xticks(rotation=45)
    plt.legend(title="Data Source")
    plt.tight_layout()
    # plt.show()

    # Return the mean comparison data for further analysis
    return comparison_df


def _calculate_correlation_matrix_for_heatmap(
    df_to_corr: pd.DataFrame,
    cols_for_matrix: List[str],
    num_cols_list: List[str],
    cat_binary_cols_list: List[str],
) -> pd.DataFrame:
    corr_matrix_calc = pd.DataFrame(
        np.nan, index=cols_for_matrix, columns=cols_for_matrix
    )
    df_local_corr = df_to_corr[cols_for_matrix].copy()

    for i, c1 in enumerate(cols_for_matrix):
        for j, c2 in enumerate(cols_for_matrix):
            if i == j:
                corr_matrix_calc.loc[c1, c2] = 1.0
                continue
            if j < i:  # Symmetric matrix, value already computed
                corr_matrix_calc.loc[c1, c2] = corr_matrix_calc.loc[c2, c1]
                continue

            temp_df_aligned = df_local_corr[[c1, c2]].dropna()
            s1_a, s2_a = temp_df_aligned[c1], temp_df_aligned[c2]
            current_corr = np.nan

            if len(s1_a) < 2 or s1_a.nunique() == 0 or s2_a.nunique() == 0:
                current_corr = np.nan
            elif s1_a.nunique() == 1 or s2_a.nunique() == 1:
                current_corr = np.nan
            else:
                is_c1_num, is_c2_num = c1 in num_cols_list, c2 in num_cols_list
                is_c1_cat_bin, is_c2_cat_bin = (
                    c1 in cat_binary_cols_list,
                    c2 in cat_binary_cols_list,
                )
                try:
                    if is_c1_num and is_c2_num:
                        res = spearmanr(s1_a, s2_a)
                        # sres = spearmanr(s1_a, s2_a, nan_policy="omit")
                        current_corr = res.correlation
                    elif is_c1_num and is_c2_cat_bin:
                        cat_fact, _ = pd.factorize(s2_a)
                        if len(np.unique(cat_fact)) == 2:
                            current_corr, _ = pointbiserialr(cat_fact, s1_a)
                    elif is_c1_cat_bin and is_c2_num:
                        cat_fact, _ = pd.factorize(s1_a)
                        if len(np.unique(cat_fact)) == 2:
                            current_corr, _ = pointbiserialr(cat_fact, s2_a)
                    elif is_c1_cat_bin and is_c2_cat_bin:
                        # Factorize both series. pd.factorize assigns a numerical label (0, 1, ...)
                        # to each unique category.
                        s1_fact, _ = pd.factorize(s1_a)
                        s2_fact, _ = pd.factorize(s2_a)

                        # The nunique checks earlier should ensure these are binary (or effectively binary
                        # after dropna and selection into cat_binary_cols_list).
                        # matthews_corrcoef expects binary inputs.
                        # If after factorization, they are not binary (e.g. only one unique value remains
                        # in s1_fact or s2_fact), MCC might be undefined or 0.
                        # The s1_a.nunique() == 1 checks should catch this, but an extra check here
                        # or reliance on MCC's behavior with constant inputs is needed.
                        # sklearn's MCC returns 0 if any of the four values in the confusion matrix are 0
                        # (e.g. if one variable is constant), and issues a RuntimeWarning.
                        # We can suppress the warning if 0 is an acceptable outcome in such cases.
                        with np.errstate(
                            divide="ignore", invalid="ignore"
                        ):  # Suppress potential warnings
                            current_corr = matthews_corrcoef(s1_fact, s2_fact)
                            if np.isnan(
                                current_corr
                            ):  # MCC can be NaN if a row/col in confusion matrix is all zero
                                current_corr = (
                                    0.0  # Or handle as np.nan if preferred
                                )
                except (ValueError, RuntimeWarning, ZeroDivisionError):
                    current_corr = np.nan
            corr_matrix_calc.loc[c1, c2] = current_corr
            corr_matrix_calc.loc[c2, c1] = current_corr
    return corr_matrix_calc


def generate_visualizations(
    true_data_base: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    mask: Optional[Dict],
    is_masked: bool,
    dataset_name: Optional[str] = None,
) -> Dict:
    """
    Generate visualizations to compare true and reconstructed data.
    Returns a dictionary of matplotlib figures.
    """
    visualizations: Dict[str, Any] = {
        "distribution_plots": {},
        "heatmaps": "Placeholder for heatmap generation.",
    }

    true_df_viz = apply_mask_to_true_data(true_data_base, mask, is_masked)
    recon_df_viz = reconstructed_data_eval.copy()

    group_var: Optional[str] = None
    if dataset_name and dataset_name in GROUP_MAPPING:
        potential_group_var = GROUP_MAPPING[dataset_name].lower()
        if potential_group_var in true_df_viz.columns:
            group_var = potential_group_var

    all_possible_group_names = set()
    if group_var:
        all_possible_group_names.update(
            true_df_viz[group_var].dropna().unique()
        )
        if group_var in recon_df_viz.columns:
            all_possible_group_names.update(
                recon_df_viz[group_var].dropna().unique()
            )
    if not all_possible_group_names:
        all_possible_group_names.add("overall")

    sorted_group_names = sorted(list(all_possible_group_names))
    palette = sns.color_palette(n_colors=len(sorted_group_names))
    color_map = {name: palette[i] for i, name in enumerate(sorted_group_names)}

    common_numeric_cols = [
        col
        for col in true_df_viz.columns
        if col in recon_df_viz.columns
        and pd.api.types.is_numeric_dtype(true_df_viz[col])
        and pd.api.types.is_numeric_dtype(recon_df_viz[col])
        and col != group_var
    ]

    for col in common_numeric_cols:
        fig, ax = plt.subplots(figsize=(12, 7))
        if group_var:
            for group_val in sorted(true_df_viz[group_var].dropna().unique()):
                subset = true_df_viz[true_df_viz[group_var] == group_val]
                if not subset.empty:
                    sns.kdeplot(
                        subset[col],
                        label=f"Original: {group_val}",
                        ax=ax,
                        color=color_map.get(str(group_val), "gray"),
                    )
        else:
            sns.kdeplot(
                true_df_viz[col],
                label="Original",
                ax=ax,
                color=color_map.get("overall", "blue"),
            )

        recon_has_group_var = group_var and group_var in recon_df_viz.columns
        if recon_has_group_var:
            for group_val_recon in sorted(
                recon_df_viz[group_var].dropna().unique()
            ):
                subset_recon = recon_df_viz[
                    recon_df_viz[group_var] == group_val_recon
                ]
                if not subset_recon.empty:
                    sns.kdeplot(
                        subset_recon[col],
                        label=f"Reconstructed: {group_val_recon}",
                        linestyle="--",
                        ax=ax,
                        color=color_map.get(str(group_val_recon), "black"),
                    )
        else:
            sns.kdeplot(
                recon_df_viz[col],
                label="Reconstructed",
                linestyle="--",
                ax=ax,
                color=color_map.get("overall", "red"),
            )
        ax.set_title(
            f"Distribution Comparison: {col}"
            + (f" (Grouped by {group_var})" if group_var else "")
        )
        ax.legend()
        plt.tight_layout()
        visualizations["distribution_plots"][col] = fig
        # plt.close(fig) # Keep figures open to be returned

    common_cols_for_heatmap_eval = list(
        set(true_df_viz.columns) & set(recon_df_viz.columns)
    )
    num_cols_heatmap, cat_cols_heatmap_binary = [], []
    for col_h in common_cols_for_heatmap_eval:
        is_numeric_true = pd.api.types.is_numeric_dtype(true_df_viz[col_h])
        is_numeric_recon = pd.api.types.is_numeric_dtype(recon_df_viz[col_h])
        if is_numeric_true and is_numeric_recon:
            num_cols_heatmap.append(col_h)
        elif not is_numeric_true and not is_numeric_recon:
            if (
                true_df_viz[col_h].dropna().nunique() <= 2
                and recon_df_viz[col_h].dropna().nunique() <= 2
            ):
                cat_cols_heatmap_binary.append(col_h)

    all_heatmap_cols = sorted(
        list(set(num_cols_heatmap + cat_cols_heatmap_binary))
    )

    if len(all_heatmap_cols) > 1:
        true_corr_matrix = _calculate_correlation_matrix_for_heatmap(
            true_df_viz,
            all_heatmap_cols,
            num_cols_heatmap,
            cat_cols_heatmap_binary,
        )
        recon_corr_matrix = _calculate_correlation_matrix_for_heatmap(
            recon_df_viz,
            all_heatmap_cols,
            num_cols_heatmap,
            cat_cols_heatmap_binary,
        )
        combined_matrix = pd.DataFrame(
            np.nan, index=all_heatmap_cols, columns=all_heatmap_cols
        )
        for r_idx, row_col in enumerate(all_heatmap_cols):
            for c_idx, col_col in enumerate(all_heatmap_cols):
                if r_idx == c_idx:
                    combined_matrix.loc[row_col, col_col] = 1.0
                elif r_idx < c_idx:
                    combined_matrix.loc[row_col, col_col] = (
                        true_corr_matrix.loc[row_col, col_col]
                    )
                else:
                    combined_matrix.loc[row_col, col_col] = (
                        recon_corr_matrix.loc[row_col, col_col]
                    )

        fig_heatmap, ax_heatmap = plt.subplots(
            figsize=(
                max(12, len(all_heatmap_cols)),
                max(10, len(all_heatmap_cols) * 0.8),
            )
        )
        sns.heatmap(
            combined_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            fmt=".2f",
            ax=ax_heatmap,
            # annot_kws={"size": 10},
        )
        ax_heatmap.set_title(
            "Correlation Heatmap\n(Upper: True Data, Lower: Reconstructed Data)"
        )

        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)
        ax_heatmap.set_xticklabels(
            ax_heatmap.get_xticklabels(), rotation=45, ha="right"
        )

        plt.tight_layout()
        visualizations["heatmaps"] = fig_heatmap
        # plt.close(fig_heatmap) # Keep figures open
    else:
        visualizations["heatmaps"] = (
            "Skipped: Not enough suitable (numeric or binary categorical) common columns for heatmap."
        )
    return visualizations
