import pandas as pd
from typing import List, Tuple, Dict, Optional


def classify_columns(
    true_data_eval: pd.DataFrame,
    reconstructed_data_eval: pd.DataFrame,
    common_cols: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Classifies common columns into numerical and categorical.
    Numeric columns with <= 2 unique values in both true and reconstructed data are
    treated as categorical.
    """
    initial_numerical_cols = [
        col
        for col in common_cols
        if pd.api.types.is_numeric_dtype(true_data_eval[col])
        and col in reconstructed_data_eval.columns
        and pd.api.types.is_numeric_dtype(reconstructed_data_eval[col])
    ]
    initial_categorical_cols = [
        col
        for col in common_cols
        if not pd.api.types.is_numeric_dtype(true_data_eval[col])
        and col in reconstructed_data_eval.columns
        and not pd.api.types.is_numeric_dtype(reconstructed_data_eval[col])
    ]

    final_numerical_cols = []
    final_categorical_cols = list(initial_categorical_cols)

    for col in initial_numerical_cols:
        is_binary_true = true_data_eval[col].dropna().nunique() <= 2
        is_binary_recon = reconstructed_data_eval[col].dropna().nunique() <= 2

        if is_binary_true and is_binary_recon:
            if col not in final_categorical_cols:
                final_categorical_cols.append(col)
        else:
            final_numerical_cols.append(col)

    return final_numerical_cols, final_categorical_cols


def apply_mask_to_true_data(
    true_data: pd.DataFrame, mask: Optional[Dict], is_masked: bool
) -> pd.DataFrame:
    if (
        is_masked
        and mask
        and isinstance(mask.get("_masked_variable_list_internal"), list)
    ):
        masked_variable_list = mask["_masked_variable_list_internal"]
        # Ensure all columns in masked_variable_list are actually in true_data.columns
        # and that the list only contains strings.
        valid_masked_cols = [
            str(col)
            for col in masked_variable_list
            if str(col) in true_data.columns
        ]
        if (
            not valid_masked_cols
        ):  # If no valid columns to select, return a copy of the original
            return true_data.copy()
        return true_data[valid_masked_cols].copy()
    return true_data.copy()
