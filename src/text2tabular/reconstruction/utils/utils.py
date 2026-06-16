import numpy as np
import pandas as pd
from math import ceil

CONSISTENT = "consistent"
OVERWRITTEN = "overwritten"
NO_GROUPS = "no_groups"


def split_by_group(values, groups):
    """
    Split values into a list of arrays, one per unique group in `groups`.
    Returns: list of arrays, list of group labels (in order)
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    split = [values[groups == g] for g in unique_groups]
    return split, unique_groups


def lowercase_json_keys_and_values(obj):
    if isinstance(obj, dict):
        return {
            (
                k.lower() if isinstance(k, str) else k
            ): lowercase_json_keys_and_values(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [lowercase_json_keys_and_values(elem) for elem in obj]
    elif isinstance(obj, str):
        return obj.lower()
    else:
        return obj


def parse_iqr(iqr):
    """
    Parse IQR value from string (e.g., "20.1-24.4", "101–248", "101—248"), list, or float.
    Returns a tuple (q1, q3) or None if not parseable.
    """
    if iqr is None:
        return None
    if isinstance(iqr, (list, tuple)) and len(iqr) == 2:
        return float(iqr[0]), float(iqr[1])
    if isinstance(iqr, str):
        # Normalize all dash-like characters to hyphen
        iqr_norm = iqr.replace("\u2013", "-").replace("\u2014", "-")
        if "-" in iqr_norm:
            parts = iqr_norm.split("-")
            if len(parts) == 2:
                try:
                    return float(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    pass
    if isinstance(iqr, (int, float)):
        return float(iqr), None
    return None


def process_binary_counts(binary_data, group_sizes, consistency_score):
    """
    Processes each binary variable for all groups:
    - If value is int, assume already processed and leave as is.
    - If value is dict with 'count' and 'denominator', process:
        - If denominator is null/missing, use group size.
        - Scale counts/denominator to group size if needed.
        - If count == group_size but denominator > group_size, set denominator to group_size, decrease consistency.
        - Adjust consistency_score based on match.
    Skips problematic entries and continues.
    Returns processed binary_data and updated consistency_score.
    """
    processed = {}
    for var, group_dict in binary_data.items():
        processed[var] = {}
        for group, entry in group_dict.items():
            try:
                # Already processed: just a count
                if isinstance(entry, int):
                    processed[var][group] = entry
                    continue
                # Empty or missing
                if not entry or "count" not in entry:
                    continue
                count = entry.get("count")
                denominator = entry.get("denominator")
                group_size = group_sizes.get(group)
                if group_size is None:
                    continue  # skip if group size missing

                if denominator is not None and denominator == group_size:
                    consistency_score += 1
                # Special case: count == group_size but denominator > group_size
                elif (
                    denominator is not None
                    and denominator > group_size
                    and count == group_size
                ):
                    consistency_score -= 1
                    denominator = group_size
                    # Do not scale count

                # Adjust consistency_score

                elif denominator is not None:
                    consistency_score -= 1

                # Use group size if denominator is null/missing
                if denominator is None:
                    denominator = group_size

                # Scale count and denominator to group size
                if (
                    denominator != group_size
                    and denominator
                    and denominator > 0
                    and not (count == group_size and denominator > group_size)
                ):
                    percent = count / denominator
                    count = percent * group_size

                processed[var][group] = int(ceil(count))
            except Exception:
                continue  # skip problematic entries
    return processed, consistency_score


def parse_json(json_input):
    """
    Parse the JSON input to extract study information and meta-information.
    All string keys and values are converted to lowercase recursively.

    Args:
        json_input: Dictionary containing the study data

    Returns:
        Dictionary with parsed study information and meta-information
    """
    # Recursively lowercase all string keys and values first
    processed_json = lowercase_json_keys_and_values(json_input)

    # Basic info
    study_size = processed_json.get("study_size")
    groups = processed_json.get(
        "groups", []
    )  # Already lowercase list of strings
    group_sizes = processed_json.get(
        "group_sizes", {}
    )  # Keys already lowercase
    # overwrite study size with sum of group sizes if available
    # and add enum for consistency check

    # Consistency check
    consistency_score = 0
    if group_sizes:
        group_sum = sum(group_sizes.values())
        if study_size == group_sum:
            consistency = CONSISTENT
            consistency_score += 1
        else:
            study_size = group_sum
            consistency = OVERWRITTEN
    else:
        consistency = NO_GROUPS

    # Variables (keys and nested keys already lowercase)
    variables = processed_json.get("variables", {})

    variables["binary"], consistency_score = process_binary_counts(
        variables.get("binary", {}), group_sizes, consistency_score
    )

    # parse counts in a secure way: if denominator is null, use group size as denominator
    # scale counts/denominator to group size and if inconsistent, decrease consistency score
    # ensure a simple format for counts (e.g., "group_1": 15, "group_2": 20)

    # if IQR or ci95 in ordinal or continuous variables for each group, turn it into unified format with parse_iqr

    # --- Parse IQR and CI95 for each variable/group ---
    for var_type in variables:
        var_type_obj = variables[var_type]
        if not isinstance(var_type_obj, dict):
            continue
        for _, var_groups in var_type_obj.items():
            if not isinstance(var_groups, dict):
                continue
            for _, stats in var_groups.items():
                if not isinstance(stats, dict):
                    continue
                # Handle IQR
                if "iqr" in stats:
                    iqr_parsed = parse_iqr(stats["iqr"])
                    if iqr_parsed and iqr_parsed[1] is not None:
                        stats["q1"], stats["q3"] = iqr_parsed
                        stats["iqr"] = stats["q3"] - stats["q1"]
                # Handle CI95
                if "ci95" in stats:
                    ci_parsed = parse_iqr(stats["ci95"])
                    if ci_parsed and ci_parsed[1] is not None:
                        stats["ci95_low"], stats["ci95_high"] = ci_parsed
                        stats["ci95"] = stats["ci95_high"] - stats["ci95_low"]

    # Statistical tests (test_type, variable names, group names already lowercase)
    statistical_tests = processed_json.get("statistical_tests", [])
    valid_groups = set(groups)
    filtered_tests = []
    # Extract correlations from statistical tests (already processed above)
    correlations = []
    for test in statistical_tests:
        test_groups = set(test.get("groups", []))
        if not test_groups.issubset(valid_groups):
            continue
        if test.get("test_type", "") in ["pearson", "spearman"]:
            # Extract the correlation coefficient which might be under 'test_statistic' or 'effect_size'
            corr_coeff = test.get("test_statistic")
            if corr_coeff is None and test.get("effect_size") is not None:
                corr_coeff = test.get("effect_size")

            correlations.append(
                {
                    "variables": test.get("variables", []),
                    "test_type": test.get("test_type"),
                    "correlation_coefficient": corr_coeff,
                    "p_value": test.get("p_value"),
                    "groups": test.get("groups", []),
                }
            )
        else:
            # For other tests, just keep the test as is
            filtered_tests.append(test)

    statistical_tests = filtered_tests
    # Detect if data is grouped
    is_grouped = len(groups) > 1

    # Count how many JSONs (support for list of JSONs)
    # Note: lowercase_json_values handles lists, so check original input type
    if isinstance(json_input, list):
        n_jsons = len(json_input)
    else:
        n_jsons = 1

    # Detect if there is an ANOVA test and extract group means if present
    has_anova = False
    anova_group_means = []
    for test in statistical_tests:
        if test.get("test_type", "") == "one_way_anova":
            has_anova = True
            if "group_means" in test:
                anova_group_means.append(
                    test["group_means"]
                )  # Means are numeric, not lowercased

    return {
        "study_size": study_size,
        "groups": groups,
        "group_sizes": group_sizes,
        "variables": variables,
        "statistical_tests": statistical_tests,
        "correlations": correlations,
        "is_grouped": is_grouped,
        "n_jsons": n_jsons,
        "has_anova": has_anova,
        "anova_group_means": anova_group_means,
        "consistency": consistency,
        "consistency_score": consistency_score,
    }


def combine_synthetic_data(synthetic_data, groups):
    """
    Combine synthetic data into a pandas DataFrame, handling potentially different group sizes.

    Args:
        synthetic_data: Dictionary where keys are (variable_name, group_name) tuples
                        and values are numpy arrays of synthetic data for that variable/group.
        groups: List of group names present in the data.

    Returns:
        pandas DataFrame containing the combined synthetic data.
    """
    data_rows = []
    all_vars = sorted(
        list(set(var for var, group in synthetic_data.keys()))
    )  # Ensure consistent order

    # For each group, determine its specific size and create rows
    for group in groups:
        # Find the size for this specific group by looking at the first variable available for it
        group_specific_size = 0
        first_var_for_group = None
        for var in all_vars:
            if (var, group) in synthetic_data:
                # Check if the value is a list or array and has length
                val = synthetic_data[(var, group)]
                if hasattr(val, "__len__"):
                    group_specific_size = len(val)
                    first_var_for_group = var
                    break  # Found the size for this group

        if first_var_for_group is None:
            print(
                f"Warning: No data found for group '{group}' in synthetic_data. Skipping."
            )
            continue  # Skip this group if no data was found

        # Iterate up to the size specific to this group
        for i in range(group_specific_size):
            row = {"group": group}
            for var in all_vars:
                key = (var, group)
                if key in synthetic_data:
                    # Ensure the array for this variable/group has enough elements
                    if i < len(synthetic_data[key]):
                        row[var] = synthetic_data[key][i]
                    else:
                        # This case should ideally not happen if all vars for a group have the same size
                        print(
                            f"Warning: Data length mismatch for var '{var}', group '{group}' at index {i}. Setting NaN."
                        )
                        row[var] = np.nan
                else:
                    # Handle case where a variable might be missing for a group (though less likely with generation)
                    row[var] = np.nan
            data_rows.append(row)

    # Create DataFrame with specified column order
    column_order = ["group"] + all_vars
    df = pd.DataFrame(data_rows, columns=column_order)
    return df


def make_positive_definite(matrix, eps=1e-6):
    """Ensure matrix is positive definite by adding eps * I shifted by min eigenvalue."""
    min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
    # If matrix is already positive definite or very close, return it
    if min_eig > eps:
        return matrix

    # Add a small multiple of identity matrix, shifted by the minimum eigenvalue
    n = matrix.shape[0]
    # Calculate the shift needed, add a small epsilon for numerical stability
    shift = max(0, -min_eig) + eps
    adjusted_matrix = matrix + shift * np.eye(n)

    return adjusted_matrix
