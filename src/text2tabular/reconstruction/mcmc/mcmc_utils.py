import numpy as np
from text2tabular.reconstruction.statistics.core import (
    robust_relation,
)


def calculate_difference(
    stats,
    corrs,
    target_stats,
    target_corrs,
    target_tests=None,
    current_tests=None,
):
    """
    Calculate the total difference between the given statistics and the target statistics,
    including statistical tests if provided.
    The total difference is a sum of weighted sums of errors from different components.

    Args:
        stats: Dictionary of current or proposed statistics
        corrs: Dictionary of current or proposed correlations
        target_stats: Dictionary of target statistics
        target_corrs: Dictionary of target correlations
        target_tests: List of target statistical test dicts (optional)
        current_tests: List of current statistical test dicts (optional)

    Returns:
        Total difference
    """
    # Multipliers for sum of component errors
    mult_mean_sum = 1.0
    mult_std_sum = 1.5
    mult_count_sum = 0.1
    mult_corr_metric_sum = (
        1.0  # For correlation coefficients and correlation test statistics
    )
    mult_other_test_sum = (
        0.25  # For other test statistics (e.g., t-stat, chi2)
    )
    # Accumulators for sum of errors
    mean_diff_sum = 0.0
    std_diff_sum = 0.0
    count_diff_sum = 0.0
    corr_metric_diff_sum = 0.0
    other_test_diff_sum = 0.0

    # --- 1. Differences from target_stats (means, stds, counts) ---
    for key, target_val_dict in target_stats.items():
        if key in stats and stats[key] is not None:
            current_val_dict = stats[key]
            if (
                "mean" in target_val_dict
                and target_val_dict["mean"] is not None
                and "mean" in current_val_dict
            ):
                target_mean = target_val_dict["mean"]
                current_mean = current_val_dict["mean"]
                mean_diff_sum += abs(target_mean - current_mean) / max(
                    0.001, abs(target_mean)
                )

                if (
                    "std" in target_val_dict
                    and target_val_dict["std"] is not None
                    and "std" in current_val_dict
                ):
                    target_std = target_val_dict["std"]
                    current_std = current_val_dict["std"]
                    std_diff_sum += abs(target_std - current_std) / max(
                        0.001, abs(target_std)
                    )
            elif (
                "median" in target_val_dict
                and target_val_dict["median"] is not None
                and "median" in current_val_dict
            ):
                target_median = target_val_dict["median"]
                current_median = current_val_dict["median"]
                mean_diff_sum += abs(target_median - current_median) / max(
                    0.001, abs(target_median)
                )

                if (
                    "iqr" in target_val_dict
                    and target_val_dict["iqr"] is not None
                    and "iqr" in current_val_dict
                ):
                    target_iqr = target_val_dict["iqr"]
                    current_iqr = current_val_dict["iqr"]
                    std_diff_sum += abs(target_iqr - current_iqr) / max(
                        0.001, abs(target_iqr)
                    )
                elif (
                    "q1" in target_val_dict
                    and target_val_dict["q1"] is not None
                    and "q1" in current_val_dict
                    and "q3" in target_val_dict
                    and target_val_dict["q3"] is not None
                    and "q3" in current_val_dict
                ):
                    target_q1 = target_val_dict["q1"]
                    current_q1 = current_val_dict["q1"]
                    target_q3 = target_val_dict["q3"]
                    current_q3 = current_val_dict["q3"]
                    target_iqr = target_q3 - target_q1
                    current_iqr = current_q3 - current_q1
                    std_diff_sum += abs(target_iqr - current_iqr) / max(
                        0.001, abs(target_iqr)
                    )

            elif (
                "categories" in target_val_dict
                and "categories" in current_val_dict
            ):
                target_categories = target_val_dict["categories"]
                target_counts_list = target_val_dict["counts"]
                current_categories = current_val_dict["categories"]
                current_counts_map = dict(
                    zip(current_categories, current_val_dict["counts"])
                )

                for t_cat, t_count in zip(
                    target_categories, target_counts_list
                ):
                    c_count = current_counts_map.get(t_cat, 0)
                    count_diff_sum += abs(t_count - c_count) / max(
                        1, t_count
                    )  # Denominator max 1 for counts
            elif (
                "count" in target_val_dict and "count" in current_val_dict
            ):  # For binary
                target_count_val = target_val_dict["count"]
                current_count_val = current_val_dict["count"]
                count_diff_sum += abs(
                    target_count_val - current_count_val
                ) / max(1, target_count_val)

    # --- 2. Differences from target_corrs (correlation coefficients) ---
    for (var1, var2, group), target_corr_coeff in target_corrs.items():
        key_tuple = (var1, var2, group)  # Ensure key is a tuple
        if key_tuple in corrs and "test_statistic" in corrs[key_tuple]:
            current_corr_coeff = corrs[key_tuple]["test_statistic"]
            corr_metric_diff_sum += (
                (target_corr_coeff**4)
                * abs(target_corr_coeff - current_corr_coeff)
                / max(0.001, abs(target_corr_coeff))
            )

    # --- 3. Differences from statistical tests (target_tests vs current_tests) ---
    if target_tests is not None and current_tests is not None:
        for t_test, c_test in zip(target_tests, current_tests):
            if (
                t_test.get("test_type") == c_test.get("test_type")
                and t_test.get("variables") == c_test.get("variables")
                and t_test.get("groups") == c_test.get("groups")
            ):

                if "test_statistic" in t_test and "test_statistic" in c_test:
                    target_stat = t_test["test_statistic"]
                    current_stat = c_test["test_statistic"]

                    if (
                        t_test.get("test_type") in ["pearson", "spearman"]
                        and current_stat is not np.nan
                    ):
                        corr_metric_diff_sum += (
                            (target_corr_coeff**2)
                            * abs(target_stat - current_stat)
                            / max(0.001, abs(target_stat))
                        )
                    else:
                        other_test_diff_sum += abs(
                            target_stat - current_stat
                        ) / max(0.001, abs(target_stat))

                if (
                    t_test.get("test_type") == "one_way_anova"
                    and "group_means" in t_test
                    and "group_means" in c_test
                    and isinstance(t_test["group_means"], list)
                    and isinstance(c_test["group_means"], list)
                ):
                    if len(t_test["group_means"]) == len(
                        c_test["group_means"]
                    ):
                        for tm, cm in zip(
                            t_test["group_means"], c_test["group_means"]
                        ):
                            mean_diff_sum += abs(tm - cm) / max(0.001, abs(tm))

    # --- Total difference is the sum of weighted sums of errors ---
    total_diff = (
        (mult_mean_sum * mean_diff_sum)
        + (mult_std_sum * std_diff_sum)
        + (mult_count_sum * count_diff_sum)
        + (mult_corr_metric_sum * corr_metric_diff_sum)
        + (mult_other_test_sum * other_test_diff_sum)
    )

    return total_diff


def calculate_target_stats(variables):
    """
    Calculate target statistics from the given variable specifications.
    """
    target_stats = {}
    for var_type in ["ordinal", "continuous"]:
        if var_type in variables:
            for var, groups_dict in variables[var_type].items():
                for group, stats in groups_dict.items():
                    # Initialize the dictionary for this variable/group pair
                    target_stats[(var, group)] = {}

                    # Now we can safely add statistics
                    for stat_name in [
                        "q1",
                        "median",
                        "q3",
                        "iqr",
                        "mean",
                        "std",
                    ]:
                        if stat_name in stats:
                            target_stats[(var, group)][stat_name] = stats[
                                stat_name
                            ]
    # Categorical: expects a dict of {category: count} in each group
    if "categorical" in variables:
        for var, groups_dict in variables["categorical"].items():
            for group, stats in groups_dict.items():
                categories = list(stats.keys())
                counts = list(stats.values())
                target_stats[(var, group)] = {
                    "categories": categories,
                    "counts": counts,
                }
    # Binary: expects a count per group (int)
    if "binary" in variables:
        for var, groups_dict in variables["binary"].items():
            for group, count in groups_dict.items():
                target_stats[(var, group)] = {"count": count}
    return target_stats


def calculate_target_corrs(correlations):
    """
    Extract target correlations from the correlations list.

    Args:
        correlations: List of dicts specifying correlation tests.

    Returns:
        Dictionary of target correlations, symmetric in both directions, per group.
    """
    target_corrs = {}
    for corr in correlations:
        var1, var2 = corr["variables"]
        corr_value = corr[
            "correlation_coefficient"
        ]  # Changed from test_statistic
        for group in corr.get("groups", []):
            target_corrs[(var1, var2, group)] = corr_value
            target_corrs[(var2, var1, group)] = corr_value  # Ensure symmetry
    return target_corrs


def calculate_target_tests(target_tests):
    """
    Calculate target statistical tests from the given specifications.

    Args:
        target_tests: List of dictionaries specifying target statistical tests.

    Returns:
        List of dictionaries of target statistical tests.
    """
    target_test_results = []
    for test in target_tests:
        test_result = {
            "test_type": test.get("test_type"),
            "variables": test.get("variables"),
        }

        # Only add values if they exist in the input test
        for key in [
            "effect_size",
            "p_value",
            "test_statistic",
            "group_means",
            "groups",
        ]:
            if key in test:
                test_result[key] = test[key]

        target_test_results.append(test_result)
    return target_test_results


def calculate_target_stats_and_corrs_and_tests(variables, correlations, tests):
    """
    Calculate target statistics, correlations, and statistical tests from the given specifications.

    Args:
        variables: Dictionary containing variable specifications
        correlations: List of dictionaries specifying correlation tests
        tests: List of dictionaries specifying target statistical tests

    Returns:
        Tuple of (target_stats, target_corrs, target_tests)
    """

    target_stats = calculate_target_stats(variables)
    target_corrs = calculate_target_corrs(correlations)
    target_tests = calculate_target_tests(tests)

    return target_stats, target_corrs, target_tests


def calculate_statistics(data, target_stats):
    """
    Calculate basic statistics for the synthetic data.

    Args:
        data: Dictionary of synthetic data, keys are (variable, group) tuples
        target_stats: Dictionary of target statistics with expected metrics

    Returns:
        Dictionary of current statistics matching the format of target_stats
    """
    current_stats = {}

    for (var, group), values in data.items():
        if (var, group) not in target_stats:
            continue

        # Continuous and ordinal variables (mean, std)
        if "mean" in target_stats[(var, group)]:
            current_stats[(var, group)] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        # Calculate quartile statistics if needed
        if "median" in target_stats[(var, group)]:
            q1 = float(np.percentile(values, 25))
            q3 = float(np.percentile(values, 75))
            current_stats[(var, group)] = {
                "median": float(np.median(values)),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
            }

        # Categorical variables (category counts)
        elif "categories" in target_stats[(var, group)]:
            values_arr = np.asarray(values)
            categories = target_stats[(var, group)]["categories"]
            counts = [int(np.sum(values_arr == cat)) for cat in categories]
            current_stats[(var, group)] = {
                "categories": categories,
                "counts": counts,
            }

        # Binary variables (sum of values)
        elif "count" in target_stats[(var, group)]:
            current_stats[(var, group)] = {"count": int(np.sum(values))}

    return current_stats


def calculate_correlations(data, target_corrs):
    """
    Calculate correlations between variables in the synthetic data.

    Args:
        data: Dictionary of synthetic data, keys are (variable, group) tuples
        target_corrs: Dictionary of target correlations

    Returns:
        Dictionary of current correlations
    """
    current_corrs = {}
    processed_corr_pairs = set()

    for (var1, var2, group), _ in target_corrs.items():
        # Process each correlation pair only once per group
        pair_key = tuple(sorted((var1, var2))) + (group,)
        if pair_key in processed_corr_pairs:
            continue

        if (var1, group) in data and (var2, group) in data:
            x = data[(var1, group)]
            y = data[(var2, group)]

            # Calculate correlation using robust_relation
            result = robust_relation(x, y, test_name="pearson")
            if result and result.get("test_statistic") is not np.nan:
                current_corrs[(var1, var2, group)] = result
                current_corrs[(var2, var1, group)] = (
                    result  # Store symmetrically
                )

            processed_corr_pairs.add(pair_key)

    return current_corrs


def calculate_statistical_tests(data, target_tests, groups):
    """
    Calculate statistical tests for the synthetic data.

    Args:
        data: Dictionary of synthetic data, keys are (variable, group) tuples
        target_tests: List of target statistical test specifications
        groups: List of all group names in the study

    Returns:
        List of current statistical test results
    """
    if not target_tests:
        return []

    current_tests = []

    for test_spec in target_tests:
        test_type = test_spec.get("test_type")
        variables = test_spec.get("variables", [])
        test_groups = test_spec.get("groups", [])

        # Case 1: Two-variable tests (typically within-group)
        if len(variables) == 2:
            var1, var2 = variables
            groups_to_run = test_groups if test_groups else groups

            for group in groups_to_run:
                if (var1, group) in data and (var2, group) in data:
                    x = data[(var1, group)]
                    y = data[(var2, group)]

                    if (
                        x is not None
                        and y is not None
                        and len(x) > 1
                        and len(y) > 1
                    ):
                        result = robust_relation(x, y, test_name=test_type)
                        if (
                            result
                            and result.get("test_statistic") is not np.nan
                        ):
                            current_tests.append(
                                {
                                    "test_type": test_type,
                                    "variables": [var1, var2],
                                    "groups": [group],
                                    **result,
                                }
                            )
                        else:
                            print(
                                f"Warning: robust_relation failed for {test_type} on ({var1}, {var2}) in group '{group}'."
                            )

        # Case 2: One-variable tests between two groups
        elif (
            len(variables) == 1
            and len(test_groups) == 2
            and test_type not in ["one_way_anova", "ranova"]
        ):
            var1 = variables[0]
            groupA, groupB = test_groups[0], test_groups[1]

            if (var1, groupA) in data and (var1, groupB) in data:
                x_groupA = data[(var1, groupA)]
                y_groupB = data[(var1, groupB)]

                result = robust_relation(
                    x_groupA, y_groupB, test_name=test_type
                )
                if result:
                    current_tests.append(
                        {
                            "test_type": test_type,
                            "variables": [var1],
                            "groups": [groupA, groupB],
                            **result,
                        }
                    )
                else:
                    print(
                        f"Warning: robust_relation failed for {test_type} on '{var1}' between groups '{groupA}' and '{groupB}'."
                    )

        # Case 3: One-variable ANOVA-type tests across multiple groups (>= 2 groups)
        elif (
            len(variables) == 1
            and len(test_groups) >= 2
            and test_type in ["one_way_anova", "ranova"]
        ):
            var_to_test = variables[0]

            samples_for_test = (
                []
            )  # Changed from all_data_points and group_labels
            valid_groups_in_test = []

            for group_name in test_groups:
                if (var_to_test, group_name) in data:
                    series_for_group = data[(var_to_test, group_name)]
                    if len(series_for_group) > 0:  # Ensure group has data
                        samples_for_test.append(
                            np.asarray(series_for_group)
                        )  # Append the series/array itself
                        if group_name not in valid_groups_in_test:
                            valid_groups_in_test.append(group_name)
                    else:
                        print(
                            f"Warning: No data for variable '{var_to_test}' in group '{group_name}' for {test_type}."
                        )
                else:
                    print(
                        f"Warning: Variable '{var_to_test}' or group '{group_name}' not found in data for {test_type}."
                    )

            # Ensure we have samples from at least two distinct groups for the test
            if len(samples_for_test) >= 2:
                # robust_relation for multisample tests expects a list of arrays as the first argument, and y=None
                try:
                    result = robust_relation(
                        samples_for_test,  # Pass the list of samples
                        y=None,  # y is None for multisample format
                        test_name=test_type,
                    )
                    if result:
                        # Store the original list of groups intended for the test
                        current_tests.append(
                            {
                                "test_type": test_type,
                                "variables": [var_to_test],
                                "groups": test_groups,  # Use the original test_groups from spec
                                **result,
                            }
                        )
                except Exception as e:
                    # Catch any exceptions from robust_relation
                    print(
                        f"Error during robust_relation for {test_type} on '{var_to_test}' with groups {test_groups}: {e}"
                    )

            elif (
                len(samples_for_test) > 0
            ):  # Data collected, but not enough groups
                print(
                    f"Warning: Not enough distinct groups with data for {test_type} on '{var_to_test}' across specified groups {test_groups}. Need at least 2."
                )
            # else: samples_for_test is empty or has only 1 group, warnings already printed

    return current_tests


def calculate_stats_and_corrs_and_tests(
    data, target_stats, target_corrs, groups, target_tests=None
):
    """
    Calculate statistics, correlations, and statistical tests for the given dataset.

    Args:
        data: Dictionary of synthetic data, keys are (variable, group) tuples.
        target_stats: Dictionary of target statistics.
        target_corrs: Dictionary of target correlations.
        groups: List of all group names in the study.
        target_tests: List of target statistical test dicts (optional).

    Returns:
        Tuple of (current_stats, current_corrs, current_tests)
    """
    current_stats = calculate_statistics(data, target_stats)
    current_corrs = calculate_correlations(data, target_corrs)
    current_tests = calculate_statistical_tests(data, target_tests, groups)

    return current_stats, current_corrs, current_tests
