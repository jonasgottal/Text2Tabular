import numpy as np
import pandas as pd

from text2tabular.reconstruction.correlation.matrix_builder import (
    create_correlation_matrix,
)
from text2tabular.reconstruction.utils.utils import (
    parse_json,
    combine_synthetic_data,
)
from text2tabular.reconstruction.distributions.marginals import (
    generate_distributions,
)
from text2tabular.reconstruction.copula.gaussian_copula import (
    apply_gaussian_copula,
)
from text2tabular.reconstruction.copula.group_expansion import (
    expand_for_anova,
    _extract_anova_pair,
)
from text2tabular.reconstruction.copula.group_reconstruction import (
    reconstruct_categorical_from_dummies,
)
from text2tabular.reconstruction.mcmc.mcmc import (
    mcmc_refinement,
    calculate_stats_and_corrs_and_tests,
    calculate_difference,
)

from text2tabular.reconstruction.mcmc.mcmc_utils import (
    calculate_target_stats_and_corrs_and_tests,
    calculate_target_stats,
)
from text2tabular.reconstruction.statistics.core import robust_relation


class SyntheticDataGenerator:
    """
    Generates synthetic tabular data based on statistical summaries.
    """

    def __init__(self, parsed_data, seed):
        """
        Initializes the generator with parsed input data and a random seed.

        Args:
            parsed_data (dict): The *already parsed* input data from parse_json.
            seed (int): Random seed for reproducibility.
        """
        self.parsed_data = parsed_data
        self.seed = seed
        np.random.seed(self.seed)
        # Initialize attributes for intermediate data
        self.ordinal_data = None
        self.continuous_data = None
        self.categorical_data = None
        self.binary_data = None
        self.large_corr_matrix = None
        self.combined_vars = None
        self.combined_indices = None
        self.dummy_vars = []
        self.removed_categorical_info = {}
        self.anova_pair = None
        self.synthetic_data_dict = None

    def generate_one_instance(self):
        """
        Generates a single instance of the synthetic dataset (dictionary format).

        Returns:
            dict: The generated synthetic data dictionary, or None if generation failed.
        """
        try:
            # Step 1: Generate base distributions
            variables = self.parsed_data.get("variables", {})
            group_sizes = self.parsed_data.get("group_sizes", {})
            (
                self.categorical_data,
                self.binary_data,
                self.continuous_data,
                self.ordinal_data,
            ) = generate_distributions(variables, group_sizes)

            # Step 2: Create correlation matrix
            (
                self.large_corr_matrix,
                self.combined_vars,
                self.combined_indices,
            ) = create_correlation_matrix(self.parsed_data)
            if self.large_corr_matrix is None:
                print("Warning: Correlation matrix could not be created.")
                return None

            # Step 3: Expand for ANOVA if needed
            (
                self.large_corr_matrix,
                self.combined_vars,
                self.combined_indices,
                self.categorical_data,
                self.binary_data,
                self.dummy_vars,
                self.removed_categorical_info,
                self.anova_pair,
            ) = expand_for_anova(
                self.parsed_data,
                self.large_corr_matrix,
                self.combined_vars,
                self.combined_indices,
                self.categorical_data,
                self.binary_data,
            )

            # Step 4: Apply Gaussian copula
            self.synthetic_data_dict = apply_gaussian_copula(
                self.ordinal_data,
                self.continuous_data,
                self.categorical_data,
                self.binary_data,
                self.large_corr_matrix,
                self.combined_vars,
                self.parsed_data.get("group_sizes", {}),
            )

            if not self.synthetic_data_dict:
                print("Error: Copula application failed to produce data.")
                return None

            # Step 5: Reconstruct categorical variables from dummies
            self.synthetic_data_dict = reconstruct_categorical_from_dummies(
                self.synthetic_data_dict,
                self.parsed_data,
                self.anova_pair,
                self.dummy_vars,
                self.removed_categorical_info,
            )

            # Return the dictionary before MCMC refinement
            return self.synthetic_data_dict
        except Exception as e:
            print(f"Error during single instance generation: {e}")
            return None


# --- Modified generate_synthetic_data function ---
def generate_synthetic_data(
    json_input, n_samples=10, seed=42, max_iterations=500, ignore_copula=False
):
    """
    Generates synthetic data by creating multiple instances and selecting the best one
    before applying MCMC refinement.

    Args:
        json_input (dict): Raw JSON input data.
        n_samples (int): Number of synthetic datasets to generate before selecting the best.
        seed (int): Base random seed.

    Returns:
        tuple: (parsed_data, final_synthetic_df)
               parsed_data (dict): The parsed input data.
               final_synthetic_df (pd.DataFrame or None): The refined synthetic DataFrame.
    """
    print(
        f"Starting synthetic data generation (n_samples={n_samples}, seed={seed})..."
    )
    # Step 1: Parse JSON Input (once)
    try:
        parsed_data = parse_json(json_input)
        print("JSON parsed successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not parse input JSON. {e}")
        return None, None

    # Calculate target statistics (once)

    try:
        target_stats, target_corrs, target_tests = (
            calculate_target_stats_and_corrs_and_tests(
                parsed_data.get("variables", {}),
                parsed_data.get("correlations", []),
                parsed_data.get("statistical_tests", []),
            )
        )

    except Exception as e:
        print(f"Error calculating target statistics: {e}")
        # Decide if this is fatal or if generation can proceed without targets
        return (
            parsed_data,
            None,
        )  # Assuming it's needed for comparison/MCMC

    best_synthetic_data = None
    best_diff = float("inf")

    # Generate multiple samples and select the best initial dataset
    print(f"Generating {n_samples} initial candidate datasets...")
    for i in range(n_samples):
        instance_seed = seed + i  # Use a different seed for each instance
        print(
            f"  Generating candidate {i+1}/{n_samples} (seed={instance_seed})..."
        )
        if ignore_copula:
            parsed_data_copy = parsed_data.copy()
            parsed_data_copy["statistical_tests"] = []
            parsed_data_copy["correlations"] = []

            generator = SyntheticDataGenerator(
                parsed_data_copy, seed=instance_seed
            )
        else:
            generator = SyntheticDataGenerator(parsed_data, seed=instance_seed)

        synthetic_data_dict = generator.generate_one_instance()

        if synthetic_data_dict is None or not synthetic_data_dict:
            print(f"  Candidate {i+1} generation failed.")
            continue

        # Calculate statistics and difference for this candidate
        try:
            current_stats, current_corrs, current_tests = (
                calculate_stats_and_corrs_and_tests(
                    synthetic_data_dict,
                    target_stats,  # Use calculated target stats dict
                    target_corrs,  # Use calculated target corrs dict
                    parsed_data.get("groups", []),
                    target_tests,  # Use calculated target tests list
                )
            )
            # Pass correct arguments to calculate_difference
            total_diff = calculate_difference(
                current_stats,
                current_corrs,
                target_stats,
                target_corrs,
                target_tests,
                current_tests,
            )
            print(f"  Candidate {i+1} initial difference: {total_diff:.4f}")

            # Update the best synthetic data if this one is better
            if total_diff < best_diff:
                best_diff = total_diff
                best_synthetic_data = synthetic_data_dict
                print(f"  New best candidate found (diff: {best_diff:.4f}).")

        except Exception as e:
            # Print traceback for detailed debugging
            import traceback

            print(f"  Error calculating stats/diff for candidate {i+1}: {e}")
            traceback.print_exc()  # Add traceback
            continue  # Skip this candidate

    if best_synthetic_data is None:
        print("Error: Failed to generate any valid initial synthetic dataset.")
        return parsed_data, None

    print(f"\nSelected best initial dataset with difference: {best_diff:.4f}")

    # Step 6: Refine the best synthetic data with MCMC
    print("Starting MCMC refinement...")
    try:
        # Ensure necessary components for MCMC are present in parsed_data
        variables = parsed_data.get("variables", {})
        groups = parsed_data.get("groups", [])
        # Pass the raw lists again, as mcmc_refinement recalculates targets internally
        correlations_raw = parsed_data.get("correlations", [])
        statistical_tests_raw = parsed_data.get("statistical_tests", [])

        if not variables or not groups:
            print(
                "Warning: Missing variables or groups for MCMC refinement. Skipping refinement."
            )
            refined_data = best_synthetic_data
        else:
            refined_data = mcmc_refinement(
                best_synthetic_data,
                variables,
                groups,
                correlations_raw,  # Pass raw correlations list
                statistical_tests_raw,  # Pass raw tests list
                max_iterations=max_iterations,  
                tolerance=0.01,  
                seed=seed,  # Use base seed for final refinement reproducibility
            )
            print("MCMC refinement complete.")

    except Exception as e:
        print(f"Error during MCMC refinement: {e}")
        print("Using data before refinement.")
        refined_data = best_synthetic_data  # Fallback

    # Combine the final (potentially refined) data into a DataFrame
    print("Combining final data into DataFrame...")
    try:
        synthetic_df = combine_synthetic_data(
            refined_data, parsed_data.get("groups", [])
        )
        print("Synthetic data generation process complete.")
        return parsed_data, synthetic_df
    except Exception as e:
        print(f"Error combining data into DataFrame: {e}")
        return parsed_data, None


def compare_statistics(parsed_data, synthetic_df, group_var="group"):
    """
    Compare the statistics of the input data with the generated synthetic data.

    Args:
        parsed_data: Dictionary containing the parsed original data specifications.
        synthetic_df: pandas DataFrame containing the synthetic data.

    Returns:
        None (prints comparison results)
    """
    if synthetic_df is None:
        print("Cannot compare statistics: Synthetic DataFrame is None.")
        return
    if not isinstance(parsed_data, dict):
        print("Cannot compare statistics: Invalid parsed_data format.")
        return

    print("\nComparing Input vs. Generated Statistics:")
    print("========================================")

    variables_spec = parsed_data.get("variables", {})
    groups = parsed_data.get("groups", [])
    target_stats = calculate_target_stats(variables_spec)

    # Compare different types of variables
    compare_numeric_variables(
        variables_spec, target_stats, synthetic_df, groups, group_var
    )
    compare_categorical_variables(
        variables_spec, target_stats, synthetic_df, groups, group_var
    )
    compare_binary_variables(
        variables_spec,
        target_stats,
        synthetic_df,
        groups,
        parsed_data,
        group_var,
    )
    compare_statistical_tests(parsed_data, synthetic_df, groups, group_var)


def format_stat(value, precision=3):
    """Format a statistical value with specified precision."""
    if (
        value is None
        or value == "N/A"
        or (isinstance(value, (float, np.number)) and np.isnan(value))
    ):
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def compare_numeric_variables(
    variables_spec, target_stats, synthetic_df, groups, group_var
):
    """Compare means and standard deviations of numeric (ordinal & continuous) variables."""
    print("\nNumeric Variables (Mean/Std):")
    print("-----------------------------")

    for var_type in ["ordinal", "continuous"]:
        if var_type not in variables_spec:
            continue

        for var, _ in variables_spec[var_type].items():
            if var not in synthetic_df.columns:
                print(
                    f"Warning: {var_type.capitalize()} variable '{var}' not found in synthetic DataFrame."
                )
                continue

            print(f"\nVariable: {var} ({var_type})")
            for group in groups:
                target_key = (var, group)
                if target_key not in target_stats:
                    continue

                group_data = synthetic_df.loc[
                    synthetic_df[group_var] == group, var
                ].dropna()
                target_mean = target_stats[target_key].get("mean")
                target_median = target_stats[target_key].get("median")
                target_std = target_stats[target_key].get("std")
                target_iqr = target_stats[target_key].get("iqr")
                target_q1 = target_stats[target_key].get("q1")
                target_q3 = target_stats[target_key].get("q3")

                print(f"  group: {group}")
                if target_mean is not None:
                    print(
                        f"    Input:     Mean = {format_stat(target_mean)}, Std = {format_stat(target_std)}"
                    )
                    if not group_data.empty and len(group_data) > 1:
                        gen_mean = group_data.mean()
                        gen_std = group_data.std()
                        print(
                            f"    Generated: Mean = {format_stat(gen_mean)}, Std = {format_stat(gen_std)}"
                        )
                elif target_median is not None and target_iqr is not None:
                    print(
                        f"    Input:     Median = {format_stat(target_median)}, IQR = {format_stat(target_iqr)}"
                    )
                    if not group_data.empty and len(group_data) > 1:
                        gen_median = group_data.median()
                        gen_q3 = group_data.quantile(0.75)
                        gen_q1 = group_data.quantile(0.25)
                        gen_iqr = gen_q3 - gen_q1

                        print(
                            f"    Generated: Median = {format_stat(gen_median)}, IQR:  Q1 -Q3 = {format_stat(gen_q1)}-{format_stat(gen_q3)} --> {format_stat(gen_iqr)}"
                        )

                elif (
                    target_median is not None
                    and target_q1 is not None
                    and target_q3 is not None
                ):
                    print(
                        f"    Input:      Median = {format_stat(target_median)},  Q1 = {format_stat(target_q1)}, Q3 = {format_stat(target_q3)}"
                    )

                    if not group_data.empty and len(group_data) > 1:
                        gen_median = group_data.median()
                        gen_q1 = group_data.quantile(0.25)
                        gen_q3 = group_data.quantile(0.75)
                        print(
                            f"    Generated: Median = {format_stat(gen_median)}, Q1 = {format_stat(gen_q1)}, Q3 = {format_stat(gen_q3)}"
                        )

                else:
                    print("    Generated: No data for this group.")


def compare_categorical_variables(
    variables_spec, target_stats, synthetic_df, groups, group_var
):
    """Compare counts of categorical variables."""
    print("\nCategorical Variables (Counts):")
    print("-------------------------------")

    if "categorical" not in variables_spec:
        return

    for var, group_stats_dict in variables_spec["categorical"].items():
        if var not in synthetic_df.columns:
            print(
                f"Warning: Categorical variable '{var}' not found in synthetic DataFrame."
            )
            continue

        print(f"\nVariable: {var} (categorical)")
        for group in groups:
            target_key = (var, group)
            if target_key not in target_stats:
                continue

            group_data = synthetic_df.loc[
                synthetic_df[group_var] == group, var
            ].dropna()
            target_categories = target_stats[target_key].get("categories", [])
            target_counts = target_stats[target_key].get("counts", [])

            print(f"  group: {group}")
            if not target_categories:
                print("    Input: No categories defined.")
                continue

            for category, target_count in zip(
                target_categories, target_counts
            ):
                generated_count = (
                    sum(group_data == category) if not group_data.empty else 0
                )
                print(f"    Category: {category}")
                print(f"      Input:     Count = {target_count}")
                print(f"      Generated: Count = {generated_count}")


def compare_binary_variables(
    variables_spec, target_stats, synthetic_df, groups, parsed_data, group_var
):
    """Compare counts of 1s in binary variables."""
    print("\nBinary Variables (Count of 1s):")
    print("---------------------------------")

    if "binary" not in variables_spec:
        return

    for var, group_stats_dict in variables_spec["binary"].items():
        # Check if it was an ANOVA dummy that should have been removed
        is_dummy_anova = False
        if parsed_data.get("has_anova", False):
            anova_p = _extract_anova_pair(parsed_data)
            if anova_p and var.startswith(anova_p[0] + "_"):  # Check prefix
                is_dummy_anova = True

        if var not in synthetic_df.columns:
            if not is_dummy_anova:  # Only warn if it wasn't an expected dummy
                print(
                    f"Warning: Binary variable '{var}' not found in synthetic DataFrame."
                )
            continue
        elif is_dummy_anova:
            print(
                f"Warning: ANOVA dummy variable '{var}' unexpectedly found in final DataFrame."
            )

        print(f"\nVariable: {var} (binary)")
        for group in groups:
            target_key = (var, group)
            if target_key not in target_stats:
                continue

            group_data = synthetic_df.loc[
                synthetic_df[group_var] == group, var
            ].dropna()
            target_count = target_stats[target_key].get(
                "count"
            )  # Should be count of 1s

            print(f"  group: {group}")
            print(f"    Input:     Count (1s) = {target_count}")

            if not group_data.empty:
                # Ensure data is numeric-like (0 or 1) before summing
                generated_count = (
                    pd.to_numeric(group_data, errors="coerce").eq(1).sum()
                )
                print(f"    Generated: Count (1s) = {generated_count}")
            else:
                print("    Generated: No data for this group.")


def compare_statistical_tests(parsed_data, synthetic_df, groups, group_var):
    """Compare statistical test results."""
    print("\nStatistical Tests:")
    print("------------------")

    tests_spec = parsed_data.get("statistical_tests", [])
    if not isinstance(tests_spec, list):
        print(
            "Warning: 'statistical_tests' format is not a list. Skipping comparison."
        )
        return

    for test_info in tests_spec:
        if not isinstance(test_info, dict):
            continue

        test_type = test_info.get("test_type")
        variables = test_info.get("variables", [])
        test_groups = test_info.get(
            "groups", groups
        )  # groups specific to the test

        if not isinstance(test_groups, list):
            test_groups = groups

        # Determine target statistic key
        target_stat_key = None
        if "test_statistic" in test_info:
            target_stat_key = "test_statistic"

        if not test_type or not variables or target_stat_key is None:
            print(
                f"Warning: Skipping test spec due to missing info: {test_info}"
            )
            continue

        # Ensure target_stat_key is lowercase for consistent access later
        target_stat_key_lower = target_stat_key.lower()
        target_statistic = test_info[
            target_stat_key
        ]  # Get value using original key

        print(f"\nTest: {test_type} on {variables}")
        print(
            f"  Input Target ({target_stat_key}): {format_stat(target_statistic)}"
        )

        try:
            compare_test_by_structure(
                test_type,
                variables,
                test_groups,
                target_stat_key_lower,
                test_info,
                synthetic_df,
                groups,
                group_var,
            )
        except Exception as e:
            import traceback

            print(f"  Error running test {test_type} on {variables}: {e}")
            traceback.print_exc()


def compare_test_by_structure(
    test_type,
    variables,
    test_groups,
    target_stat_key_lower,
    test_info,
    synthetic_df,
    groups,
    group_var,
):
    """Compare statistical tests based on their structure (number of variables and groups)."""
    result = None

    # Case 1: Tests between two variables (correlation, chi-square, ANOVA/MWU)
    if len(variables) == 2:
        var1, var2 = variables
        if var1 == group_var or var2 == group_var:

            data1 = synthetic_df[var1]
            data2 = synthetic_df[var2]
            # robust_relation should handle different test types based on data types and test_name
            result = robust_relation(data1, data2, test_name=test_type)
            process_test_result(
                result,
                groups,
                target_stat_key_lower,
                test_type,
                test_info,
            )

        else:

            groups_to_run_on = test_groups if test_groups else groups
            # print(f"  Running for groups: {', '.join(groups_to_run_on)}")
            for group in groups_to_run_on:
                if group not in groups:
                    continue

                group_df = synthetic_df[synthetic_df[group_var] == group]
                if var1 in group_df.columns and var2 in group_df.columns:
                    data1 = group_df[var1]
                    data2 = group_df[var2]
                    # robust_relation should handle different test types based on data types and test_name
                    result = robust_relation(data1, data2, test_name=test_type)
                    process_test_result(
                        result,
                        group,
                        target_stat_key_lower,
                        test_type,
                        test_info,
                    )
                else:
                    missing_vars = [
                        v for v in [var1, var2] if v not in group_df.columns
                    ]
                    print(
                        f"    group '{group}': Skipped (missing variables: {missing_vars})"
                    )

    # Case 2: Tests on one variable between two specific groups (t-test, MWU)
    elif (
        len(variables) == 1
        and len(test_groups) >= 2
        and test_type not in ["one_way_anova", "kruskal_wallis"]
    ):
        var1 = variables[0]
        groupA, groupB = test_groups[0], test_groups[1]
        print(f"  Comparing groups: {groupA} vs {groupB}")

        if (
            groupA in groups
            and groupB in groups
            and var1 in synthetic_df.columns
        ):
            dataA = synthetic_df.loc[synthetic_df[group_var] == groupA, var1]
            dataB = synthetic_df.loc[synthetic_df[group_var] == groupB, var1]
            # Pass data as x, y to robust_relation for unpaired tests
            result = robust_relation(dataA, dataB, test_name=test_type)
            process_test_result(
                result, None, target_stat_key_lower, test_type, test_info
            )
        else:
            print(f"    Skipped (invalid groups or variable '{var1}' missing)")

    # Case 3: Tests for ANOVA/KW with 1 variable across multiple groups
    elif (
        test_type in ["one_way_anova", "kruskal_wallis"]
        and len(variables) == 1
    ):
        var1 = variables[0]
        groups_to_run_on = test_groups if test_groups else groups
        print(
            f"  Comparing variable '{var1}' across groups: {', '.join(groups_to_run_on)}"
        )

        if var1 in synthetic_df.columns:
            samples = []
            valid_groups_found = True
            group_names_for_means = (
                []
            )  # Store group names in order for mean comparison

            for group in groups_to_run_on:
                if group not in groups:
                    print(
                        f"    Warning: group '{group}' not in study groups. Skipping."
                    )
                    valid_groups_found = False
                    break

                sample_data = synthetic_df.loc[
                    synthetic_df[group_var] == group, var1
                ]
                if sample_data.empty:
                    print(
                        f"    Warning: No data for group '{group}'. Skipping test."
                    )
                    valid_groups_found = False
                    break

                samples.append(sample_data)
                group_names_for_means.append(group)  # Add group name

            if valid_groups_found and len(samples) >= 2:
                # Pass list of samples as 'x' to robust_relation
                result = robust_relation(samples, y=None, test_name=test_type)
                process_test_result(
                    result, None, target_stat_key_lower, test_type, test_info
                )
            elif valid_groups_found:
                print(f"    Skipped (need at least 2 groups with data)")
        else:
            print(f"    Skipped (variable '{var1}' missing)")

    # Case 4: Unhandled test structure
    else:
        print(
            f"  Skipped (unhandled test structure: {len(variables)} vars, {len(test_groups)} groups for test type {test_type})"
        )


def process_test_result(
    result, group, target_stat_key_lower, test_type, test_info
):
    """Process and display the results of a statistical test."""
    # Check if the expected statistic key exists in the result (case-insensitive)
    gen_stat = None
    if result:
        result_keys_lower = {k.lower(): v for k, v in result.items()}
        if target_stat_key_lower in result_keys_lower:
            gen_stat = result_keys_lower[target_stat_key_lower]

    group_prefix = f"group '{group}' " if group else ""

    if gen_stat is not None:
        print(
            f"    {group_prefix}Generated ({target_stat_key_lower}): {format_stat(gen_stat)}"
        )
        # Check for group means if it was an ANOVA
        if (
            test_type == "one_way_anova"
            and "group_means" in test_info
            and result
            and "group_means" in result
        ):
            target_means = test_info["group_means"]
            gen_means = result["group_means"]
            print(
                f"      Input Means: {[format_stat(m) for m in target_means]}"
            )
            print(
                f"      Generated Means: {[format_stat(m) for m in gen_means]}"
            )
    elif result:
        print(
            f"    {group_prefix}Generated: N/A (robust_relation returned no '{target_stat_key_lower}')"
        )
    else:
        print(f"    {group_prefix}Generated: N/A (robust_relation failed)")


# Example Usage (within main.py)
if __name__ == "__main__":
    # Load the example JSON data (same structure as in the notebook 'data' variable)
    json_input_data = {
        "study_size": 400,
        "groups": ["Drug intervention group", "Control group"],
        "group_sizes": {"Drug intervention group": 200, "Control group": 200},
        "variables": {
            "ordinal": {
                "Age": {
                    "Drug intervention group": {"mean": 50.4, "std": 8.7},
                    "Control group": {"mean": 52.5, "std": 9.4},
                }
            },
            "continuous": {
                "BMI": {
                    "Drug intervention group": {"mean": 26.5, "std": 4.2},
                    "Control group": {"mean": 25.9, "std": 3.8},
                }
            },
            "categorical": {
                "Sex": {
                    "Drug intervention group": {"Male": 89, "Female": 111},
                    "Control group": {"Male": 118, "Female": 82},
                }
            },
            "binary": {
                "Stage I": {
                    "Drug intervention group": 70,  # Count of 1s
                    "Control group": 68,  # Count of 1s
                }
            },
        },
        # Using 'statistical_tests' to infer correlations/comparisons AND for MCMC target
        "statistical_tests": [
            {
                "variables": ["Age", "BMI"],
                "test_type": "pearson",
                "test_statistic": 0.22,
                "p_value": 0.12,
                "groups": ["Drug intervention group", "Control group"],
            },
            {
                "variables": ["Age"],
                "test_type": "unpaired_t_test",
                "test_statistic": -2.4,
                "p_value": 0.017,
                "groups": ["Drug intervention group", "Control group"],
            },
            {
                "variables": ["Age", "Stage I"],
                "test_type": "spearman",
                "test_statistic": 0.1,
                "p_value": 0.12,
                "groups": ["Drug intervention group", "Control group"],
            },
            {
                "variables": ["Sex", "Stage I"],
                "test_type": "chi_square",
                "test_statistic": 2.43,
                "p_value": 0.12,
                "groups": ["Drug intervention group", "Control group"],
            },
            {
                "variables": ["Sex", "Age"],
                "test_type": "wilcoxon_mann_whitney",
                "test_statistic": 15820,
                "p_value": 0.001,
                "groups": ["Drug intervention group", "Control group"],
            },
            {
                "variables": ["group", "Age"],
                "test_type": "unpaired_t_test",
                "test_statistic": -2.4,
                "p_value": 0.017,
                "groups": ["Drug intervention group", "Control group"],
            },
            {
                "variables": ["Sex", "BMI"],
                "test_type": "one_way_anova",
                "test_statistic": 5.76,
                "p_value": 0.017,
                "group_means": [25.4, 26.5],
                "groups": ["Drug intervention group", "Control group"],
            },
        ],
        # Explicitly define correlations for comparison AND MCMC target
        "correlations": [
            {
                "variables": ["Age", "BMI"],
                "correlation_coefficient": 0.22,
                "groups": ["Drug intervention group", "Control group"],
            },
            {
                "variables": ["Age", "Stage I"],
                "correlation_coefficient": 0.1,
                "groups": ["Drug intervention group", "Control group"],
            },
            # Add other known correlations if available
        ],
    }

    print("--- Running Main Synthetic Data Generation ---")
    # Call the main generation function which now uses the class internally
    parsed_data_result, final_synthetic_df = generate_synthetic_data(
        json_input_data,
        n_samples=5,
        seed=42,  # Generate 5 candidates before MCMC
    )

    if final_synthetic_df is not None:
        print("\n--- Generated Synthetic Data (First 5 Rows) ---")
        print(final_synthetic_df.head())

        # Call compare_statistics after successful generation
        print("\n--- Comparing Final Statistics ---")
        compare_statistics(parsed_data_result, final_synthetic_df)
    else:
        print("\n--- Synthetic Data Generation Failed ---")
