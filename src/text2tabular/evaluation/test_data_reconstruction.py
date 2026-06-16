from text2tabular.reconstruction.statistics.core import robust_relation
from text2tabular.evaluation.test_data_utils import (
    load_test_datasets,
)
from text2tabular.reconstruction.main import (
    generate_synthetic_data,
    compare_statistics,
)
import warnings
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os
from text2tabular.evaluation.test_data_utils import (
    GROUP_MAPPING,
    ANOVA_MAPPING,
)


def generate_dataset_summary(
    dataset_name, dataset_info, IQR=False, Grouped=True
):
    """
    Generate a statistical summary of a dataset in the format required for data reconstruction.

    Args:
        dataset_name: Name of the dataset
        dataset_info: Dictionary containing dataset information
        IQR: Whether to use IQR-based statistics for continuous variables

    Returns:
        Dictionary containing statistical summary in the format required by generate_synthetic_data
    """
    print(f"Generating summary for {dataset_name} dataset...")
    df = dataset_info["data"]

    if Grouped:
        # Determine if we should use a specific grouping variable
        group_var = GROUP_MAPPING.get(dataset_name)
    else:
        group_var = None
    # Define groups based on the mapping or use "overall"
    if group_var and group_var in df.columns:
        # turn col into string
        df[group_var] = df[group_var].astype(str)
        groups = list(df[group_var].unique())
        print(f"Using {group_var} as groups: {groups}")

        # Calculate size for each group
        group_sizes = {
            group: len(df[df[group_var] == group]) for group in groups
        }
    else:
        groups = ["overall"]
        group_sizes = {"overall": len(df)}

    # Calculate study size
    study_size = len(df)

    # Initialize variables dictionary
    variables = {
        "ordinal": {},
        "continuous": {},
        "categorical": {},
        "binary": {},
    }

    # Get variable types from the structure
    var_types = dataset_info["variables"]

    # Calculate statistics for each variable type
    for var, var_type in var_types.items():
        if var not in df.columns:
            continue

        if (var_type == "continuous") or (var_type == "ordinal"):
            var_dict = (
                variables["continuous"]
                if var_type == "continuous"
                else variables["ordinal"]
            )

            # If using grouping and this isn't the grouping variable
            if group_var:
                if var != group_var:
                    var_dict[var] = {}
                    for group in groups:
                        group_data = df[df[group_var] == group][var]
                        if IQR:
                            var_dict[var][group] = {
                                "median": float(group_data.median()),
                                "q1": float(group_data.quantile(0.25)),
                                "q3": float(group_data.quantile(0.75)),
                                "min_val": float(group_data.min()),
                                "max_val": float(group_data.max()),
                            }
                        else:
                            var_dict[var][group] = {
                                "mean": float(group_data.mean()),
                                "std": float(group_data.std()),
                            }
            else:
                # Use overall statistics
                if IQR:
                    var_dict[var] = {
                        "overall": {
                            "median": float(df[var].median()),
                            "q1": float(df[var].quantile(0.25)),
                            "q3": float(df[var].quantile(0.75)),
                            "min_val": float(df[var].min()),
                            "max_val": float(df[var].max()),
                        }
                    }
                else:
                    var_dict[var] = {
                        "overall": {
                            "mean": float(df[var].mean()),
                            "std": float(df[var].std()),
                        }
                    }

        elif var_type == "categorical":
            if group_var:
                if var != group_var:
                    variables["categorical"][var] = {}
                    for group in groups:
                        group_data = df[df[group_var] == group][var]
                        value_counts = group_data.value_counts().to_dict()
                        variables["categorical"][var][group] = {
                            k: int(v) for k, v in value_counts.items()
                        }
            else:
                # Use overall statistics for categorical variables
                variables["categorical"][var] = {}
                value_counts = df[var].value_counts().to_dict()
                variables["categorical"][var]["overall"] = {
                    k: int(v) for k, v in value_counts.items()
                }

        elif var_type == "binary":
            if group_var:
                if var != group_var:
                    variables["binary"][var] = {}
                    for group in groups:
                        group_data = df[df[group_var] == group][var]
                        if group_data.dtype == "bool":
                            count_1s = int(group_data.sum())
                        else:
                            count_1s = int(group_data.value_counts().get(1, 0))
                        variables["binary"][var][group] = count_1s
            else:
                if df[var].dtype == "bool":
                    count_1s = int(df[var].sum())
                else:
                    count_1s = int(df[var].value_counts().get(1, 0))
                variables["binary"][var] = {"overall": count_1s}

    # Generate statistical tests
    statistical_tests = []
    if (
        group_var and group_var in df.columns and Grouped
    ):  # Ensure group_var is valid
        anova_cont_var = ANOVA_MAPPING.get(dataset_name)
        if (
            anova_cont_var
            and anova_cont_var in df.columns
            and var_types.get(anova_cont_var) in ["continuous", "ordinal"]
        ):

            # Prepare data for f_oneway: list of arrays, one for each group
            anova_groups_data = []
            group_means_for_anova = []
            valid_groups_for_anova_test = []

            for group_name in groups:  # 'groups' list is already defined
                group_specific_data = df[df[group_var] == group_name][
                    anova_cont_var
                ].dropna()
                if not group_specific_data.empty:
                    anova_groups_data.append(group_specific_data.values)
                    group_means_for_anova.append(
                        float(group_specific_data.mean())
                    )
                    valid_groups_for_anova_test.append(group_name)

            if (
                len(anova_groups_data) >= 2
            ):  # Need at least two groups for ANOVA
                from scipy.stats import f_oneway

                try:
                    f_stat, p_value = f_oneway(*anova_groups_data)

                    # Check for NaN f_stat or p_value which can happen if input arrays are problematic (e.g. all same values within groups)
                    if pd.notna(f_stat) and pd.notna(p_value):
                        anova_test_entry = {
                            "variables": [
                                anova_cont_var
                            ],  # ANOVA on this variable
                            "test_type": "one_way_anova",
                            "test_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "groups": valid_groups_for_anova_test,  # The groups involved in this specific ANOVA
                        }
                        statistical_tests.append(anova_test_entry)
                        print(
                            f"Added ANOVA for {anova_cont_var} across groups of {group_var}: F={f_stat:.2f}, p={p_value:.3f}"
                        )
                    else:
                        print(
                            f"Skipping ANOVA for {anova_cont_var} across {group_var} due to NaN F-statistic or p-value."
                        )
                except Exception as e:
                    print(
                        f"Error calculating ANOVA for {anova_cont_var} across {group_var}: {e}"
                    )
            else:
                print(
                    f"Skipping ANOVA for {anova_cont_var}: Not enough groups with data ({len(anova_groups_data)} found)."
                )
        elif anova_cont_var:
            print(
                f"Warning: ANOVA continuous variable '{anova_cont_var}' for dataset '{dataset_name}' not found in data columns or not numeric."
            )

    # Process each variable pair and run appropriate tests
    for x_col, y_col in dataset_info["pairs_to_test"]:
        x_type = var_types.get(x_col)
        y_type = var_types.get(y_col)

        if not x_type or not y_type:
            print(f"Warning: Missing type for {x_col} or {y_col}")
            continue

        if x_col not in df.columns or y_col not in df.columns:
            continue

        # Handle tests with grouping
        if group_var and x_col != group_var and y_col != group_var:
            # Now run tests per group
            for group in groups:
                group_data = df[df[group_var] == group][
                    [x_col, y_col]
                ].dropna()
                if len(group_data) < 30:  # Skip if too few samples
                    continue
                if x_type in ["continuous", "ordinal"] and y_type in [
                    "continuous",
                    "ordinal",
                ]:
                    try:
                        corr = float(
                            group_data[x_col].corr(
                                group_data[y_col], method="spearman"
                            )
                        )
                        statistical_tests.append(
                            {
                                "variables": [x_col, y_col],
                                "test_type": "spearman",
                                "test_statistic": float(corr),
                                "groups": [group],
                            }
                        )
                    except:
                        pass
                else:

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        test_result = robust_relation(
                            group_data[x_col], group_data[y_col]
                        )

                    if test_result is not None and "test_type" in test_result:
                        test_stat = test_result.get("test_statistic")
                        if (
                            test_stat is not None
                            and test_result.get("p_value", 0.0) < 0.1
                        ):
                            statistical_tests.append(
                                {
                                    "variables": [x_col, y_col],
                                    "test_type": test_result["test_type"],
                                    "test_statistic": float(test_stat),
                                    "p_value": float(
                                        test_result.get("p_value", 0.0)
                                    ),
                                    "groups": [group],
                                }
                            )

        else:
            # Original code for datasets without grouping or when group var is one of the test variables
            data = df[[x_col, y_col]].dropna()
            if len(data) < 50:  # Skip if too few samples
                continue
            # If both are continuous or ordinal
            if x_type in ["continuous", "ordinal"] and y_type in [
                "continuous",
                "ordinal",
            ]:
                try:
                    corr = float(
                        data[x_col].corr(data[y_col], method="spearman")
                    )
                    statistical_tests.append(
                        {
                            "variables": [x_col, y_col],
                            "test_type": "spearman",
                            "test_statistic": float(corr),
                            "groups": groups,
                        }
                    )
                except:
                    pass
            else:
                # Get test result using robust_relation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_result = robust_relation(data[x_col], data[y_col])

                if test_result is not None and "test_type" in test_result:
                    # Add to statistical tests
                    test_stat = test_result.get("test_statistic")

                    if (
                        test_stat is not None
                        and test_result.get("p_value", 0.0) < 0.1
                    ):
                        statistical_tests.append(
                            {
                                "variables": [x_col, y_col],
                                "test_type": test_result["test_type"],
                                "test_statistic": float(test_stat),
                                "p_value": float(
                                    test_result.get("p_value", 0.0)
                                ),
                                "groups": groups,
                            }
                        )

    # Combine everything into the final summary
    summary = {
        "study_size": study_size,
        "groups": groups,
        "group_sizes": group_sizes,
        "variables": variables,
        "statistical_tests": statistical_tests,
    }

    return summary


def test_data_reconstruction(
    dataset_name,
    dataset_info,
    max_iterations=250,
    n_samples=15,
    IQR=False,
    ignore_copula=False,
):
    """
    Test the full data reconstruction pipeline using a real dataset.

    Args:
        dataset_name: Name of the dataset
        dataset_info: Dictionary containing dataset information

    Returns:
        Tuple of (original_df, reconstructed_df, summary)
    """
    print(f"\n\n{'='*80}")
    print(f"TESTING DATA RECONSTRUCTION FOR {dataset_name.upper()}")
    print(f"{'='*80}")

    # Step 1: Generate statistical summary
    summary = generate_dataset_summary(dataset_name, dataset_info, IQR)

    # Print summary for debugging
    print("\nGenerated statistical summary:")
    print(f"- Study size: {summary['study_size']}")
    print(f"- Groups: {summary['groups']}")
    print(
        f"- Variables: {len(summary['variables']['continuous']) + len(summary['variables']['ordinal']) + len(summary['variables']['categorical']) + len(summary['variables']['binary'])}"
    )
    print(f"- Statistical tests: {len(summary['statistical_tests'])}")

    # Step 2: Run data reconstruction
    print("\nRunning data reconstruction...")
    parsed_data, reconstructed_df = generate_synthetic_data(
        summary,
        n_samples=n_samples,  # Generate 3 candidates for faster testing
        seed=42,
        max_iterations=max_iterations,  # Limit iterations for faster testing,
        ignore_copula=ignore_copula,
    )

    if reconstructed_df is None:
        print("Data reconstruction failed!")
        return dataset_info["data"], None, summary

    # Step 3: Compare original and reconstructed data
    print("\nComparing original vs reconstructed data:")
    print(f"Original data shape: {dataset_info['data'].shape}")
    print(f"Reconstructed data shape: {reconstructed_df.shape}")
    # use group_mapping to replace "groups" col with actual group name if not all elements are "overall"
    if reconstructed_df["group"].nunique() > 1:

        group_var = GROUP_MAPPING.get(dataset_name)
        if group_var:
            reconstructed_df[group_var] = reconstructed_df["group"]
            reconstructed_df.drop(columns=["group"], inplace=True)

    # Compare statistics
    print("\nDetailed comparison:")
    compare_statistics(parsed_data, reconstructed_df, group_var)

    return dataset_info["data"], reconstructed_df, summary


def test_anova_expansion(
    dataset_name="diamonds",
    seed=42,
    IQR=False,
    Grouped=False,
    n_samples=5,
    max_iterations=500,
):
    """
    Test ANOVA expansion functionality by manually adding an ANOVA test
    to the dataset summary.

    Args:
        dataset_name: Name of the dataset to use (default: diamonds)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (original_df, reconstructed_df, summary)
    """
    print(f"\n\n{'='*80}")
    print(f"TESTING ANOVA EXPANSION WITH {dataset_name.upper()}")
    print(f"{'='*80}")

    # Load datasets
    datasets = load_test_datasets()
    dataset_info = datasets[dataset_name]

    # Step 1: Generate statistical summary
    summary = generate_dataset_summary(
        dataset_name, dataset_info, IQR, Grouped
    )

    # Determine which categorical and continuous variables to use for ANOVA
    if dataset_name == "diamonds":
        cat_var = "cut"  # categorical variable (cut quality)
        cont_var = "price"  # continuous outcome
        # Get categorical variable unique values for group means
        df = dataset_info["data"]
        categories = df[cat_var].unique()
        group_means = [
            float(df[df[cat_var] == cat][cont_var].mean())
            for cat in categories
        ]

        # Calculate F statistic for ANOVA (or use a reasonable value)
        from scipy.stats import f_oneway

        groups = [
            df[df[cat_var] == cat][cont_var].values for cat in categories
        ]
        f_stat, p_value = f_oneway(*groups)
    elif dataset_name == "tips":
        cat_var = "day"
        cont_var = "tip"
        df = dataset_info["data"]
        categories = df[cat_var].unique()
        group_means = [
            float(df[df[cat_var] == cat][cont_var].mean())
            for cat in categories
        ]

        # Calculate F statistic
        groups = [
            df[df[cat_var] == cat][cont_var].values for cat in categories
        ]
        f_stat, p_value = f_oneway(*groups)
    else:
        raise ValueError(
            f"Dataset {dataset_name} not configured for ANOVA test"
        )

    # Step 2: Modify summary to include an explicit ANOVA test
    anova_test = {
        "variables": [cat_var, cont_var],
        "test_type": "one_way_anova",
        "test_statistic": float(
            f_stat
        ),  # Must be float for JSON serialization
        "p_value": float(p_value),
        "group_means": group_means,
        "groups": ["overall"],
    }

    # Add ANOVA test to beginning of list to ensure it's detected first
    summary["statistical_tests"] = [anova_test]

    # Print summary for debugging
    print("\nModified statistical summary with explicit ANOVA test:")
    print(f"- ANOVA test: {cat_var} vs {cont_var}")
    print(f"- F statistic: {f_stat:.4f}")
    print(f"- p value: {p_value:.4f}")
    print(f"- Group means: {[f'{m:.2f}' for m in group_means]}")

    # Step 3: Run data reconstruction
    print("\nRunning data reconstruction with ANOVA test...")
    parsed_data, reconstructed_df = generate_synthetic_data(
        summary,
        n_samples=n_samples,  # Generate 3 candidates for faster testing
        seed=seed,
        max_iterations=max_iterations,  # Limit iterations for faster testing
    )

    if reconstructed_df is None:
        print("Data reconstruction failed!")
        return dataset_info["data"], None, summary

    # Step 4: Compare original and reconstructed data
    print("\nComparing original vs reconstructed data:")
    print(f"Original data shape: {dataset_info['data'].shape}")
    print(f"Reconstructed data shape: {reconstructed_df.shape}")

    # Compare statistics
    print("\nDetailed comparison:")
    compare_statistics(parsed_data, reconstructed_df)

    if dataset_name in GROUP_MAPPING:
        group_var = GROUP_MAPPING[dataset_name]
        if "group" in reconstructed_df.columns and Grouped:
            reconstructed_df[group_var] = reconstructed_df["group"]
            reconstructed_df.drop(columns=["group"], inplace=True)

    return dataset_info["data"], reconstructed_df, summary, cat_var, cont_var
