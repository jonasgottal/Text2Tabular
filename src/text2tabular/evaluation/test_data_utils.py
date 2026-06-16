import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
import random
from typing import Dict
from text2tabular.reconstruction.statistics.core import robust_relation
from text2tabular.reconstruction.correlation.matrix_builder import (
    convert_test_to_correlation,
)
from scipy.stats import pointbiserialr, chi2_contingency


GROUP_MAPPING: Dict[str, str] = {
    "iris": "species",
    "titanic": "survived",
    "diamonds": "expensive",
    "tips": "time",
}
ANOVA_MAPPING = {
    "iris": "sepal_length",
    "titanic": "age",
    "diamonds": "price",
    "tips": "tip",
}


def load_test_datasets():
    """
    Load various seaborn datasets and prepare them for correlation testing.
    Returns a dictionary of datasets with variable types labeled.
    """

    # Helper function to lowercase string columns
    def lowercase_string_columns(df):
        for col in df.select_dtypes(include=["object", "category"]):
            if (
                df[col].dtype == "object"
            ):  # For object columns, apply to string instances
                try:
                    df[col] = df[col].apply(
                        lambda x: x.lower() if isinstance(x, str) else x
                    )
                except (
                    TypeError
                ):  # Handle potential errors if a column is mixed and not all are strings
                    pass  # Or add more specific error handling/logging
            elif pd.api.types.is_categorical_dtype(
                df[col]
            ):  # For categorical columns
                # Check if categories are strings before trying to lowercase
                if all(isinstance(cat, str) for cat in df[col].cat.categories):
                    df[col] = df[col].cat.rename_categories(
                        [cat.lower() for cat in df[col].cat.categories]
                    )
        return df

    datasets = {}

    # 1. Iris dataset (continuous variables)
    iris = sns.load_dataset("iris")
    iris = lowercase_string_columns(iris)  # Lowercase string columns

    datasets["iris"] = {
        "data": iris,
        "variables": {
            "sepal_length": "continuous",
            "sepal_width": "continuous",
            "petal_length": "continuous",
            "petal_width": "continuous",
            "species": "categorical",
        },
        "pairs_to_test": [
            ("sepal_length", "sepal_width"),
            ("petal_length", "petal_width"),
            ("petal_length", "sepal_length"),
            ("petal_width", "sepal_width"),
            ("petal_width", "groups"),
        ],
        "description": "Flower measurements with continuous variables",
    }

    # 2. Titanic dataset (mix of categorical, binary, and continuous)
    titanic = sns.load_dataset("titanic")
    # only keep relevant columns: age, fare, survived, pclass, sex
    titanic = titanic[["age", "fare", "survived", "sex"]]
    titanic = lowercase_string_columns(titanic)
    # Create binary survival variable
    datasets["titanic"] = {
        "data": titanic,
        "variables": {
            "age": "continuous",
            "fare": "continuous",
            "survived": "binary",
            # "pclass": "ordinal",
            "sex": "categorical",
        },
        "pairs_to_test": [
            ("age", "fare"),
            ("survived", "age"),
            # ("survived", "pclass"),
            ("sex", "survived"),
        ],
        "description": "Titanic passenger data with mixed variable types",
    }

    # 3. Tips dataset (continuous, ordinal, and categorical)
    tips = sns.load_dataset("tips")
    # Create binary smoker variable from string
    tips["smoker"] = tips["smoker"].map({"Yes": 1, "No": 0})
    tips = lowercase_string_columns(tips)
    # Discretize total_bill
    datasets["tips"] = {
        "data": tips,
        "variables": {
            "total_bill": "continuous",
            "sex": "categorical",
            "time": "categorical",
            "tip": "continuous",
            "size": "ordinal",
            "day": "categorical",
            "smoker": "binary",
        },
        "pairs_to_test": [
            ("total_bill", "tip"),
            ("tip", "size"),
            ("total_bill", "size"),
            ("day", "tip"),
            ("size", "day"),
            ("smoker", "tip"),
            ("sex", "tip"),
        ],
        "description": "Restaurant tips data with various relationship types",
    }

    # 4. Diamonds dataset (for larger dataset testing)
    diamonds = sns.load_dataset("diamonds").sample(
        1000, random_state=42
    )  # Sample for speed
    # Create binary variable
    diamonds = diamonds[["carat", "price", "color", "cut", "depth", "table"]]
    diamonds["expensive"] = (
        diamonds["price"] > diamonds["price"].median()
    ).astype(int)
    diamonds = lowercase_string_columns(diamonds)
    datasets["diamonds"] = {
        "data": diamonds,
        "variables": {
            "carat": "continuous",
            "price": "continuous",
            "cut": "categorical",
            # "clarity": "categorical",
            "color": "categorical",
            "expensive": "binary",
            "depth": "continuous",
            "table": "continuous",
            # "x": "continuous",
            # "y": "continuous",
            # "z": "continuous",
        },
        "pairs_to_test": [
            ("carat", "price"),
            # ("carat", "cut"),
            # ("cut", "price"),
            # ("color", "price"),
            ("depth", "table"),
            ("price", "table"),
            # ("depth", "price"),
            # ("clarity", "price"),
            # ("clarity", "cut"),
            # ("price", "color"),
            # ("x", "price"),
            # ("x", "y"),
            # ("y", "price"),
            # ("y", "z"),
            # ("z", "price"),
            # ("z", "x"),
        ],
        "description": "Diamond characteristics and prices",
    }

    return datasets


def signed_cramers_v(x, y):
    """
    Calculate a signed version of Cramer's V that preserves direction information.
    Works for 2×2 tables and tables where at least one variable is binary.
    """
    # Create the contingency table
    table = pd.crosstab(x, y)

    # Calculate standard Cramer's V
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.sum().sum()
    min_dim = min(table.shape) - 1
    if min_dim == 0:
        return 0

    v = np.sqrt(chi2 / (n * min_dim))

    # Case 1: For 2×2 tables, use the phi coefficient directly
    if table.shape == (2, 2):
        phi = (
            table.iloc[0, 0] * table.iloc[1, 1]
            - table.iloc[0, 1] * table.iloc[1, 0]
        ) / np.sqrt(
            table.iloc[:, 0].sum()
            * table.iloc[:, 1].sum()
            * table.iloc[0, :].sum()
            * table.iloc[1, :].sum()
        )
        return np.sign(phi) * v

    # Case 2: For tables where one variable is binary (2×n or n×2)
    elif table.shape[0] == 2 or table.shape[1] == 2:
        # Determine if x or y is the binary variable
        if table.shape[0] == 2:  # 2×n table (rows are binary)
            # Calculate correlation between binary rows and column indices
            binary_var = np.repeat(
                [0, 1], table.shape[1]
            )  # Binary indicator for rows
            other_var = np.tile(np.arange(table.shape[1]), 2)  # Column indices
            weights = table.values.flatten()  # Cell frequencies
        else:  # n×2 table (columns are binary)
            # Calculate correlation between row indices and binary columns
            binary_var = np.tile(
                [0, 1], table.shape[0]
            )  # Binary indicator for columns
            other_var = np.repeat(np.arange(table.shape[0]), 2)  # Row indices
            weights = table.values.flatten()  # Cell frequencies

        # Calculate weighted correlation for direction
        total_weight = weights.sum()
        if total_weight > 0:
            # Normalize weights
            weights = weights / total_weight

            # Calculate weighted means
            mean_binary = np.sum(weights * binary_var)
            mean_other = np.sum(weights * other_var)

            # Calculate weighted covariance
            cov = np.sum(
                weights * (binary_var - mean_binary) * (other_var - mean_other)
            )

            # Calculate weighted variances
            var_binary = np.sum(weights * (binary_var - mean_binary) ** 2)
            var_other = np.sum(weights * (other_var - mean_other) ** 2)

            # Calculate correlation
            if var_binary > 0 and var_other > 0:
                corr = cov / np.sqrt(var_binary * var_other)
                return np.sign(corr) * v

    # For all other tables, return unsigned Cramer's V
    return v


def visualize_relationship(df, x_col, y_col, x_type, y_type):
    """
    Visualize the relationship between two variables based on their types.
    """
    plt.figure(figsize=(10, 6))

    if x_type == "continuous" and y_type == "continuous":
        # Scatter plot for continuous vs continuous
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"Relationship between {x_col} and {y_col}")

    elif (
        x_type == "continuous"
        and y_type in ["categorical", "binary", "ordinal"]
    ) or (
        y_type == "continuous"
        and x_type in ["categorical", "binary", "ordinal"]
    ):
        # Box plot for continuous vs categorical/binary/ordinal
        if x_type == "continuous":
            sns.boxplot(data=df, x=y_col, y=x_col)
            plt.title(f"Distribution of {x_col} by {y_col}")
        else:
            sns.boxplot(data=df, x=x_col, y=y_col)
            plt.title(f"Distribution of {y_col} by {x_col}")

    elif x_type in ["categorical", "binary", "ordinal"] and y_type in [
        "categorical",
        "binary",
        "ordinal",
    ]:
        # Mosaic plot or heatmap for categorical relationships
        contingency = pd.crosstab(df[x_col], df[y_col])
        plt.imshow(contingency, cmap="viridis")
        plt.colorbar(label="Count")
        plt.xticks(range(len(contingency.columns)), contingency.columns)
        plt.yticks(range(len(contingency.index)), contingency.index)
        plt.xlabel(y_col)
        plt.ylabel(x_col)
        plt.title(f"Contingency Table: {x_col} vs {y_col}")

    plt.tight_layout()
    plt.show()


def get_random_test_for_types(x_type, y_type):
    """
    Randomly select an appropriate test for a given pair of variable types.

    Args:
        x_type: Type of first variable ('continuous', 'ordinal', 'categorical', 'binary')
        y_type: Type of second variable ('continuous', 'ordinal', 'categorical', 'binary')

    Returns:
        A randomly selected test name suitable for this variable type combination
    """
    types = sorted([x_type, y_type])
    test_options = {
        ("continuous", "continuous"): [
            "pearson",
            # "unpaired_t_test",
            # "paired_t_test",
        ],
        ("continuous", "ordinal"): [
            "spearman",
            # "wilcoxon_mann_whitney",
            # "wilcoxon_signed_rank",
        ],
        ("ordinal", "ordinal"): ["spearman"],
        ("categorical", "continuous"): [
            # "kruskal_wallis",
            # "friedman",
            "one_way_anova",
        ],
        ("categorical", "ordinal"): [
            # "kruskal_wallis",
            # "friedman",
            "one_way_anova",
        ],
        ("binary", "continuous"): [
            # "wilcoxon_mann_whitney",
            "unpaired_t_test"
        ],
        ("binary", "ordinal"): [
            # "wilcoxon_mann_whitney",
            "unpaired_t_test"
        ],
        ("categorical", "categorical"): ["chi_square"],
        ("binary", "categorical"): ["chi_square"],
        ("binary", "binary"): ["chi_square"],
    }

    if tuple(types) in test_options:
        return random.choice(test_options[tuple(types)])

    print(
        f"Warning: No specific test defined for {x_type} vs {y_type}, defaulting to spearman"
    )
    return "spearman"


def analyze_correlation(df, x_col, y_col, x_type, y_type):
    """
    Analyze correlation between two variables and compare direct correlation with estimated correlation.
    """
    print(
        f"Analyzing relationship between {x_col} ({x_type}) and {y_col} ({y_type})"
    )

    # Get data and handle missing values
    data = df[[x_col, y_col]].dropna()
    x = data[x_col]
    y = data[y_col]

    # Calculate direct correlation when possible
    real_corr = None

    try:
        # 1. Continuous vs Continuous: Pearson correlation
        if x_type == "continuous" and y_type == "continuous":
            real_corr = x.corr(y, method="pearson")
            direct_corr_type = "pearson"
            print(f"Direct Pearson correlation: {real_corr:.4f}")

        # 2. At least one ordinal, both numeric: Spearman correlation
        elif x_type in ["continuous", "ordinal"] and y_type in [
            "continuous",
            "ordinal",
        ]:
            real_corr = x.corr(y, method="spearman")
            direct_corr_type = "spearman"
            print(f"Direct Spearman correlation: {real_corr:.4f}")

        # 3. Binary vs Continuous: Point-biserial correlation
        elif (x_type == "binary" and y_type == "continuous") or (
            x_type == "continuous" and y_type == "binary"
        ):
            binary_var, cont_var = (x, y) if x_type == "binary" else (y, x)
            # Ensure binary variable is not string; parse from Yes, yes, Y, etc into 0, 1
            if binary_var.dtype == "object" or pd.CategoricalDtype.is_dtype(
                binary_var
            ):
                binary_var = binary_var.astype(str).str.lower()
                binary_var = binary_var.map(
                    lambda x: (
                        1
                        if x in ["Yes", "Y", "True", "yes", "y", "true", "1"]
                        else 0
                    )
                )
            # Ensure binary variable is numeric
            r, p = pointbiserialr(binary_var, cont_var)
            real_corr = r
            direct_corr_type = "point_biserial"
            print(f"Direct Point-Biserial correlation: {real_corr:.4f}")

        # 4. Binary vs Binary: Phi coefficient
        elif x_type == "binary" and y_type == "binary":
            table = pd.crosstab(x, y)
            chi2, p, dof, expected = chi2_contingency(table)
            n = table.sum().sum()
            phi = np.sqrt(chi2 / n)
            real_corr = phi
            direct_corr_type = "phi_coefficient"
            print(f"Direct Phi coefficient: {real_corr:.4f}")

        # 5. Categorical vs Categorical: Cramer's V and 6. Binary vs Categorical: Cramer's V
        elif (x_type == "binary" and y_type == "categorical") or (
            x_type == "categorical"
            and (y_type == "binary" or y_type == "categorical")
        ):
            # Standard Cramer's V
            table = pd.crosstab(x, y)
            chi2, p, dof, expected = chi2_contingency(table)
            n = table.sum().sum()
            min_dim = min(table.shape) - 1
            if min_dim > 0:
                v = np.sqrt(chi2 / (n * min_dim))
                real_corr = v
                direct_corr_type = "cramers_v"
                print(f"Direct Cramer's V: {real_corr:.4f}")

                # Also calculate signed version for comparison
                real_corr = signed_cramers_v(x, y)
                print(f"Signed Cramer's V: {real_corr:.4f}")

        # 7. Categorical vs Continuous: Correlation ratio (Eta)
        elif (x_type == "categorical" and y_type == "continuous") or (
            x_type == "continuous" and y_type == "categorical"
        ):
            cat_var, cont_var = (x, y) if x_type == "categorical" else (y, x)
            categories = cat_var.unique()

            # Between-group variance
            cat_means = [cont_var[cat_var == cat].mean() for cat in categories]
            cat_sizes = [cont_var[cat_var == cat].size for cat in categories]

            overall_mean = cont_var.mean()
            overall_var = cont_var.var(ddof=1)

            # Between-group sum of squares
            bgss = sum(
                [
                    size * ((mean - overall_mean) ** 2)
                    for size, mean in zip(cat_sizes, cat_means)
                ]
            )
            # Calculate eta-squared
            eta_squared = bgss / (overall_var * len(cont_var))
            real_corr = np.sqrt(eta_squared)
            direct_corr_type = "eta_correlation_ratio"
            print(f"Direct Correlation Ratio (Eta): {real_corr:.4f}")

        # 8. ordinal vs Binary/Categorical
        elif (x_type == "ordinal" and y_type in ["binary", "categorical"]) or (
            y_type == "ordinal" and x_type in ["binary", "categorical"]
        ):
            # Use appropriate association measure based on the ordinal variable range
            disc_var, other_var = (x, y) if x_type == "ordinal" else (y, x)
            other_type = x_type if x_type != "ordinal" else y_type

            n_unique = len(disc_var.unique())
            if n_unique <= 5:  # Treat small ordinal range as categorical
                table = pd.crosstab(x, y)
                chi2, p, dof, expected = chi2_contingency(table)
                n = table.sum().sum()
                min_dim = min(table.shape) - 1
                if min_dim > 0:
                    v = np.sqrt(chi2 / (n * min_dim))
                    real_corr = v
                    direct_corr_type = "cramers_v"
                    print(
                        f"Direct Cramer's V (ordinal-{other_type}): {real_corr:.4f}"
                    )
                    real_corr = signed_cramers_v(x, y)
                    print(f"Signed Cramer's V: {real_corr:.4f}")
            else:  # Treat large ordinal range more like continuous
                if other_type == "binary":
                    # Point-biserial if other var is binary
                    bin_var, disc_var = (
                        (other_var, disc_var)
                        if other_type == "binary"
                        else (disc_var, other_var)
                    )
                    r, p = pointbiserialr(bin_var, disc_var)
                    real_corr = r
                    direct_corr_type = "point_biserial"
                    print(
                        f"Direct Point-Biserial (ordinal-binary): {real_corr:.4f}"
                    )
                else:
                    # Correlation ratio if other var is categorical
                    cat_var, disc_var = (
                        (other_var, disc_var)
                        if other_type == "categorical"
                        else (disc_var, other_var)
                    )
                    categories = cat_var.unique()

                    cat_means = [
                        disc_var[cat_var == cat].mean() for cat in categories
                    ]
                    cat_sizes = [
                        disc_var[cat_var == cat].size for cat in categories
                    ]

                    overall_mean = disc_var.mean()
                    overall_var = disc_var.var(ddof=1)

                    bgss = sum(
                        [
                            size * ((mean - overall_mean) ** 2)
                            for size, mean in zip(cat_sizes, cat_means)
                        ]
                    )
                    eta_squared = bgss / (overall_var * len(disc_var))
                    real_corr = np.sqrt(eta_squared)
                    direct_corr_type = "eta_correlation_ratio"
                    print(
                        f"Direct Correlation Ratio (ordinal-categorical): {real_corr:.4f}"
                    )
    except Exception as e:
        print(f"Could not calculate direct correlation: {e}")
    # Run robust_relation to get the appropriate test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        random_test = get_random_test_for_types(x_type, y_type)

        test_result = robust_relation(x, y, random_test)

    if test_result is None:
        print("No suitable test found for these variables")
        return None

    print(f"Selected test: {test_result.get('test_type', 'unknown')}")

    # Print test statistics
    stat_val = test_result.get("test_statistic", "N/A")
    p_val = test_result.get("p_value", "N/A")

    stat_val_str = (
        f"{stat_val:.4f}"
        if isinstance(stat_val, (int, float, np.number))
        else str(stat_val)
    )
    p_val_str = (
        f"{p_val:.4f}"
        if isinstance(p_val, (int, float, np.number))
        else str(p_val)
    )

    print(f"Test statistic: {stat_val_str}")
    print(f"p-value: {p_val_str}")

    # Convert test result to correlation coefficient
    study_data_mock = {"study_size": len(data)}
    if "group_sizes" in test_result:
        if isinstance(test_result["group_sizes"], dict):
            study_data_mock["group_sizes"] = test_result["group_sizes"]
        elif isinstance(test_result["group_sizes"], list):
            study_data_mock["group_sizes"] = {
                f"group_{i}": size
                for i, size in enumerate(test_result["group_sizes"])
            }

    estimated_corr = convert_test_to_correlation(test_result, study_data_mock)
    print(f"Estimated correlation: {estimated_corr:.4f}")

    # Compare with direct correlation if available
    if real_corr is not None:
        diff = abs(estimated_corr - real_corr)
        print(f"Difference from direct correlation: {diff:.4f}")

    return {
        "x_col": x_col,
        "y_col": y_col,
        "x_type": x_type,
        "y_type": y_type,
        "test_type": test_result.get("test_type"),
        "direct_corr": real_corr,
        "estimated_corr": estimated_corr,
        "test_statistic": stat_val,
        "p_value": p_val,
    }


def run_dataset_analysis(dataset_name, dataset_info):
    """
    Run analysis on all variable pairs in a dataset.
    """
    results = []
    df = dataset_info["data"]
    variable_types = dataset_info["variables"]
    # The configuration uses "pairs_to_test" which is a list of tuples (col1, col2)
    pairs_to_test_config = dataset_info.get("pairs_to_test", [])

    print(f"===== Analyzing {dataset_name} dataset =====")
    print(dataset_info["description"])
    print(f"Dataset shape: {df.shape}")
    # Use the correct key "pairs_to_test"
    print(f"Number of variable pairs to test: {len(pairs_to_test_config)}")
    print("=" * 50)

    for x_col_orig, y_col_orig in pairs_to_test_config:
        x_col = x_col_orig
        y_col = y_col_orig

        # Resolve "groups" to actual column name if present in the pair.
        # "groups" is a placeholder for the main grouping/target variable of a dataset.
        if x_col == "groups":
            if dataset_name in GROUP_MAPPING:
                x_col = GROUP_MAPPING[dataset_name]
            else:
                print(
                    f"Warning: 'groups' used for x_col in dataset '{dataset_name}' but no mapping found in GROUP_MAPPING. Skipping pair ({x_col_orig}, {y_col_orig})."
                )
                continue

        if y_col == "groups":
            if dataset_name in GROUP_MAPPING:
                y_col = GROUP_MAPPING[dataset_name]
            else:
                print(
                    f"Warning: 'groups' used for y_col in dataset '{dataset_name}' but no mapping found in GROUP_MAPPING. Skipping pair ({x_col_orig}, {y_col_orig})."
                )
                continue

        print("\n" + "-" * 50)

        # Retrieve variable types from the 'variables' dictionary
        try:
            type_x = variable_types[x_col]
            type_y = variable_types[y_col]
        except KeyError as e:
            print(
                f"Warning: Column {e} (derived from '{x_col_orig}' or '{y_col_orig}') not found in 'variables' for dataset '{dataset_name}'. Skipping pair ({x_col}, {y_col})."
            )
            continue

        # Visualize the relationship
        visualize_relationship(df, x_col, y_col, type_x, type_y)

        # Analyze correlation
        result = analyze_correlation(df, x_col, y_col, type_x, type_y)
        if result:
            results.append(result)
        print("-" * 50)

    return results


def generate_test_summary_report():
    """
    Generate a comprehensive report on correlation test conversions.
    """
    # Load all datasets
    datasets = load_test_datasets()

    # Collect results from all datasets
    all_results = []
    for dataset_name, dataset_info in datasets.items():
        print(f"Processing {dataset_name} dataset...")
        df = dataset_info["data"]

        for pair in dataset_info["variable_pairs"]:
            # Skip if too many missing values
            data = df[[pair["x"], pair["y"]]].dropna()
            if len(data) < 50:  # Skip if too few samples
                continue

            # Analyze correlation
            result = analyze_correlation(
                df, pair["x"], pair["y"], pair["type_x"], pair["type_y"]
            )
            if result:
                result["dataset"] = dataset_name
                all_results.append(result)

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)

    # Add type combination and difference columns
    summary_df["type_combo"] = (
        summary_df["x_type"] + " vs " + summary_df["y_type"]
    )
    summary_df["difference"] = summary_df.apply(
        lambda row: (
            abs(row["direct_corr"] - row["estimated_corr"])
            if not pd.isna(row["direct_corr"])
            and not pd.isna(row["estimated_corr"])
            else None
        ),
        axis=1,
    )

    # Save results to CSV
    summary_df.to_csv("correlation_test_summary.csv", index=False)
    print(
        f"Saved summary of {len(summary_df)} tests to correlation_test_summary.csv"
    )

    # Create visualizations
    create_summary_visualizations(summary_df)

    return summary_df


def create_summary_visualizations(summary_df):
    """Create visualizations for the summary report."""
    # Filter to rows with both correlations
    valid_df = summary_df.dropna(
        subset=["direct_corr", "estimated_corr", "difference"]
    )

    # 1. Scatter plot of direct vs estimated correlations
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=valid_df,
        x="direct_corr",
        y="estimated_corr",
        hue="test_type",
        s=100,
        alpha=0.7,
    )

    # Add perfect correlation line
    min_val = min(
        valid_df["direct_corr"].min(), valid_df["estimated_corr"].min()
    )
    max_val = max(
        valid_df["direct_corr"].max(), valid_df["estimated_corr"].max()
    )
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel("Direct Correlation")
    plt.ylabel("Estimated Correlation")
    plt.title("Direct vs. Estimated Correlations by Test Type")
    plt.grid(True)
    plt.legend(title="Test Type")
    plt.tight_layout()
    plt.savefig("correlation_comparison_by_test.png")

    # 2. Average difference by variable type combination
    plt.figure(figsize=(12, 6))
    avg_diff_by_type = (
        valid_df.groupby("type_combo")["difference"].mean().sort_values()
    )
    sns.barplot(x=avg_diff_by_type.index, y=avg_diff_by_type.values)
    plt.xticks(rotation=45)
    plt.title(
        "Average Correlation Estimation Error by Variable Type Combination"
    )
    plt.ylabel("Average Absolute Difference")
    plt.tight_layout()
    plt.savefig("correlation_error_by_type.png")

    # 3. Average difference by test type
    plt.figure(figsize=(12, 6))
    avg_diff_by_test = (
        valid_df.groupby("test_type")["difference"].mean().sort_values()
    )
    sns.barplot(x=avg_diff_by_test.index, y=avg_diff_by_test.values)
    plt.xticks(rotation=45)
    plt.title("Average Correlation Estimation Error by Test Type")
    plt.ylabel("Average Absolute Difference")
    plt.tight_layout()
    plt.savefig("correlation_error_by_test.png")

    print("Created 3 summary visualizations:")
    print("1. correlation_comparison_by_test.png")
    print("2. correlation_error_by_type.png")
    print("3. correlation_error_by_test.png")
