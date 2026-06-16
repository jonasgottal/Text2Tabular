import numpy as np
from scipy.stats import norm
import traceback
import random


def _get_test_context_params(test_data, study_data):
    """
    Helper to derive N, k, group sizes etc. from JSON test entry and study context.
    Prioritizes values directly available in test_data if provided by robust_relation.

    Args:
        test_data: Dictionary containing statistical test information
        study_data: Dictionary containing study context information

    Returns:
        Dictionary of derived parameters needed for effect size conversion
    """
    #    {'study_size': 1000,
    #    'groups': ['0', '1'],
    #    'group_sizes': {'0': 500, '1': 500},
    #    'variables': {'ordinal': {},
    #    'statistical_tests': [{'variables': ['price'],
    #    'test_type': 'one_way_anova',
    #    'test_statistic': 998.9192173757565,
    #    'p_value': 1.731811519007751e-152,
    #    'groups': ['0', '1']},

    #    {'study_size': 1000,
    #    'groups': ['overall'],
    #    'group_sizes': {'overall': 1000},
    #    'variables': {'ordinal': {},
    #    'categorical': {'cut': {'overall': {'ideal': 402,
    #    'premium': 262,
    #    'very good': 213,
    #    'good': 92,
    #    'fair': 31}},

    #    'statistical_tests': [{'variables': ['price', 'cut'],
    #    'test_type': 'one_way_anova',
    #    'test_statistic': 998.9192173757565,
    #    'p_value': 1.731811519007751e-152,
    #    'group_means': ['3660.371', '4441.643', '5084.073', '2950.290', '3523.065']
    #    'groups': ['overall']},

    params = _initialize_params()

    # Extract test type for context-specific processing
    test_type = test_data.get("test_type")

    # Process parameters in order of priority
    params = _extract_direct_params(params, test_data)
    params = _derive_from_study_data(params, test_data, study_data, test_type)
    params = _calculate_degrees_of_freedom(params, test_type)
    params = _clean_nan_values(params)

    return params


def _initialize_params():
    """Initialize default parameter values."""
    return {
        "n": None,  # Sample size for single/paired group tests (e.g., number of pairs or subjects)
        "group_sample_sizes": None,  # List of sample sizes for each group/category in multi-group/categorical tests
        "n_total": None,  # Total sample size across all groups/categories involved in the test
        "k": None,  # Number of groups/conditions/categories
        "df": None,  # Degrees of freedom (general or for specific tests like t-test, chi-square)
        "df_between": None,  # Between-groups degrees of freedom (e.g., for ANOVA)
        "df_within": None,  # Within-groups degrees of freedom (e.g., for ANOVA)
    }


def _extract_direct_params(params, test_data):
    """Extract parameters directly from test_data, which has highest priority."""
    # Direct numerical parameters
    params["k"] = test_data.get("k", params["k"])
    params["n_total"] = test_data.get("total_n", params["n_total"])
    params["n"] = test_data.get("n", params["n"])
    params["df"] = test_data.get("df", params["df"])
    params["df_between"] = test_data.get("df_between", params["df_between"])
    params["df_within"] = test_data.get("df_within", params["df_within"])

    # Process group sizes if directly provided in test_data
    test_group_sizes_data = test_data.get("group_sizes")
    extracted_sizes = []

    if isinstance(test_group_sizes_data, list):
        if all(
            isinstance(s, (int, float)) and s > 0
            for s in test_group_sizes_data
        ):
            extracted_sizes = [
                s
                for s in test_group_sizes_data
                if isinstance(s, (int, float)) and s > 0
            ]
    elif isinstance(test_group_sizes_data, dict):
        sizes_from_dict = list(test_group_sizes_data.values())
        if all(isinstance(s, (int, float)) and s > 0 for s in sizes_from_dict):
            extracted_sizes = [
                s
                for s in sizes_from_dict
                if isinstance(s, (int, float)) and s > 0
            ]

    if extracted_sizes:
        params["group_sample_sizes"] = extracted_sizes
        # If n_total or k were not directly provided, derive them from group_sample_sizes
        if params["n_total"] is None:
            params["n_total"] = sum(extracted_sizes)
        if params["k"] is None:
            params["k"] = len(extracted_sizes)

    return params


def _get_group_sizes_from_study_context(test_data, study_data):
    """
    Tries to derive group/category sample sizes based on test_data and study_data.
    Handles two main scenarios:
    1. Comparison across predefined study groups (from study_data.group_sizes).
    2. Comparison across categories of a categorical variable (from study_data.variables.categorical).
    """
    test_variables = test_data.get("variables", [])
    # groups listed in the test, e.g., ['0', '1'] or ['overall']
    test_specific_groups_filter = test_data.get("groups")

    # Scenario 2: Categorical variable involved
    # Check if any of the test variables are listed as categorical in study_data
    if len(test_variables) > 0:  # Must have variables to check
        categorical_vars_in_study = study_data.get("variables", {}).get(
            "categorical", {}
        )
        cat_var_name_found = None
        # The test might involve one primary variable and one categorical variable,
        # or be a chi-square test on two categorical variables.
        # We need to find which variable in test_data.variables corresponds to a known categorical variable.
        for var_in_test in test_variables:
            if var_in_test in categorical_vars_in_study:
                cat_var_name_found = var_in_test
                break

        if cat_var_name_found:
            # Data for the identified categorical variable, e.g., {'overall': {'catA': 10, 'catB': 12}, 'group1': ...}
            cat_var_data_all_contexts = categorical_vars_in_study.get(
                cat_var_name_found
            )
            if isinstance(cat_var_data_all_contexts, dict):
                # Determine which context's (e.g., 'overall' or a specific study group's) categorical breakdown to use.
                context_key_for_counts = None
                if (
                    test_specific_groups_filter
                    and len(test_specific_groups_filter) == 1
                ):
                    # If test explicitly refers to 'overall' or a specific group like 'groupA'
                    potential_key = test_specific_groups_filter[0]
                    if potential_key in cat_var_data_all_contexts:
                        context_key_for_counts = potential_key

                if not context_key_for_counts:  # Fallback or default
                    if "overall" in cat_var_data_all_contexts:
                        context_key_for_counts = "overall"
                    # If 'overall' is not present, but there's only one other context, use that.
                    elif len(cat_var_data_all_contexts) == 1:
                        context_key_for_counts = list(
                            cat_var_data_all_contexts.keys()
                        )[0]

                if context_key_for_counts:
                    category_counts_dict = cat_var_data_all_contexts.get(
                        context_key_for_counts
                    )
                    if isinstance(category_counts_dict, dict):
                        sizes = [
                            s
                            for s in category_counts_dict.values()
                            if isinstance(s, (int, float)) and s > 0
                        ]
                        if sizes:  # Successfully extracted category sizes
                            return sizes

    # Scenario 1: Predefined study groups (from study_data.group_sizes)
    # This is used if Scenario 2 did not yield results, or for tests that inherently use study_data.group_sizes.
    study_group_sizes_map = study_data.get(
        "group_sizes", {}
    )  # e.g., {'groupA': 50, 'groupB': 50} or {'overall': 100}

    if not study_group_sizes_map or not isinstance(
        study_group_sizes_map, dict
    ):
        return None  # No group size information available from study_data.group_sizes

    # If test_specific_groups_filter is provided (e.g., ['groupA', 'groupB'] for a test on these specific groups)
    # and it's not just a single ['overall']
    if test_specific_groups_filter and not (
        len(test_specific_groups_filter) == 1
        and test_specific_groups_filter[0] == "overall"
    ):
        sizes = []
        for group_name in test_specific_groups_filter:
            size = study_group_sizes_map.get(group_name)
            if isinstance(size, (int, float)) and size > 0:
                sizes.append(size)
        if sizes:  # Return if we found sizes for the specified groups
            return sizes

    # If test_specific_groups_filter was ['overall'] or not specific enough,
    # and the test type implies multiple groups (e.g., ANOVA, Kruskal-Wallis),
    # consider all groups in study_group_sizes_map (excluding 'overall' if other specific groups exist).
    test_type = test_data.get("test_type")
    if test_type in [
        "one_way_anova",
        "kruskal_wallis",
        "unpaired_t_test",
        "wilcoxon_mann_whitney",
    ]:
        # Filter out 'overall' if other specific group sizes are present, as 'overall' would be total N.
        specific_group_sizes_dict = {
            k: v for k, v in study_group_sizes_map.items() if k != "overall"
        }

        candidate_sizes_source = (
            study_group_sizes_map  # Default to all study groups
        )
        if (
            specific_group_sizes_dict
        ):  # If specific groups exist, prefer them over an 'overall' entry
            candidate_sizes_source = specific_group_sizes_dict

        sizes = [
            s
            for s in candidate_sizes_source.values()
            if isinstance(s, (int, float)) and s > 0
        ]

        if test_type in ["one_way_anova", "kruskal_wallis"]:
            if len(sizes) >= 2:  # ANOVA/K-W need at least 2 groups
                return sizes
        elif test_type in ["unpaired_t_test", "wilcoxon_mann_whitney"]:
            # For 2-group tests, if exactly two sizes are found, use them.
            # This handles the case where study_data.group_sizes = {'g1': N1, 'g2': N2}
            if len(sizes) == 2:
                return sizes
            # If more than 2 sizes, it's ambiguous for a 2-group test without explicit test_specific_groups_filter
            # If test_specific_groups_filter *was* provided and led to 2 groups, it's handled above.

    return None  # Could not determine group sizes from this context


def _derive_from_study_data(params, test_data, study_data, test_type):
    """Derive parameters from study_data when not available in test_data."""

    # Attempt to derive group_sample_sizes if not already set by _extract_direct_params
    if params["group_sample_sizes"] is None:
        derived_group_sizes = _get_group_sizes_from_study_context(
            test_data, study_data
        )
        if derived_group_sizes:
            params["group_sample_sizes"] = derived_group_sizes
            if (
                params["k"] is None
            ):  # If k wasn't set, derive from new group_sample_sizes
                params["k"] = len(derived_group_sizes)
            if (
                params["n_total"] is None
            ):  # If n_total wasn't set, derive from new group_sample_sizes
                params["n_total"] = sum(derived_group_sizes)

    # Fallback for n_total from overall study_size if still missing
    overall_study_size = study_data.get("study_size")
    if params["n_total"] is None and overall_study_size is not None:
        params["n_total"] = overall_study_size

    # Derive 'n' for paired/single-group tests.
    # This function uses study_data.get("group_sizes") which is all_group_sizes_study.
    # And params["n_total"] if specific group size is not found.
    all_group_sizes_study = study_data.get("group_sizes", {})
    params = _derive_paired_sample_size(
        params, all_group_sizes_study, test_type
    )

    # Fallback for 'k' if it's still not determined, especially for ANOVA-type tests,
    # and group_sample_sizes could not be determined by _get_group_sizes_from_study_context.
    # This might happen if group_sizes are in study_data.group_sizes but not picked up by the new helper.
    if (
        params["k"] is None
        and params["group_sample_sizes"] is None
        and test_type in ["one_way_anova", "kruskal_wallis"]
    ):

        study_groups_data_map = study_data.get("group_sizes", {})
        if isinstance(study_groups_data_map, dict):
            # Consider groups specified in test_data.groups if available and not 'overall'
            test_groups_filter = test_data.get("groups")

            relevant_group_names_for_k = []
            if test_groups_filter and not (
                len(test_groups_filter) == 1
                and test_groups_filter[0] == "overall"
            ):
                relevant_group_names_for_k = [
                    g_name
                    for g_name in test_groups_filter
                    if g_name in study_groups_data_map
                ]
            else:  # Consider all keys in study_groups_data_map, excluding 'overall' if other specific keys exist
                specific_keys = [
                    k_name
                    for k_name in study_groups_data_map.keys()
                    if k_name != "overall"
                ]
                if specific_keys:
                    relevant_group_names_for_k = specific_keys
                # If only 'overall' exists, it doesn't define k for multiple groups.
                # Otherwise (no 'overall' or 'overall' plus others), use all keys.
                elif (
                    "overall" not in study_groups_data_map
                    or len(study_groups_data_map) > 1
                ):
                    relevant_group_names_for_k = list(
                        study_groups_data_map.keys()
                    )

            valid_sizes_for_k_inference = []
            for name in relevant_group_names_for_k:
                size = study_groups_data_map.get(name)
                if isinstance(size, (int, float)) and size > 0:
                    valid_sizes_for_k_inference.append(size)

            if (
                len(valid_sizes_for_k_inference) >= 2
            ):  # Need at least 2 groups for these tests
                params["k"] = len(valid_sizes_for_k_inference)
                # If group_sample_sizes is still None, and we found valid sizes here, populate it.
                # This is a fallback if _get_group_sizes_from_study_context missed them.
                if params["group_sample_sizes"] is None:
                    params["group_sample_sizes"] = valid_sizes_for_k_inference
                    if (
                        params["n_total"] is None
                    ):  # And update n_total if it was also None
                        params["n_total"] = sum(valid_sizes_for_k_inference)

    return params


def _derive_paired_sample_size(params, group_sizes, test_type):
    """Derive sample size for paired/single-group tests."""
    paired_tests = [
        "paired_t_test",
        "wilcoxon_signed_rank",
        "friedman",
        "ranova",
    ]

    if params["n"] is None and test_type in paired_tests:
        # Case 1: Single group size provided
        if isinstance(group_sizes, dict) and len(group_sizes) == 1:
            params["n"] = list(group_sizes.values())[0]
        # Case 2: Fall back to total sample size if available
        elif params["n_total"] is not None:
            params["n"] = params["n_total"]

    return params


def _calculate_degrees_of_freedom(params, test_type):
    """Calculate degrees of freedom if not directly provided."""
    # For paired t-test
    if (
        test_type == "paired_t_test"
        and params["df"] is None
        and params["n"] is not None
        and params["n"] >= 2
    ):
        params["df"] = params["n"] - 1

    # For unpaired t-test
    elif (
        test_type == "unpaired_t_test"
        and params["df"] is None
        and params["group_sample_sizes"] is not None
        and len(params["group_sample_sizes"]) == 2
        and all(
            isinstance(s, (int, float)) and s > 0
            for s in params["group_sample_sizes"]
        )
    ):
        n1, n2 = params["group_sample_sizes"]
        if (n1 + n2) >= 3:  # df must be at least 1
            params["df"] = n1 + n2 - 2

    # For chi-square tests
    elif test_type == "chi_square" and params["df"] is None:
        # Default assumption for 2x2 table or when specific df isn't provided.
        # More complex scenarios (e.g. R_rows x C_cols) would need k_rows and k_cols.
        # For now, if k is available and > 1 (e.g. from categorical variable analysis),
        # we could potentially use (k-1) if it's a goodness-of-fit, or (k_rows-1)*(k_cols-1) for contingency.
        # This part might need refinement if more detailed chi-square contexts are common.
        if params["k"] is not None and params["k"] > 1:
            # This is a simplification, assuming k refers to one dimension of a chi-square test (e.g. goodness of fit)
            # Or if it's a 2xk table, df would be (2-1)*(k-1) = k-1.
            # This is a heuristic and might not cover all chi-square df calculations.
            # params["df"] = params["k"] - 1 # Example: Goodness of fit with k categories
            pass  # Keeping df=1 as default for now unless more info is available

        if params["df"] is None:  # If not set by a more specific rule above
            params["df"] = 1  # Fallback default assumption for 2x2 table

    # For one-way ANOVA
    elif (
        test_type == "one_way_anova"
        and params["df_between"] is None
        and params["df_within"] is None
    ):
        if (
            params["k"] is not None
            and params["n_total"] is not None
            and params["k"] >= 2
            and params["n_total"] > params["k"]
        ):
            params["df_between"] = params["k"] - 1
            params["df_within"] = params["n_total"] - params["k"]

    # For repeated measures ANOVA
    elif (
        test_type == "ranova"
        and params["df_between"] is None
        and params["df_within"] is None
    ):
        if (
            params["k"] is not None
            and params["n"] is not None
            and params["k"] >= 2
            and params["n"] >= 2
        ):
            params["df_between"] = params["k"] - 1
            params["df_within"] = (params["n"] - 1) * (params["k"] - 1)

    return params


def _clean_nan_values(params):
    """Replace any NaN values with None for consistency."""
    for key, value in params.items():
        if isinstance(value, (float, int)) and np.isnan(value):
            params[key] = None

    return params


def _convert_pearson_spearman_to_r(test_data, study_data):
    """Return effect size directly for Pearson/Spearman from JSON."""
    # Assumes 'test_statistic' in the JSON test entry holds the correlation coefficient (r)
    r = test_data.get("test_statistic")
    if r is None:
        return np.nan
    return np.clip(r, -1.0, 1.0)


def _convert_chi_square_to_r(test_data, study_data):
    """Convert Chi-square statistic from JSON to Phi or sqrt(chi2/N)."""
    chi2_stat = test_data.get("test_statistic")
    params = _get_test_context_params(test_data, study_data)
    n = params.get("n_total")  # Use total N relevant to the test context
    df = params.get("df")  # Use inferred df (often assumed 1)

    if chi2_stat is None or n is None or n <= 0:
        return np.nan
    if np.isnan(chi2_stat) or np.isnan(n):
        return np.nan

    signum = random.choice([-1, 1])

    # If df is inferred as 1 (assumed 2x2), calculate Phi
    if df == 1:
        phi = np.sqrt(chi2_stat / n)
        phi = np.clip(phi, 0.0, 1.0)
        return signum * phi
    else:
        # If df is unknown or > 1, return sqrt(chi2/N) as a general measure
        print(
            "Warning: Returning sqrt(chi2/N) for Chi-square; interpretation differs from Cramer's V (df unknown/ > 1)."
        )
        assoc = np.sqrt(chi2_stat / n)
        assoc = np.clip(assoc, 0.0, 1.0)

        return signum * assoc


def _convert_mcnemar_to_r(test_data, study_data):
    """Convert McNemar - No standard correlation conversion exists. Returning NaN."""
    print(
        "Warning: No standard correlation equivalent for McNemar test. Returning NaN."
    )
    return np.nan


def _convert_wilcoxon_signed_rank_to_r(test_data, study_data):
    """Convert Wilcoxon signed-rank from JSON using p-value and derived N."""
    p_value = test_data.get("p_value")
    params = _get_test_context_params(test_data, study_data)
    n = params.get("n")  # N pairs

    if p_value is None or n is None or n <= 0:
        return np.nan
    if np.isnan(p_value) or np.isnan(n):
        return np.nan
    if p_value == 1.0:
        return 0.0
    if p_value <= 0.0 or p_value > 1.0:
        return np.nan

    try:
        z_abs = abs(norm.ppf(p_value / 2.0))
        if np.isinf(z_abs):
            return 1.0
    except ValueError:
        return np.nan

    r = z_abs / np.sqrt(n)
    return np.clip(r, 0.0, 1.0)


def _convert_mann_whitney_to_r(test_data, study_data):
    """Convert Mann-Whitney U from JSON using U statistic and derived Ns."""
    # Prioritize U statistic if available for more direct conversion
    u_stat = test_data.get("test_statistic")
    p_value = test_data.get("p_value")
    params = _get_test_context_params(test_data, study_data)

    group_sizes = params.get("group_sample_sizes")
    n_total = params.get("n_total")

    if (
        group_sizes is None
        or len(group_sizes) != 2
        or not all(isinstance(s, (int, float)) and s > 0 for s in group_sizes)
        or n_total is None
        or n_total <= 0
    ):
        print(
            "Warning: Missing or invalid group sizes for Mann-Whitney conversion."
        )
        return np.nan

    n1, n2 = group_sizes[0], group_sizes[1]

    # Method 1: Using U statistic to calculate Rank Biserial Correlation or Z -> r
    if u_stat is not None and not np.isnan(u_stat):
        try:
            # Calculate Z score from U (approximation for larger samples)
            mu = n1 * n2 / 2.0
            # Handle potential ties - variance formula is more complex with ties.
            # Using simplified variance formula (no ties assumed):
            sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1.0) / 12.0)

            if sigma_u == 0:
                # This happens if n1 or n2 is 0, or n1+n2+1 is 0, which shouldn't occur with checks above
                return 0.0

            # Apply continuity correction
            z = (u_stat - mu - 0.5 * np.sign(u_stat - mu)) / sigma_u

            # Convert Z to r (Rosenthal's formula)
            r = z / np.sqrt(n_total)
            return np.clip(r, -1.0, 1.0)

        except Exception as e:
            print(
                f"Warning: Error converting Mann-Whitney U={u_stat} to r: {e}. Falling back to p-value."
            )

    # Method 2: Fallback to p-value conversion if U is missing or invalid
    if p_value is not None and not np.isnan(p_value):
        try:
            if p_value == 1.0:
                return 0.0
            if p_value <= 0.0 or p_value > 1.0:
                return np.nan

            # Convert two-tailed p-value to Z-score
            z_abs = abs(norm.ppf(p_value / 2.0))
            if np.isinf(z_abs):
                return 1.0

            r = z_abs / np.sqrt(n_total)
            return np.clip(r, 0.0, 1.0)
        except Exception as e:
            print(
                f"Warning: Error converting Mann-Whitney p-value={p_value} to r: {e}."
            )

    print(
        "Warning: Insufficient data (U or p-value) for Mann-Whitney conversion."
    )
    return np.nan


def _convert_paired_t_test_to_r(test_data, study_data):
    """Convert paired t-test statistic from JSON to r using derived N."""
    t_stat = test_data.get("test_statistic")
    params = _get_test_context_params(test_data, study_data)
    df = params.get("df")

    if t_stat is None or df is None or df <= 0:
        return np.nan
    if np.isnan(t_stat) or np.isnan(df):
        return np.nan

    denominator = t_stat**2 + df
    if denominator == 0:
        return np.sign(t_stat) * 1.0

    r_squared = max(0, (t_stat**2) / denominator)
    r = np.sqrt(r_squared)
    r = np.copysign(r, t_stat)
    return np.clip(r, -1.0, 1.0)


def _convert_unpaired_t_test_to_r(test_data, study_data):
    """Convert unpaired t-test statistic from JSON to r using derived Ns."""
    t_stat = test_data.get("test_statistic")
    params = _get_test_context_params(test_data, study_data)
    df = params.get("df")  # N-2 approximation

    if t_stat is None or df is None or df <= 0:
        return np.nan
    if np.isnan(t_stat) or np.isnan(df):
        return np.nan

    denominator = t_stat**2 + df
    if denominator == 0:
        return np.sign(t_stat) * 1.0

    r_squared = max(0, (t_stat**2) / denominator)
    r = np.sqrt(r_squared)
    r = np.copysign(r, t_stat)
    return np.clip(r, -1.0, 1.0)


def _convert_friedman_to_r(test_data, study_data):
    """Convert Friedman test statistic from JSON to Kendall's W using derived N and k."""
    chi2_stat = test_data.get("test_statistic")
    params = _get_test_context_params(test_data, study_data)
    n = params.get("n")  # Number of subjects
    k = params.get("k")  # Number of conditions

    if chi2_stat is None or n is None or k is None or n <= 0 or k < 2:
        return np.nan
    if np.isnan(chi2_stat) or np.isnan(n) or np.isnan(k):
        return np.nan

    denominator = n * (k - 1)
    if denominator == 0:
        return np.nan

    kendalls_w = chi2_stat / denominator
    return np.clip(kendalls_w, 0.0, 1.0)


def _convert_kruskal_wallis_to_r(test_data, study_data):
    """Convert Kruskal-Wallis H from JSON to sqrt(eta_sq_H) using derived N_total and k."""
    h_stat = test_data.get("test_statistic")
    params = _get_test_context_params(test_data, study_data)
    n_total = params.get("n_total")
    k = params.get("k")  # Number of groups

    if h_stat is None or n_total is None or k is None or k < 2 or n_total < k:
        print(
            f"Warning: Missing or invalid info for Kruskal-Wallis conversion (H={h_stat}, N={n_total}, k={k})"
        )
        return np.nan
    if np.isnan(h_stat) or np.isnan(n_total) or np.isnan(k):
        return np.nan

    try:

        if (n_total - 1) <= 0:
            print(
                f"Warning: Cannot calculate eta_sq for Kruskal-Wallis with N={n_total}"
            )
            return np.nan

        if h_stat < k - 1:

            eta_squared = h_stat / (n_total - 1.0)
            eta_squared = max(0, min(eta_squared, 1))  # Clamp between 0 and 1

        else:
            # Use the more common eta_sq_H formula
            eta_squared = (h_stat - k + 1) / (n_total - k)
            eta_squared = max(0, min(eta_squared, 1))
        r_equiv = np.sqrt(eta_squared)
        signum = random.choice([-1, 1])
        r_equiv = np.clip(r_equiv, 0.0, 1.0)

        return signum * r_equiv

    except Exception as e:
        print(
            f"Warning: Could not convert Kruskal-Wallis H={h_stat} to r: {e}"
        )
        return np.nan


def _convert_anova_to_r(test_data, study_data):
    """Convert One-Way ANOVA F from JSON to sqrt(eta_sq) using derived DFs."""
    # Use test_statistic key from JSON
    f_stat = test_data.get("test_statistic")
    params = _get_test_context_params(test_data, study_data)
    df_between = params.get("df_between")
    df_within = params.get("df_within")

    if (
        f_stat is None
        or df_between is None
        or df_within is None
        or df_within <= 0
        or f_stat < 0
    ):
        if df_between == 0:
            return 0.0  # Only 1 group
        return np.nan
    if np.isnan(f_stat) or np.isnan(df_between) or np.isnan(df_within):
        return np.nan

    denominator = f_stat * df_between + df_within
    if denominator == 0:
        return (
            1.0 if f_stat > 0 else np.nan
        )  # Assume perfect fit if F>0, dfw=0

    eta_squared = max(0, (f_stat * df_between) / denominator)
    r_equiv = np.sqrt(eta_squared)
    return np.clip(r_equiv, 0.0, 1.0)


def _convert_ranova_to_r(test_data, study_data):
    """Convert RANOVA F from JSON to sqrt(partial_eta_sq) using derived DFs or p-value."""
    from scipy.stats import f  # Import F-distribution locally

    f_stat = test_data.get("test_statistic")
    p_value = test_data.get("p_value")
    params = _get_test_context_params(test_data, study_data)
    df_between = params.get("df_between")  # df effect
    df_within = params.get("df_within")  # df error

    if df_between is None or df_within is None or df_within <= 0:
        return np.nan

    if df_between == 0:
        return 0.0  # k=1 condition

    # Fallback to reverse-engineering F-stat from P-value
    if (
        (f_stat is None or np.isnan(f_stat))
        and p_value is not None
        and not np.isnan(p_value)
    ):
        try:
            if p_value >= 1.0:
                f_stat = 0.0
            elif p_value <= 0.0:
                return 1.0  # Max correlation
            else:
                # Inverse survival function gets F-stat from right-tailed p-value
                f_stat = f.isf(p_value, df_between, df_within)
        except Exception as e:
            print(
                f"Warning: Error converting RANOVA p-value={p_value} to F: {e}"
            )
            return np.nan

    if f_stat is None or np.isnan(f_stat) or f_stat < 0:
        return np.nan

    denominator = f_stat * df_between + df_within
    if denominator == 0:
        return 1.0 if f_stat > 0 else np.nan

    partial_eta_squared = max(0, (f_stat * df_between) / denominator)
    r_equiv = np.sqrt(partial_eta_squared)
    return np.clip(r_equiv, 0.0, 1.0)


def convert_test_to_correlation(test_data, study_data):
    """
    Convert statistical test result (from parsed JSON) to a correlation-like coefficient.

    Args:
        test_data (dict): Dictionary representing one entry from the
                          'statistical_tests' list in the parsed JSON.
        study_data (dict): Dictionary containing broader study context derived
                           from the parsed JSON (e.g., study_size, group_sizes, variables).

    Returns:
        float: A correlation-like coefficient (often r, phi, Kendall's W, or sqrt(eta_sq)).
               Returns np.nan if conversion is not possible or not defined.
    """

    effect_size = test_data.get("effect_size")
    if effect_size is not None:
        # Check if effect size is a number
        if isinstance(effect_size, (int, float, np.number)):
            return float(effect_size)

    test_type = test_data.get("test_type")

    if test_type is None:
        print("Error: 'test_type' missing in test_data entry.")
        return np.nan

    # Handle potential case variations if JSON wasn't lowercased (though parse_json should handle it)
    test_type = test_type.lower()

    if test_type not in CONVERSION_REGISTRY:
        print(
            f"Warning: No correlation conversion function defined for test type '{test_type}'."
        )
        return np.nan

    converter = CONVERSION_REGISTRY[test_type]

    try:
        # Pass the raw JSON test entry and the study context to the helper
        correlation_coefficient = converter(test_data, study_data)

        # Validate output
        if not isinstance(correlation_coefficient, (int, float, np.number)):
            print(
                f"Warning: Converter for '{test_type}' did not return a number."
            )
            return np.nan
        if np.isnan(correlation_coefficient):
            return np.nan  # Return NaN if helper explicitly returned it

        return float(correlation_coefficient)
    except Exception as e:
        print(f"Error during conversion for test '{test_type}': {e}")
        traceback.print_exc()
        return np.nan


CONVERSION_REGISTRY = {
    "pearson": _convert_pearson_spearman_to_r,
    "spearman": _convert_pearson_spearman_to_r,
    "chi_square": _convert_chi_square_to_r,
    "mcnemar": _convert_chi_square_to_r,
    "wilcoxon_signed_rank": _convert_wilcoxon_signed_rank_to_r,
    "paired_t_test": _convert_paired_t_test_to_r,
    "wilcoxon_mann_whitney": _convert_mann_whitney_to_r,
    "unpaired_t_test": _convert_unpaired_t_test_to_r,
    "friedman": _convert_friedman_to_r,  # Kendall's W
    "kruskal_wallis": _convert_kruskal_wallis_to_r,  # sqrt(eta_squared_H)
    "one_way_anova": _convert_anova_to_r,  # sqrt(eta_squared)
    "ranova": _convert_ranova_to_r,  # sqrt(partial_eta_squared)
}
