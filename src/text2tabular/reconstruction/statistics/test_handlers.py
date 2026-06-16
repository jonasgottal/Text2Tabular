import numpy as np
import pandas as pd
from scipy.stats import (
    f_oneway,
    pearsonr,
    spearmanr,
    chi2_contingency,
    chi2,
    mannwhitneyu,
    wilcoxon,
    ttest_rel,
    ttest_ind,
    friedmanchisquare,
    kruskal,
    f,
)

try:
    from statsmodels.contingency_tables import mcnemar as sm_mcnemar

    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

from text2tabular.reconstruction.statistics.validation import _get_data_helper


def _handle_correlation(test_name, data, var1_key, var2_key):
    """Handles Pearson and Spearman correlations."""
    result = {}
    x = np.asarray(_get_data_helper(data, var1_key, test_name))
    y = np.asarray(_get_data_helper(data, var2_key, test_name))
    if len(x) != len(y):
        raise ValueError(
            f"{test_name} requires arrays from '{var1_key}' and '{var2_key}' to have the same length."
        )
    if len(x) < 2:
        print(f"Warning: Need at least 2 pairs for {test_name} correlation.")
        result["test_statistic"], result["p_value"] = np.nan, np.nan
    else:
        if test_name == "pearson":
            corr_coef, p_value = pearsonr(x, y)
        else:  # spearman
            res = spearmanr(x, y)
            corr_coef, p_value = res.correlation, res.pvalue
        result["test_statistic"] = corr_coef
        result["p_value"] = p_value
    return result


def _handle_chi_square(data, var1_key, var2_key):
    """Handles Chi-square test."""
    result = {}
    study_data_info = {}
    var1_data = _get_data_helper(data, var1_key, "chi_square")
    var2_data = _get_data_helper(data, var2_key, "chi_square")
    n_total = 0  # Initialize N
    try:
        # Convert to pandas Series if not already
        if not isinstance(var1_data, pd.Series):
            var1_data = pd.Series(var1_data)
        if not isinstance(var2_data, pd.Series):
            var2_data = pd.Series(var2_data)

        # Align the series and drop missing pairs (pairwise deletion)
        var1_aligned, var2_aligned = var1_data.align(var2_data, join="inner")

        # Remove any remaining NaN values
        mask = pd.notna(var1_aligned) & pd.notna(var2_aligned)
        var1_clean = var1_aligned[mask]
        var2_clean = var2_aligned[mask]

        # Now both arrays have the same length
        contingency_table = pd.crosstab(var1_clean, var2_clean)
        n_total = contingency_table.sum().sum()  # Calculate N from table

        if contingency_table.empty or n_total == 0:
            print("Warning: Contingency table is empty for Chi-square test.")
            chi2_stat, p, dof = 0.0, 1.0, 0
        else:
            # Let scipy handle the math and potential warnings/errors
            chi2_stat, p, dof, expected = chi2_contingency(contingency_table)

        result["test_statistic"] = chi2_stat
        result["p_value"] = p
        study_data_info["df"] = dof
        study_data_info["n"] = n_total  # Store N
    except Exception as e:
        print(f"Error during chi-square: {e}")
        result["test_statistic"], result["p_value"] = np.nan, np.nan
        study_data_info["df"] = 0
        study_data_info["n"] = 0  # Set n to 0 on error
    result.update(study_data_info)
    return result


def _handle_mcnemar(data, table_key):
    """Handles McNemar test."""
    result = {}
    table_data = _get_data_helper(data, table_key, "mcnemar")
    table = np.asarray(table_data)
    if table.shape != (2, 2):
        raise ValueError(
            f"McNemar test requires the table from '{table_key}' to be 2x2."
        )

    n_total = table.sum()  # Get N from table sum
    result["n"] = n_total  # Store N

    if _has_statsmodels:
        test_result = sm_mcnemar(table, exact=False)
        result["test_statistic"], result["p_value"] = (
            test_result.statistic,
            test_result.pvalue,
        )
    else:
        print(
            "Warning: statsmodels not found. Using manual McNemar calculation."
        )
        b, c = table[0, 1], table[1, 0]
        if b + c == 0:
            stat, p = 0.0, 1.0
        else:
            # Continuity correction applied
            stat = (abs(b - c) - 1) ** 2 / (b + c) if abs(b - c) >= 1 else 0.0
            # Need scipy.stats.chi2 for sf method
            p = chi2.sf(stat, 1)
        result["test_statistic"], result["p_value"] = stat, p
    return result


def _handle_multi_sample_tests(test_name, data, sample_keys):
    """Handles Friedman, Kruskal-Wallis, and One-Way ANOVA."""
    result = {}
    study_data_info = {}
    if (
        sample_keys is None
        or not isinstance(sample_keys, list)
        or len(sample_keys) < 2
    ):
        raise ValueError(
            f"{test_name} requires 'sample_keys' (list of >= 2 keys)."
        )

    samples_data = [
        _get_data_helper(data, key, test_name) for key in sample_keys
    ]
    # Ensure samples are numpy arrays for consistent processing
    valid_samples = []
    for s in samples_data:
        clean_s = pd.Series(s).dropna().values
        if clean_s.size > 0:
            valid_samples.append(clean_s)
    # Filter out samples that are entirely empty *after* conversion to array
    valid_samples = [s for s in valid_samples if s.size > 0]

    k = len(valid_samples)
    study_data_info["k"] = k  # Store k for conversion
    total_n = sum(len(s) for s in valid_samples)  # Calculate total N
    study_data_info["total_n"] = total_n  # Store total N

    if k < 2:
        print(f"Warning: Less than 2 non-empty samples for {test_name}.")
        stat, p = 0.0, 1.0
        f_stat = 0.0  # For ANOVA case
        if test_name == "one_way_anova":
            result["test_statistic"] = f_stat
        else:
            result["test_statistic"] = stat
        result["p_value"] = p
    else:
        if test_name == "friedman":
            try:
                lengths = [len(s) for s in valid_samples]
                if len(set(lengths)) > 1:
                    raise ValueError(
                        "All samples must have the same length for Friedman test."
                    )
                n_per_condition = lengths[0]  # N per condition for Friedman
                if n_per_condition == 0:
                    raise ValueError(
                        "Samples cannot be empty for Friedman test."
                    )
                stat, p = friedmanchisquare(*valid_samples)
                study_data_info["n"] = (
                    n_per_condition  # Store N (per condition)
                )
                result["test_statistic"] = stat
                result["p_value"] = p
            except ValueError as e:
                print(f"Error during Friedman test: {e}")
                result["test_statistic"], result["p_value"] = 0.0, 1.0
                if "n" not in study_data_info:
                    study_data_info["n"] = 0  # Ensure n exists
        elif test_name == "kruskal_wallis":
            stat, p = kruskal(*valid_samples)
            result["test_statistic"] = stat
            result["p_value"] = p
            # study_data_info["total_n"] already calculated and stored above
        else:  # one_way_anova
            f_stat, p = f_oneway(*valid_samples)
            result["test_statistic"] = f_stat
            result["p_value"] = p
            study_data_info["group_means"] = [
                np.mean(s) for s in valid_samples
            ]
            study_data_info["group_sizes"] = [len(s) for s in valid_samples]
            # Calculate df for ANOVA conversion
            df_between = k - 1
            # total_n already calculated
            df_within = total_n - k
            study_data_info["df_between"] = df_between
            study_data_info["df_within"] = df_within
            # study_data_info["total_n"] already calculated and stored above

    result.update(study_data_info)
    return result


def _handle_paired_tests(test_name, data, var1_key, var2_key):
    """Handles Wilcoxon Signed-Rank and Paired T-test."""
    result = {}
    x = np.asarray(_get_data_helper(data, var1_key, test_name))
    y = np.asarray(_get_data_helper(data, var2_key, test_name))
    if len(x) != len(y):
        raise ValueError(
            f"{test_name} requires arrays from '{var1_key}' and '{var2_key}' to have the same length."
        )

    if test_name == "wilcoxon_signed_rank":
        diff = x - y
        diff = diff[diff != 0]  # Remove zero differences
        if len(diff) < 1:  # Check if any non-zero differences remain
            print(
                "Warning: No non-zero differences for Wilcoxon signed-rank test."
            )
            stat, p = 0.0, 1.0
        else:
            # Scipy's wilcoxon needs at least one data point after removing zeros
            try:
                stat, p = wilcoxon(diff, alternative="two-sided")
            except ValueError as e:
                print(
                    f"Warning during Wilcoxon signed-rank: {e}. Returning NaN."
                )
                stat, p = np.nan, np.nan  # Indicate failure more clearly
    else:  # paired_t_test
        if len(x) < 2:
            print("Warning: Need at least 2 pairs for paired t-test.")
            stat, p = 0.0, 1.0
        else:
            stat, p = ttest_rel(x, y, alternative="two-sided")
    result["test_statistic"] = stat
    result["p_value"] = p
    # Add n for potential conversion use
    result["n"] = len(x)  # Use original length before diff for paired n
    return result


def _handle_unpaired_tests(test_name, data, var1_key, var2_key):
    """Handles Wilcoxon Mann-Whitney and Unpaired T-test."""
    result = {}
    study_data_info = {}
    x = np.asarray(_get_data_helper(data, var1_key, test_name))
    y = np.asarray(_get_data_helper(data, var2_key, test_name))
    n1, n2 = len(x), len(y)

    if n1 < 1 or n2 < 1:  # Allow test if at least one observation per group
        print(
            f"Warning: Need at least 1 observation in each group for {test_name}."
        )
        stat, p = 0.0, 1.0
    else:
        if test_name == "wilcoxon_mann_whitney":
            # Check for zero variance which causes issues
            if np.var(x) == 0 and np.var(y) == 0 and np.mean(x) == np.mean(y):
                print(f"Warning: Both groups are identical for {test_name}.")
                stat, p = 0.0, 1.0  # Or handle as appropriate, maybe NaN stat?
            else:
                try:
                    stat, p = mannwhitneyu(x, y, alternative="two-sided")
                except ValueError as e:
                    print(
                        f"Warning during Mann-Whitney U: {e}. Returning NaN."
                    )
                    stat, p = np.nan, np.nan
        else:  # unpaired_t_test (Welch's)
            # ttest_ind handles n=1 case gracefully (returns nan)
            stat, p = ttest_ind(x, y, equal_var=False, alternative="two-sided")
    result["test_statistic"] = stat
    result["p_value"] = p
    study_data_info["group_sizes"] = {"group1": n1, "group2": n2}
    result.update(study_data_info)
    return result


def _handle_ranova(data):
    """Handles Repeated Measures ANOVA (from raw arrays or pre-calculated stats)."""
    result = {}
    study_data_info = {}

    # Pathway 1: Pre-calculated parameter fallback
    if "test_statistic" in data and "k" in data and "n" in data:
        f_stat = data["test_statistic"]
        k, n = data["k"], data["n"]  # k=conditions, n=subjects
        result["test_statistic"] = f_stat
        study_data_info.update({"k": k, "n": n})

        df1, df2 = k - 1, (n - 1) * (k - 1)
        study_data_info.update(
            {"df_between": df1, "df_within": df2}
        )  # Store DFs

        if df1 > 0 and df2 > 0:
            result["p_value"] = f.sf(f_stat, df1, df2)
        else:
            print(
                "Warning: Invalid degrees of freedom for RANOVA p-value calculation."
            )
            result["p_value"] = np.nan

        result.update(study_data_info)
        return result

    # Pathway 2: Compute from raw arrays
    valid_data = {}
    for key, val in data.items():
        # Exclude pre-calculated stat keys
        if key not in [
            "test_statistic",
            "k",
            "n",
            "df_between",
            "df_within",
        ] and isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            valid_data[key] = pd.Series(val)

    if len(valid_data) < 2:
        print("Warning: Need at least 2 conditions for RANOVA.")
        return {"test_statistic": np.nan, "p_value": np.nan}

    # Align by index and perform listwise deletion (removes subjects if any condition is missing)
    df_aligned = pd.DataFrame(valid_data).dropna()

    # Extract the cleaned components
    samples = [df_aligned[col].values for col in df_aligned.columns]

    k = len(samples)
    if k < 2:
        print(
            "Warning: Need at least 2 conditions for RANOVA after dropping missing."
        )
        return {"test_statistic": np.nan, "p_value": np.nan}

    n = len(samples[0]) if len(samples) > 0 else 0
    if n < 2:
        print(
            "Warning: Need at least 2 subjects with complete data for RANOVA."
        )
        return {"test_statistic": np.nan, "p_value": np.nan}

    # This shouldn't trigger anymore because dropna() aligns them perfectly, but leaving it as a safeguard
    if any(len(s) != n for s in samples):
        print("Warning: All groups must have same length (repeated measures).")
        return {"test_statistic": np.nan, "p_value": np.nan}

    try:
        # Create subjects (rows) x conditions (cols) matrix
        X = np.column_stack(samples)

        # Means
        grand_mean = np.mean(X)
        subject_means = np.mean(X, axis=1)
        condition_means = np.mean(X, axis=0)

        # Sum of Squares
        ss_total = np.sum((X - grand_mean) ** 2)
        ss_conditions = n * np.sum((condition_means - grand_mean) ** 2)
        ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)

        # Error sum of squares (avoid floating point negative zeros)
        ss_error = max(0, ss_total - ss_conditions - ss_subjects)

        # Degrees of Freedom
        df_conditions = k - 1
        df_error = (k - 1) * (n - 1)

        if df_error <= 0 or ss_error == 0:
            f_stat, p_val = np.nan, np.nan
        else:
            ms_conditions = ss_conditions / df_conditions
            ms_error = ss_error / df_error
            f_stat = ms_conditions / ms_error
            p_val = f.sf(f_stat, df_conditions, df_error)

        result["test_statistic"] = f_stat
        result["p_value"] = p_val
        study_data_info.update(
            {
                "k": k,
                "n": n,
                "df_between": df_conditions,
                "df_within": df_error,
            }
        )
    except Exception as e:
        print(f"Error during RANOVA calculation: {e}")
        result["test_statistic"], result["p_value"] = np.nan, np.nan

    result.update(study_data_info)
    return result
