import numpy as np
import pandas as pd


def _select_or_validate_test(x_num, y_num, x_type, y_type, test_name):
    """Select appropriate test or validate user-specified test."""
    if test_name is None:
        # Automatic test selection
        return _auto_select_test(x_num, y_num, x_type, y_type)
    else:
        # User specified a test, validate and prepare data
        return _prepare_for_specific_test(
            x_num, y_num, x_type, y_type, test_name
        )


def _auto_select_test(x_num, y_num, x_type, y_type):
    """Automatically select appropriate test based on variable types."""
    type_key = tuple(sorted((x_type, y_type)))

    if type_key in AUTO_TEST_SELECTION_MAP:
        preparation_func = AUTO_TEST_SELECTION_MAP[type_key]
        # Call the preparation function
        test_info = preparation_func(x_num, y_num, x_type, y_type)
        if test_info[0] is None:
            print(
                f"Error: Data preparation failed for inferred types {type_key}."
            )
            return None
        return test_info
    else:
        print(
            f"Warning: No default test for types {type_key}. Defaulting to Spearman."
        )
        return _prepare_disc_cont(x_num, y_num, x_type, y_type)


def _prepare_for_specific_test(x_num, y_num, x_type, y_type, test_name):
    """Prepare data for a specific test selected by the user."""
    # Define mapping from test names to preparation functions
    test_prep_map = {
        "pearson": _prepare_cont_cont,
        "spearman": _prepare_disc_cont,
        "unpaired_t_test": _prepare_cont_cont,
        "wilcoxon_mann_whitney": _prepare_cont_cont,
        "chi_square": _prepare_cat_cat,
        "one_way_anova": _prepare_cont_cat,
        "kruskal_wallis": _prepare_cont_cat,
    }

    # For paired tests, use simple key assignment
    if test_name in ["paired_t_test", "wilcoxon_signed_rank"]:
        data_for_conduct = {"input_x": x_num, "input_y": y_num}
        key_args = {"var1_key": "input_x", "var2_key": "input_y"}
        return test_name, data_for_conduct, key_args

    # For tests with special preparation
    if test_name in test_prep_map:
        prep_func = test_prep_map[test_name]
        _, data_for_conduct, key_args = prep_func(x_num, y_num, x_type, y_type)
        if data_for_conduct is None:
            print(f"Error: Data preparation failed for test {test_name}.")
            return None
        return test_name, data_for_conduct, key_args

    # Default case for unrecognized tests
    print(
        f"Warning: No special preparation for test '{test_name}'. Using default."
    )
    data_for_conduct = {"input_x": x_num, "input_y": y_num}
    key_args = {"var1_key": "input_x", "var2_key": "input_y"}
    return test_name, data_for_conduct, key_args


def _prepare_cont_cont(x_num, y_num, x_type, y_type):
    """Prepare data for continuous vs continuous (Pearson)."""
    data_for_conduct = {"data_x": x_num, "data_y": y_num}
    key_args = {"var1_key": "data_x", "var2_key": "data_y"}
    return "pearson", data_for_conduct, key_args


def _prepare_cont_bin(x_num, y_num, x_type, y_type):
    """Prepare data for continuous/ordinal vs binary (Mann-Whitney U)."""
    # Continuous or ordinal variable
    cont_disc_var = x_num if x_type in ["continuous", "ordinal"] else y_num
    bin_var = y_num if x_type in ["continuous", "ordinal"] else x_num
    labels = pd.unique(
        bin_var[~np.isnan(bin_var)]
    )  # Ensure NaNs are ignored for unique labels
    if len(labels) != 2:
        print(
            "Warning: Binary variable does not have exactly 2 levels after NaN removal."
        )
        return None, None, None  # Indicate failure

    # Ensure labels are usable as dict keys (convert to string if numeric)
    label0_key = f"group_{str(labels[0])}"
    label1_key = f"group_{str(labels[1])}"

    data_for_conduct = {
        label0_key: cont_disc_var[bin_var == labels[0]],
        label1_key: cont_disc_var[bin_var == labels[1]],
    }
    # Mann-Whitney U handler expects var1_key, var2_key
    key_args = {"var1_key": label0_key, "var2_key": label1_key}
    return "wilcoxon_mann_whitney", data_for_conduct, key_args  # Changed test


def _prepare_cont_cat(x_num, y_num, x_type, y_type):
    """Prepare data for continuous/ordinal vs categorical (Kruskal-Wallis)."""
    # Continuous or ordinal variable
    cont_disc_var = x_num if x_type in ["continuous", "ordinal"] else y_num
    cat_var = y_num if x_type in ["continuous", "ordinal"] else x_num
    unique_groups = sorted(
        pd.unique(cat_var[~np.isnan(cat_var)])
    )  # Ignore NaNs for groups
    if len(unique_groups) < 2:
        print(
            "Warning: Need at least 2 categories for Kruskal-Wallis after NaN removal."
        )
        return None, None, None  # Indicate failure

    data_for_conduct = {}
    sample_keys_generated = []
    for i, group in enumerate(unique_groups):
        # Ensure group label is usable as dict key
        group_key = f"cat_group_{str(group)}_{i}"
        data_for_conduct[group_key] = cont_disc_var[cat_var == group]
        sample_keys_generated.append(group_key)

    # Kruskal-Wallis handler expects sample_keys
    key_args = {"sample_keys": sample_keys_generated}
    return "kruskal_wallis", data_for_conduct, key_args


def _prepare_disc_cont(x_num, y_num, x_type, y_type):
    """Prepare data for ordinal vs continuous/ordinal (Spearman)."""
    data_for_conduct = {"data_x": x_num, "data_y": y_num}
    key_args = {"var1_key": "data_x", "var2_key": "data_y"}
    return "spearman", data_for_conduct, key_args


def _prepare_cat_cat(x_num, y_num, x_type, y_type):
    """Prepare data for categorical/binary vs categorical/binary (Chi-Square)."""
    # Use numeric codes directly for crosstab
    data_for_conduct = {"cat_x": x_num, "cat_y": y_num}
    key_args = {"var1_key": "cat_x", "var2_key": "cat_y"}
    return "chi_square", data_for_conduct, key_args


AUTO_TEST_SELECTION_MAP = {
    ("continuous", "continuous"): _prepare_cont_cont,  # Use Spearman prep
    ("binary", "continuous"): _prepare_cont_bin,  # Use Mann-Whitney prep
    ("binary", "binary"): _prepare_cat_cat,  # Chi2 prep
    ("binary", "categorical"): _prepare_cat_cat,  # Chi2 prep
    ("categorical", "categorical"): _prepare_cat_cat,  # Chi2 prep
    ("categorical", "continuous"): _prepare_cont_cat,  # Kruskal prep
    ("binary", "ordinal"): _prepare_cont_bin,  # Use Mann-Whitney prep
    ("categorical", "ordinal"): _prepare_cont_cat,  # Use Kruskal prep
    ("continuous", "ordinal"): _prepare_disc_cont,  # Use Spearman prep
    ("ordinal", "ordinal"): _prepare_disc_cont,  # Use Spearman prep
}
