import numpy as np
import pandas as pd


def _validate_and_determine_format(x, y, test_name):
    """Validate inputs and determine if we have multisample or standard format."""
    # Check if we have multisample input (list of samples for ANOVA/Kruskal/Friedman)
    is_multisample = isinstance(x, (list, tuple)) and test_name in [
        "one_way_anova",
        "kruskal_wallis",
        "friedman",
        "ranova",
    ]

    # Validate that multisample input is compatible with the test
    if is_multisample and test_name in [
        "paired_t_test",
        "wilcoxon_signed_rank",
        "mcnemar",
    ]:
        print(f"Error: {test_name} cannot use multi-sample input.")
        return None

    # Validate input for automatic selection with multisample format
    if is_multisample and test_name is None:
        print(
            "Error: Automatic selection not supported with list of samples. Specify test_name."
        )
        return None

    return "multisample" if is_multisample else "standard"


def _clean_input_arrays(x, y, test_name):
    """Clean and convert input arrays, handling NaN values appropriately."""
    # Convert inputs to numpy arrays
    x_arr = np.asarray(x).copy()
    # Only convert y if it's not None
    y_arr = np.asarray(y).copy() if y is not None else None

    # Define tests that require paired data of the same length
    paired_tests = [
        "pearson",
        "spearman",
        "paired_t_test",
        "wilcoxon_signed_rank",
    ]

    # For paired tests, handle NaN values by removing corresponding pairs
    if test_name in paired_tests and y_arr is not None:
        if x_arr.shape != y_arr.shape:
            print(
                f"Error: Input arrays for paired test '{test_name}' have different shapes."
            )
            return None, None

        mask = ~(pd.isna(x_arr) | pd.isna(y_arr))
        if not np.all(mask):
            print(
                "Warning: NaN values detected in paired data. Removing corresponding pairs."
            )
            x_arr = x_arr[mask]
            y_arr = y_arr[mask]

        if len(x_arr) < 3:
            print(
                f"Error: Not enough non-NaN data points ({len(x_arr)} < 3) for paired test."
            )
            return None, None

    # For paired tests, y must be provided
    elif y_arr is None and test_name in paired_tests:
        print(
            f"Error: Paired test '{test_name}' requires two input arrays, but y is None."
        )
        return None, None

    return x_arr, y_arr


def _infer_data_types(x_arr, y_arr):
    """Infer the data types of input variables."""
    x_num, x_type = _infer_single_variable_type(x_arr)
    y_num, y_type = (
        _infer_single_variable_type(y_arr)
        if y_arr is not None
        else (None, None)
    )
    return x_num, x_type, y_num, y_type


def _infer_single_variable_type(arr):
    """Infer the type of a single variable array."""
    if arr is None:
        return None, None

    # Convert to pandas Series for easier handling
    arr_pd = pd.Series(arr)
    # Remove NaNs before type detection
    arr_pd = arr_pd.dropna()

    if arr_pd.empty:
        return arr_pd.values, None

    original_kind = arr_pd.dtype.kind if hasattr(arr_pd.dtype, "kind") else "O"
    is_original_numeric = pd.api.types.is_numeric_dtype(arr_pd.dtype)
    is_original_integer_like = original_kind in ("i", "u", "b")

    if not is_original_numeric:
        # Check if string values represent boolean concepts before factorizing
        if original_kind == "O":  # Object dtype (usually strings)
            # Define patterns for boolean-like values
            true_patterns = ["true", "yes", "y", "t", "1"]
            false_patterns = ["false", "no", "n", "f", "0"]

            # Convert to lowercase for comparison
            lower_vals = arr_pd.astype(str).str.lower()

            # Check if all values match boolean patterns
            all_bool = all(
                val in true_patterns or val in false_patterns
                for val in lower_vals
            )

            if all_bool and len(pd.unique(lower_vals)) <= 2:
                # Convert to binary numeric representation
                num_arr = np.array(
                    [
                        1.0 if val in true_patterns else 0.0
                        for val in lower_vals
                    ]
                )
                return num_arr, "binary"

        # If not identified as boolean, factorize as usual
        codes, _ = pd.factorize(arr_pd)
        num_arr = np.array(codes, dtype=float)
    else:
        num_arr = arr_pd.astype(float).values

    unique_vals = pd.unique(num_arr)
    n_unique = len(unique_vals)
    n_total = len(num_arr)

    # Determine variable type based on characteristics
    if n_unique == 2:
        return num_arr, "binary"

    # Check if values are integer-like
    is_int_like = np.all(np.equal(np.mod(unique_vals, 1), 0))

    # Check if there are relatively few unique values
    is_few_unique = n_unique < max(
        3, min(int(0.1 * n_total) if n_total > 0 else 3, 20)
    )

    # Calculate if converting to int would lose information
    # This checks if float values are exactly their integer equivalents
    would_lose_info = False
    if is_original_numeric and not is_original_integer_like:
        int_equivalent = num_arr.astype(int)
        would_lose_info = not np.array_equal(
            num_arr, int_equivalent.astype(float)
        )

    if (
        (not is_original_numeric or is_original_integer_like)
        and is_int_like
        and is_few_unique
    ):
        return num_arr, "categorical"
    elif (
        is_original_integer_like or (is_int_like and not would_lose_info)
    ) and is_few_unique:
        return int_equivalent, "ordinal"
    else:
        return num_arr, "continuous"


def _get_data_helper(data, key, test_name):
    """Helper to retrieve data and raise error if key missing."""
    if key is None:
        raise ValueError(
            f"Required key argument missing for test '{test_name}'."
        )
    if key not in data:
        raise ValueError(
            f"Key '{key}' not found in provided data dictionary for test '{test_name}'."
        )
    return data[key]
