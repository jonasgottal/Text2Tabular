from text2tabular.reconstruction.statistics.validation import (
    _validate_and_determine_format,
)
import traceback
from text2tabular.reconstruction.statistics.validation import (
    _clean_input_arrays,
    _infer_data_types,
)

from text2tabular.reconstruction.statistics.test_selection import (
    _select_or_validate_test,
)
from text2tabular.reconstruction.statistics.test_handlers import (
    _handle_correlation,
    _handle_chi_square,
    _handle_mcnemar,
    _handle_paired_tests,
    _handle_unpaired_tests,
    _handle_multi_sample_tests,
    _handle_ranova,
)


TEST_REGISTRY = {
    "pearson": lambda data, var1_key, var2_key, **kwargs: _handle_correlation(
        "pearson", data, var1_key, var2_key
    ),
    "spearman": lambda data, var1_key, var2_key, **kwargs: _handle_correlation(
        "spearman", data, var1_key, var2_key
    ),
    "chi_square": lambda data, var1_key, var2_key, **kwargs: _handle_chi_square(
        data, var1_key, var2_key
    ),
    "mcnemar": lambda data, table_key, **kwargs: _handle_mcnemar(
        data, table_key
    ),
    "wilcoxon_signed_rank": lambda data, var1_key, var2_key, **kwargs: _handle_paired_tests(
        "wilcoxon_signed_rank", data, var1_key, var2_key
    ),
    "paired_t_test": lambda data, var1_key, var2_key, **kwargs: _handle_paired_tests(
        "paired_t_test", data, var1_key, var2_key
    ),
    "wilcoxon_mann_whitney": lambda data, var1_key, var2_key, **kwargs: _handle_unpaired_tests(
        "wilcoxon_mann_whitney", data, var1_key, var2_key
    ),
    "unpaired_t_test": lambda data, var1_key, var2_key, **kwargs: _handle_unpaired_tests(
        "unpaired_t_test", data, var1_key, var2_key
    ),
    "friedman": lambda data, sample_keys, **kwargs: _handle_multi_sample_tests(
        "friedman", data, sample_keys
    ),
    "kruskal_wallis": lambda data, sample_keys, **kwargs: _handle_multi_sample_tests(
        "kruskal_wallis", data, sample_keys
    ),
    "one_way_anova": lambda data, sample_keys, **kwargs: _handle_multi_sample_tests(
        "one_way_anova", data, sample_keys
    ),
    "ranova": lambda data, **kwargs: _handle_ranova(data),
}


def robust_relation(x, y, test_name=None, study_data=None):
    """
    Calculates a robust measure of relationship between x and y.

    If test_name is provided, it attempts to run that specific test.
    If test_name is None, it infers data types and selects an appropriate test.

    Args:
        x: First variable (array-like or list of samples for ANOVA/Kruskal/Friedman)
        y: Second variable (array-like, group labels, or None if x is list of samples)
        test_name: Specific test to run (e.g., "pearson", "chi_square"). Defaults to None.
        study_data: Additional study context. Defaults to None.

    Returns:
        dict: Test results or None if calculation fails.
    """
    # Initialize study data if not provided
    if study_data is None:
        study_data = {}

    # Validate inputs and determine input format
    input_format = _validate_and_determine_format(x, y, test_name)
    if not input_format:
        return None

    # Process data based on input format
    if input_format == "multisample":
        return _process_multisample_input(x, test_name, study_data)
    else:
        return _process_standard_input(x, y, test_name, study_data)


def conduct_test(
    test_name,
    data,
    var1_key=None,
    var2_key=None,
    sample_keys=None,
    table_key=None,
):
    """
    Conduct the specified statistical test using a registry of handlers.

    Args:
        test_name (str): Name of the test/correlation (e.g., "pearson", "chi_square").
                         Must exist as a key in TEST_REGISTRY.
        data (dict): Dictionary containing the data arrays/table under various keys.
        var1_key (str, optional): Key for the first variable/array.
        var2_key (str, optional): Key for the second variable/array.
        sample_keys (list[str], optional): List of keys for multiple sample arrays.
        table_key (str, optional): Key for the contingency table.

    Returns:
        dict: Results including 'test_type', statistic ('test_statistic'
              or 'effect_size'), p-value, and other relevant info (df, k, n, etc.).

    Raises:
        ValueError: If test_name is not supported or required keys are missing/invalid.
        NotImplementedError: For tests requiring pre-calculated stats if those are missing.
    """

    if test_name not in TEST_REGISTRY:
        raise ValueError(f"Test '{test_name}' not supported or misspelled.")

    handler = TEST_REGISTRY[test_name]

    try:
        # Pass all potential key arguments to the handler
        # The handler's lambda definition selects the ones it needs
        result_payload = handler(
            data=data,
            var1_key=var1_key,
            var2_key=var2_key,
            sample_keys=sample_keys,
            table_key=table_key,
        )
        # Add the test_type to the results
        result = {"test_type": test_name, **result_payload}
        return result
    except (ValueError, NotImplementedError, KeyError, IndexError) as e:
        # Catch errors related to missing keys, invalid data shapes, etc.
        print(f"Error executing test '{test_name}': {e}")
        # Optionally re-raise or return a specific error structure
        raise  # Re-raise the caught exception for clarity


def _execute_test(test_name, data_for_conduct, key_args):
    """Execute the selected statistical test and handle any errors."""
    if not test_name or not data_for_conduct:
        print("Error: Missing test name or prepared data.")
        return None

    try:
        return conduct_test(test_name, data_for_conduct, **key_args)
    except Exception as e:
        print(f"Error during test execution for '{test_name}': {e}")
        traceback.print_exc()
        return None


def _process_multisample_input(x, test_name, study_data):
    """Process multisample input (list of samples)."""
    # Prepare data for multisample tests
    sample_keys_generated = []
    data_for_conduct = {}

    # Create keys for each sample
    for i, sample in enumerate(x):
        group_key = f"group_{i}"
        data_for_conduct[group_key] = sample
        sample_keys_generated.append(group_key)

    key_args = {"sample_keys": sample_keys_generated}

    # Execute the test
    return _execute_test(test_name, data_for_conduct, key_args)


def _process_standard_input(x, y, test_name, study_data):
    """Process standard input format (x and y variables)."""
    # Clean and convert input arrays
    x_arr, y_arr = _clean_input_arrays(x, y, test_name)
    if x_arr is None:  # If cleaning failed
        return None

    # Infer data types
    x_num, x_type, y_num, y_type = _infer_data_types(x_arr, y_arr)
    if x_type is None or (y_arr is not None and y_type is None):
        print("Error: Failed to infer data types.")
        return None

    # Select test or validate user-specified test
    test_info = _select_or_validate_test(
        x_num, y_num, x_type, y_type, test_name
    )
    if test_info is None:
        return None

    chosen_test_name, data_for_conduct, key_args = test_info

    # Execute the test
    return _execute_test(chosen_test_name, data_for_conduct, key_args)
