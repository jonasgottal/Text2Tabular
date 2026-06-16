import numpy as np
from text2tabular.reconstruction.utils.utils import make_positive_definite
from text2tabular.reconstruction.correlation.estimate_corr import (
    convert_test_to_correlation,
)


def create_correlation_matrix(study_data):
    """
    Create a single large correlation matrix representing within-group and
    potentially between-group correlations derived from study data.

    Args:
        study_data (dict): The dictionary returned by parse_json, containing
                           variables, groups, group_sizes, correlations,
                           statistical_tests, study_size etc.

    Returns:
        Tuple of the large correlation matrix, list of (variable, group) names,
        and mapping of (variable, group) names to indices.
        Returns (None, None, None) if essential data is missing.
    """
    # Validate and extract essential study data
    validation_result = _validate_study_data(study_data)
    if not validation_result["is_valid"]:
        return None, None, None

    # Extract validated data
    variables = validation_result["variables"]
    groups = validation_result["groups"]
    correlations = validation_result["correlations"]
    statistical_tests = validation_result["statistical_tests"]

    # Get all unique variables and create combined variable list
    all_vars = _extract_all_variables(
        variables, correlations, statistical_tests
    )
    if not all_vars:
        print("Warning: No variables found to create correlation matrix.")
        return None, None, None

    # Create combined (variable, group) list and index mapping
    combined_vars, combined_indices = _create_combined_variables(
        all_vars, groups
    )
    n_combined = len(combined_vars)

    # Initialize the correlation matrix
    large_corr_matrix = np.zeros((n_combined, n_combined))

    # Fill within-group correlations
    large_corr_matrix = _fill_within_group_correlations(
        large_corr_matrix,
        combined_indices,
        correlations,
        statistical_tests,
        groups,
        study_data,
    )

    # Fill between-group correlations for the same variable
    if statistical_tests and len(groups) > 1:
        large_corr_matrix = _fill_between_group_correlations(
            large_corr_matrix,
            combined_indices,
            statistical_tests,
            groups,
            study_data,
        )

    # Set diagonal to 1
    np.fill_diagonal(large_corr_matrix, 1.0)

    # Ensure the matrix is positive definite and valid
    try:
        large_corr_matrix = _finalize_correlation_matrix(large_corr_matrix)
    except Exception as e:
        print(f"Error during matrix finalization: {e}")
        return None, None, None

    return large_corr_matrix, combined_vars, combined_indices


def _validate_study_data(study_data):
    """Validate input data and set default values if needed."""
    result = {"is_valid": True}

    # Extract necessary info from study_data
    variables = study_data.get("variables", {})
    groups = study_data.get("groups", [])
    group_sizes = study_data.get("group_sizes", {})
    correlations = study_data.get("correlations", [])
    statistical_tests = study_data.get("statistical_tests", [])
    study_size = study_data.get("study_size")

    # Handle default group if none provided
    if not groups:
        groups = ["overall"]  # Treat as one group if none provided
        if study_size and not group_sizes:
            # Ensure group_sizes reflects the 'overall' group
            group_sizes = {"overall": study_size}
            study_data["group_sizes"] = (
                group_sizes  # Update study_data for consistency
            )
            study_data["groups"] = groups

    # Check for essential data
    if not variables:
        print(
            "Warning: 'variables' dictionary is empty in study_data. Cannot create matrix."
        )
        result["is_valid"] = False
        return result

    if not study_size and not all(group_sizes.values()):
        print(
            "Warning: Cannot determine study size or group sizes. Conversions may fail."
        )

    # Store validated data
    result.update(
        {
            "variables": variables,
            "groups": groups,
            "group_sizes": group_sizes,
            "correlations": correlations,
            "statistical_tests": statistical_tests,
            "study_size": study_size,
        }
    )

    return result


def _extract_all_variables(variables, correlations, statistical_tests):
    """Extract all unique variable names from the study data."""
    all_vars = []

    # Extract from variables dictionary
    var_type_keys = ["ordinal", "continuous", "categorical", "binary"]
    for var_type in var_type_keys:
        if var_type in variables:
            var_dict = variables[var_type]
            if isinstance(var_dict, dict):
                all_vars.extend(list(var_dict.keys()))
            else:
                print(
                    f"Warning: Expected dictionary for variables['{var_type}'], found {type(var_dict)}. Skipping."
                )

    # Add variables from correlations/tests not in the main variables dict
    temp_vars = set(all_vars)
    if correlations:
        for corr in correlations:
            for var in corr.get("variables", []):
                if var not in temp_vars:
                    all_vars.append(var)
                    temp_vars.add(var)

    if statistical_tests:
        for test_data in statistical_tests:
            for var in test_data.get("variables", []):
                if var not in temp_vars:
                    all_vars.append(var)
                    temp_vars.add(var)

    return sorted(list(set(all_vars)))  # Ensure unique and sorted order


def _create_combined_variables(all_vars, groups):
    """Create combined (variable, group) list and index mapping."""
    combined_vars = [(var, group) for group in groups for var in all_vars]
    combined_indices = {vg_pair: i for i, vg_pair in enumerate(combined_vars)}
    return combined_vars, combined_indices


def _fill_within_group_correlations(
    large_corr_matrix,
    combined_indices,
    correlations,
    statistical_tests,
    groups,
    study_data,
):
    """Fill correlation matrix with within-group correlations."""
    for group in groups:
        # Process explicit correlations first
        if correlations:
            large_corr_matrix = _process_explicit_correlations(
                large_corr_matrix, combined_indices, correlations, group
            )

        # Fill remaining gaps from statistical tests
        if statistical_tests:
            large_corr_matrix = _process_statistical_tests(
                large_corr_matrix,
                combined_indices,
                statistical_tests,
                group,
                study_data,
            )

    return large_corr_matrix


def _process_explicit_correlations(
    large_corr_matrix, combined_indices, correlations, group
):
    """Process explicit correlations for a specific group."""
    for corr_entry in correlations:
        corr_groups = corr_entry.get("groups", [])
        # Correlation applies if general (no groups) or specific to this group
        if not corr_groups or group in corr_groups:
            variables_in_corr = corr_entry.get("variables", [])
            if len(variables_in_corr) >= 2:  # Must involve two variables
                var1, var2 = variables_in_corr[0], variables_in_corr[1]
                # Check if variables exist for this group context
                if (var1, group) in combined_indices and (
                    var2,
                    group,
                ) in combined_indices:
                    idx1 = combined_indices[(var1, group)]
                    idx2 = combined_indices[(var2, group)]
                    # Only fill if not already filled (avoid overwriting)
                    if idx1 != idx2 and large_corr_matrix[idx1, idx2] == 0:
                        corr_value = corr_entry.get("correlation_coefficient")
                        # Check if corr_value is valid before assigning
                        if corr_value is not None and not np.isnan(corr_value):
                            large_corr_matrix[idx1, idx2] = corr_value
                            large_corr_matrix[idx2, idx1] = corr_value
                        else:
                            print(
                                f"Warning: Skipping None/NaN correlation for ({var1}, {var2}) in group '{group}' from 'correlations'."
                            )

    return large_corr_matrix


def _process_statistical_tests(
    large_corr_matrix, combined_indices, statistical_tests, group, study_data
):
    """Process statistical tests for a specific group to derive correlations."""
    for test_data in statistical_tests:
        test_groups = test_data.get("groups", [])
        # Test applies if general or specific to this group
        if not test_groups or group in test_groups:
            variables_in_test = test_data.get("variables", [])
            # Test must involve exactly two variables for within-group correlation
            if len(variables_in_test) == 2:
                var1, var2 = variables_in_test[0], variables_in_test[1]
                # Check if variables exist for this group context
                if (var1, group) in combined_indices and (
                    var2,
                    group,
                ) in combined_indices:
                    idx1 = combined_indices[(var1, group)]
                    idx2 = combined_indices[(var2, group)]
                    # Only fill if not already filled
                    if idx1 != idx2 and large_corr_matrix[idx1, idx2] == 0:
                        corr_from_test = convert_test_to_correlation(
                            test_data, study_data
                        )
                        # Check if corr_from_test is valid before assigning
                        if corr_from_test is not None and not np.isnan(
                            corr_from_test
                        ):
                            corr_from_test = np.clip(corr_from_test, -1.0, 1.0)
                            large_corr_matrix[idx1, idx2] = corr_from_test
                            large_corr_matrix[idx2, idx1] = corr_from_test
                        else:
                            print(
                                f"Warning: Skipping None/NaN correlation for ({var1}, {var2}) in group '{group}' from test '{test_data.get('test_type')}'."
                            )

    return large_corr_matrix


def _fill_between_group_correlations(
    large_corr_matrix, combined_indices, statistical_tests, groups, study_data
):
    """Fill between-group correlations for the same variable."""
    # Iterate through all statistical tests
    for test_data in statistical_tests:
        test_vars = test_data.get("variables", [])
        test_groups_in_spec = test_data.get(
            "groups", []
        )  # Groups specified in the test
        test_type = test_data.get("test_type")

        multi_group_tests = [
            "one_way_anova",
            "ranova",
            "kruskal_wallis",
            "friedman",
        ]

        # Condition 1: Test involves ONE variable and explicitly lists TWO groups (e.g., t-test)
        # Exclude multi-group tests here so they are handled correctly even if they only have 2 groups
        if (
            len(test_vars) == 1
            and len(test_groups_in_spec) == 2
            and test_type not in multi_group_tests
        ):
            groupA, groupB = test_groups_in_spec[0], test_groups_in_spec[1]
            varX = test_vars[0]

            # Ensure this pair of groups is valid in the overall group structure
            # and that we are not double-processing (e.g. (A,B) vs (B,A))
            # This outer loop structure might need refinement if groups in test_groups_in_spec
            # are not always a subset of the main 'groups' list or if order matters.
            # For now, assume test_groups_in_spec are valid.

            if (varX, groupA) in combined_indices and (
                varX,
                groupB,
            ) in combined_indices:
                idxA = combined_indices[(varX, groupA)]
                idxB = combined_indices[(varX, groupB)]

                # Only fill if not already filled and not diagonal
                if idxA != idxB and large_corr_matrix[idxA, idxB] == 0:
                    corr_from_test = convert_test_to_correlation(
                        test_data, study_data
                    )
                    if corr_from_test is not None and not np.isnan(
                        corr_from_test
                    ):
                        corr_from_test = np.clip(corr_from_test, -1.0, 1.0)
                        large_corr_matrix[idxA, idxB] = corr_from_test
                        large_corr_matrix[idxB, idxA] = corr_from_test
                    else:
                        print(
                            f"Warning: Skipping None/NaN cross-group correlation for variable '{varX}' between '{groupA}' and '{groupB}' from test '{test_type}' (2-group case)."
                        )

        # Condition 2: Test involves ONE variable and MULTIPLE (>=2) groups (e.g., ANOVA, Kruskal-Wallis)
        elif (
            len(test_vars) == 1
            and len(test_groups_in_spec) >= 2
            and test_type in multi_group_tests
        ):
            varX = test_vars[0]
            # The correlation derived from this test applies between varX in one group
            # and varX in every other group listed in test_groups_in_spec.

            corr_from_multigroup_test = convert_test_to_correlation(
                test_data, study_data
            )

            if corr_from_multigroup_test is not None and not np.isnan(
                corr_from_multigroup_test
            ):
                corr_val_clipped = np.clip(
                    corr_from_multigroup_test, -1.0, 1.0
                )

                # Apply this correlation to all pairs of groups within test_groups_in_spec
                for i in range(len(test_groups_in_spec)):
                    for j in range(i + 1, len(test_groups_in_spec)):
                        groupA = test_groups_in_spec[i]
                        groupB = test_groups_in_spec[j]

                        if (varX, groupA) in combined_indices and (
                            varX,
                            groupB,
                        ) in combined_indices:
                            idxA = combined_indices[(varX, groupA)]
                            idxB = combined_indices[(varX, groupB)]

                            # Only fill if not already filled and not diagonal
                            if (
                                idxA != idxB
                                and large_corr_matrix[idxA, idxB] == 0
                            ):
                                large_corr_matrix[idxA, idxB] = (
                                    corr_val_clipped
                                )
                                large_corr_matrix[idxB, idxA] = (
                                    corr_val_clipped
                                )
                            # If already filled, we might have a conflict or a more specific value.
                            # Current logic prioritizes not overwriting.
            else:
                print(
                    f"Warning: Skipping None/NaN cross-group correlation for variable '{varX}' across groups {test_groups_in_spec} from test '{test_type}' (multi-group case)."
                )

    return large_corr_matrix


def _finalize_correlation_matrix(large_corr_matrix):
    """Ensure the correlation matrix is positive definite and properly normalized."""
    # Check for NaNs before making positive definite
    if np.isnan(large_corr_matrix).any():
        print(
            "Warning: NaN values found in the correlation matrix before positive definite adjustment. Check conversion results."
        )

    large_corr_matrix = make_positive_definite(large_corr_matrix)

    # Re-normalize diagonal to 1 after potential adjustment
    np.fill_diagonal(large_corr_matrix, 1.0)

    # Clip values again to ensure they are within [-1, 1] after adjustments
    large_corr_matrix = np.clip(large_corr_matrix, -1.0, 1.0)

    return large_corr_matrix
