import numpy as np
from text2tabular.reconstruction.utils.utils import make_positive_definite


def expand_for_anova(
    parsed_data,
    large_corr_matrix,
    combined_vars,
    combined_indices,
    categorical_data,
    binary_data,
):
    """
    Expand categorical variables for ANOVA analysis if applicable.

    This function:
    1. Detects if ANOVA tests are present in the data
    2. Extracts ANOVA variable pairs
    3. Expands the correlation matrix to handle categorical variables
    4. Converts categorical variables to dummy binary variables

    Args:
        parsed_data: Dictionary containing parsed JSON data
        large_corr_matrix: Combined correlation matrix for all variables
        combined_vars: List of (var, group) tuples in matrix order
        combined_indices: Dictionary mapping (var, group) to matrix indices
        categorical_data: Dictionary of categorical data by (var, group) keys
        binary_data: Dictionary of binary data by (var, group) keys

    Returns:
        Tuple containing:
        - Updated large correlation matrix
        - Updated combined variables list
        - Updated combined indices dictionary
        - Updated categorical data
        - Updated binary data
        - List of dummy variable names
        - Dictionary of removed categorical information
        - ANOVA pair (or None if no ANOVA)
    """
    dummy_vars = []  # List of dummy variable names (without group)
    removed_categorical_info = (
        {}
    )  # Store original categorical data before dummy creation
    anova_pair = None

    if parsed_data["has_anova"]:
        print("ANOVA test detected. Expanding categorical variable.")
        # Extract the first ANOVA pair found (adjust if multiple need handling)
        anova_pair = _extract_anova_pair(parsed_data)

        if anova_pair:
            categorical_var_to_expand, numeric_var_for_anova = anova_pair
            print(
                f"Found ANOVA pair: ({categorical_var_to_expand}, {numeric_var_for_anova})"
            )

            # 1. Expand the large correlation matrix
            (
                large_corr_matrix,
                combined_vars,
                combined_indices,
                dummy_vars,  # Get the base names of the dummy variables
            ) = expand_matrix_for_anova(
                large_corr_matrix,
                combined_vars,
                anova_pair,
                parsed_data,
            )

            # 2. Make the expanded matrix positive definite
            try:
                print("Making expanded matrix positive definite...")
                large_corr_matrix = make_positive_definite(large_corr_matrix)
                # Re-normalize diagonal to 1 and clip
                np.fill_diagonal(large_corr_matrix, 1.0)
                large_corr_matrix = np.clip(large_corr_matrix, -1.0, 1.0)
                print("Expanded matrix is now positive definite.")
            except np.linalg.LinAlgError:
                print(
                    "Warning: Could not make the expanded correlation matrix positive definite."
                )

            # 3. Adjust the base data: remove original categorical, add binary dummies
            print(
                f"Adjusting base data: Removing '{categorical_var_to_expand}', adding dummies..."
            )
            categorical_data, binary_data, removed_categorical_info = (
                reduce_categorical_extend_binary(
                    categorical_data,
                    binary_data,
                    parsed_data["variables"],
                    parsed_data["group_sizes"],
                    parsed_data["groups"],  # Pass the list of groups
                    categorical_var_to_expand,
                )
            )
            print("Base data adjusted.")
        else:
            print("Could not extract a valid ANOVA pair to expand.")
    else:
        print("No ANOVA test detected. Skipping expansion.")

    # Print status after potential expansion
    print(f"Matrix size after potential expansion: {large_corr_matrix.shape}")

    return (
        large_corr_matrix,
        combined_vars,
        combined_indices,
        categorical_data,
        binary_data,
        dummy_vars,
        removed_categorical_info,
        anova_pair,
    )


def _extract_anova_pair(data):
    # Get variable types
    categorical_vars = set(data["variables"].get("categorical", {}).keys())
    continuous_vars = set(data["variables"].get("continuous", {}).keys())
    ordinal_vars = set(data["variables"].get("ordinal", {}).keys())
    num_vars = continuous_vars | ordinal_vars

    for test in data.get("statistical_tests", []):
        if (
            test.get("test_type") == "one_way_anova"
            or test.get("test_type") == "ranova"
        ):
            if len(test["variables"]) == 2:
                # Only necessary for scenarious where one variable is categorical
                # if the data is grouped and ANOVA is for those groups, this is not necessary
                var1, var2 = test["variables"]
                # Return as (categorical, numeric)
                if var1 in categorical_vars and var2 in num_vars:
                    return (var1, var2)
                elif var2 in categorical_vars and var1 in num_vars:
                    return (var2, var1)
    return None  # Not found


def expand_matrix_for_anova(
    large_corr_matrix, combined_vars, anova_pair, data
):
    """
    Expands the large correlation matrix and variable list to one-hot-encode
    a categorical variable involved in an ANOVA test across all groups.

    Args:
        large_corr_matrix: Combined correlation matrix for all variables
        combined_vars: List of (var, group) tuples in matrix order
        anova_pair: Tuple (categorical_var, test_type) for ANOVA expansion
        data: Dictionary containing variable statistics by group

    Returns:
        Tuple containing:
        - expanded correlation matrix
        - expanded variable list
        - mapping of expanded variables to indices
        - list of dummy variable names
    """
    categorical_var, _ = anova_pair
    groups = data["groups"]
    n_groups = len(groups)

    # Extract categories and create dummy names
    categories, dummy_var_names = _extract_categorical_info(
        categorical_var, data
    )

    # Log expansion info
    print(
        f"Expanding '{categorical_var}' into {len(categories)} dummies ({dummy_var_names}) across {n_groups} groups for ANOVA."
    )

    # Create expanded variables list and index mappings
    expanded_vars_and_indices = _create_expanded_variables(
        combined_vars, categorical_var, dummy_var_names, categories
    )

    expanded_combined_vars = expanded_vars_and_indices[
        "expanded_combined_vars"
    ]
    old_to_new_indices = expanded_vars_and_indices["old_to_new_indices"]
    expanded_combined_indices = expanded_vars_and_indices[
        "expanded_combined_indices"
    ]

    # Create and fill expanded correlation matrix
    n_expanded = len(expanded_combined_vars)
    expanded_large_corr_matrix = np.eye(n_expanded)
    print(
        f"Matrix size expanded from {large_corr_matrix.shape[0]} to {n_expanded}"
    )

    # Fill the expanded matrix
    expanded_large_corr_matrix = _fill_expanded_matrix(
        large_corr_matrix,
        combined_vars,
        old_to_new_indices,
        categorical_var,
        categories,
        data,
    )

    return (
        expanded_large_corr_matrix,
        expanded_combined_vars,
        expanded_combined_indices,
        dummy_var_names,
    )


def _extract_categorical_info(categorical_var, data):
    """Extract categories and create dummy variable names for a categorical variable."""
    # Get all unique categories across groups
    all_categories = set().union(
        *(
            group_stats.keys()
            for group_stats in data["variables"]["categorical"][
                categorical_var
            ].values()
        )
    )
    categories = sorted(list(all_categories))
    n_cats = len(categories)

    if n_cats <= 1:
        print(
            f"Warning: Categorical variable '{categorical_var}' has {n_cats} categories. Expansion might not be meaningful."
        )

    # Create dummy variable names
    dummy_var_names = [f"{categorical_var}_{cat}" for cat in categories]

    return categories, dummy_var_names


def _create_expanded_variables(
    combined_vars, categorical_var, dummy_var_names, categories
):
    """Create expanded variable list and index mappings."""
    expanded_combined_vars = []
    expanded_combined_indices = {}
    old_to_new_indices = {}
    new_idx_counter = 0
    n_cats = len(categories)

    for old_idx, (var, group) in enumerate(combined_vars):
        if var == categorical_var:
            # Create dummy indices for the categorical variable
            dummy_indices = tuple(new_idx_counter + i for i in range(n_cats))

            # Add each dummy variable to the expanded list
            for i, dummy_name in enumerate(dummy_var_names):
                expanded_combined_vars.append((dummy_name, group))
                expanded_combined_indices[(dummy_name, group)] = dummy_indices[
                    i
                ]

            old_to_new_indices[old_idx] = dummy_indices
            new_idx_counter += n_cats
        else:
            # Non-categorical variables remain the same
            expanded_combined_vars.append((var, group))
            expanded_combined_indices[(var, group)] = new_idx_counter
            old_to_new_indices[old_idx] = new_idx_counter
            new_idx_counter += 1

    return {
        "expanded_combined_vars": expanded_combined_vars,
        "expanded_combined_indices": expanded_combined_indices,
        "old_to_new_indices": old_to_new_indices,
    }


def _fill_expanded_matrix(
    large_corr_matrix,
    combined_vars,
    old_to_new_indices,
    categorical_var,
    categories,
    data,
):
    """Fill the expanded correlation matrix with appropriate values."""
    n_expanded = len(
        sum(
            [
                [idx] if not isinstance(idx, tuple) else list(idx)
                for idx in old_to_new_indices.values()
            ],
            [],
        )
    )

    expanded_large_corr_matrix = np.eye(n_expanded)

    # Process each pair of variables in the original matrix
    for old_idx1, (var1, group1) in enumerate(combined_vars):
        for old_idx2, (var2, group2) in enumerate(combined_vars):
            if old_idx1 >= old_idx2:
                continue  # Skip diagonal and lower triangle

            new_indices1 = old_to_new_indices[old_idx1]
            new_indices2 = old_to_new_indices[old_idx2]
            original_corr = large_corr_matrix[old_idx1, old_idx2]

            is_cat1 = var1 == categorical_var
            is_cat2 = var2 == categorical_var

            # Handle different cases based on whether variables are categorical
            if not is_cat1 and not is_cat2:
                _handle_non_categorical_pair(
                    expanded_large_corr_matrix,
                    new_indices1,
                    new_indices2,
                    original_corr,
                )
            elif is_cat1 and is_cat2:
                _handle_both_categorical_pair(
                    expanded_large_corr_matrix,
                    new_indices1,
                    group1,
                    group2,
                    categorical_var,
                    categories,
                    data,
                )
            elif is_cat1 ^ is_cat2:
                _handle_mixed_pair(
                    expanded_large_corr_matrix,
                    new_indices1,
                    new_indices2,
                    var1,
                    var2,
                    group1,
                    group2,
                    categorical_var,
                    categories,
                    original_corr,
                    data,
                    is_cat1,
                )

    # Ensure symmetry and valid correlation matrix properties
    return _ensure_valid_correlation_matrix(expanded_large_corr_matrix)


def _handle_non_categorical_pair(expanded_matrix, idx1, idx2, corr_val):
    """Handle correlation between two non-categorical variables."""
    # For non-categorical variables, indices are single values
    expanded_matrix[idx1, idx2] = corr_val
    expanded_matrix[idx2, idx1] = corr_val


def _handle_both_categorical_pair(
    expanded_matrix,
    cat_indices,
    group1,
    group2,
    categorical_var,
    categories,
    data,
):
    """Handle correlation when both variables are the same categorical variable."""
    if group1 == group2:
        group_cats = data["variables"]["categorical"][categorical_var].get(
            group1, {}
        )
        total_count = sum(group_cats.values())

        if total_count > 0 and len(categories) > 1:
            # Calculate theoretical correlation for mutually exclusive categories
            # For categories with proportions p_i and p_j, correlation = -sqrt(p_i * p_j / ((1-p_i) * (1-p_j)))

            category_proportions = {
                cat: count / total_count for cat, count in group_cats.items()
            }

            for i, idx1 in enumerate(cat_indices):
                for j, idx2 in enumerate(cat_indices):
                    if i != j:
                        cat1 = categories[i]
                        cat2 = categories[j]
                        p1 = category_proportions.get(cat1, 0.0)
                        p2 = category_proportions.get(cat2, 0.0)

                        if p1 > 0 and p2 > 0 and p1 < 1 and p2 < 1:
                            # Theoretical correlation for mutually exclusive categories
                            corr_val = -np.sqrt(
                                (p1 * p2) / ((1 - p1) * (1 - p2))
                            )
                        else:
                            corr_val = 0.0

                        expanded_matrix[idx1, idx2] = corr_val





def _handle_mixed_pair(
    expanded_matrix,
    indices1,
    indices2,
    var1,
    var2,
    group1,
    group2,
    categorical_var,
    categories,
    original_corr,
    data,
    is_cat1,
):
    """Handle correlation when exactly one variable is categorical."""
    if group1 == group2:  # Only consider within-group correlations
        # Determine which indices belong to the categorical variable
        if is_cat1:
            cat_new_indices, other_new_idx = indices1, indices2
            current_group = group1
        else:
            cat_new_indices, other_new_idx = indices2, indices1
            current_group = group2

        group_cats = data["variables"]["categorical"][categorical_var].get(
            current_group, {}
        )
        total_count = sum(group_cats.values())

        if total_count > 0 and len(categories) > 0:
            # Calculate weights for each category based on frequency
            weights = np.array(
                [group_cats.get(cat, 0) / total_count for cat in categories]
            )

            # Distribute correlation based on category weights
            for i, cat_new_idx in enumerate(cat_new_indices):
                # Use square root of weight for variance partitioning
                corr_val = (
                    original_corr * np.sqrt(weights[i])
                    if weights[i] > 0
                    else 0
                )
                expanded_matrix[cat_new_idx, other_new_idx] = corr_val


def _ensure_valid_correlation_matrix(matrix):
    """Ensure the matrix is a valid correlation matrix (symmetric, diagonal=1, -1≤values≤1)."""
    # Make symmetric by copying upper triangle to lower triangle
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))

    # Ensure diagonal is 1
    np.fill_diagonal(matrix, 1.0)

    # Ensure all values are in valid correlation range
    return np.clip(matrix, -1.0, 1.0)


def reduce_categorical_extend_binary(
    categorical_data,
    binary_data,
    variables,
    group_sizes,
    groups,
    categorical_var,
):
    """
    Convert a categorical variable into multiple binary (dummy) variables.

    This function:
    1. Removes the original categorical variable from the dataset
    2. Creates binary indicator variables for each category
    3. Tracks the original categorical data for potential later reconstruction

    Args:
        categorical_data: Dictionary of categorical data by (var, group) keys
        binary_data: Dictionary of binary data by (var, group) keys
        variables: Dictionary containing variable information by data type and group
        group_sizes: Dictionary mapping group names to their sizes
        groups: List of group names
        categorical_var: Name of the categorical variable to convert

    Returns:
        Tuple containing:
        - Updated categorical_data (with target variable removed)
        - Updated binary_data (with added dummy variables)
        - Dictionary storing the removed categorical data for potential reconstruction
    """
    # Validate inputs and check if categorical variable exists
    if not _validate_categorical_variable(categorical_var, variables):
        return categorical_data, binary_data, {}

    # Get all unique categories across groups
    categories = _get_unique_categories(categorical_var, variables)

    # Store original data and prepare keys for removal
    removed_categorical_info = _store_original_data(
        categorical_data, categorical_var, groups
    )

    # Create binary dummy variables for each category and group
    binary_data = _create_binary_dummies(
        binary_data,
        removed_categorical_info,
        categories,
        categorical_var,
        groups,
        group_sizes,
    )

    # Remove original categorical variable entries
    categorical_data = _remove_original_categorical(
        categorical_data, removed_categorical_info
    )

    return categorical_data, binary_data, removed_categorical_info


def _validate_categorical_variable(categorical_var, variables):
    """Validate that the categorical variable exists in the variables dictionary."""
    if categorical_var not in variables.get("categorical", {}):
        print(
            f"Warning: Categorical variable '{categorical_var}' not found for dummy creation."
        )
        return False
    return True


def _get_unique_categories(categorical_var, variables):
    """Extract all unique categories for the categorical variable across all groups."""
    # Find all categories across groups
    all_categories = set().union(
        *(
            group_stats.keys()
            for group_stats in variables["categorical"][
                categorical_var
            ].values()
        )
    )
    return sorted(list(all_categories))


def _store_original_data(categorical_data, categorical_var, groups):
    """Store the original categorical data and prepare keys for removal."""
    removed_categorical_info = {}
    for group in groups:
        key = (categorical_var, group)
        if key in categorical_data:
            removed_categorical_info[key] = categorical_data[key]

    return removed_categorical_info


def _remove_original_categorical(categorical_data, removed_categorical_info):
    """Remove the original categorical variable entries from the data."""
    for key in removed_categorical_info.keys():
        if key in categorical_data:
            del categorical_data[key]

    return categorical_data


def _create_binary_dummies(
    binary_data,
    removed_categorical_info,
    categories,
    categorical_var,
    groups,
    group_sizes,
):
    """Create binary dummy variables for each category and group."""
    for group in groups:
        size = group_sizes.get(group)
        if not _validate_group_size(group, size):
            continue

        original_group_data = removed_categorical_info.get(
            (categorical_var, group)
        )

        if original_group_data is not None:
            binary_data = _create_dummies_from_data(
                binary_data,
                original_group_data,
                categories,
                categorical_var,
                group,
            )

    return binary_data


def _validate_group_size(group, size):
    """Validate that the group size exists and is valid."""
    if size is None:
        print(
            f"Warning: Size for group '{group}' not found. Skipping dummy creation for this group."
        )
        return False
    return True


def _create_dummies_from_data(
    binary_data, original_group_data, categories, categorical_var, group
):
    """Create dummy variables from the original categorical data."""
    for cat in categories:
        dummy_name = f"{categorical_var}_{cat}"
        is_category = np.array(original_group_data) == cat
        binary_data[(dummy_name, group)] = is_category.astype(int)

    return binary_data
