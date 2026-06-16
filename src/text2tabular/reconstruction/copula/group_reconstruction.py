import numpy as np
from collections import Counter


def reconstruct_categorical_from_dummies(
    synthetic_data,
    parsed_data,
    anova_pair,
    dummy_vars,
    removed_categorical_info,
):
    """
    Reconstruct categorical variables from dummy variables created during ANOVA expansion.

    Args:
        synthetic_data: Dictionary of synthetic data with (var, group) keys
        parsed_data: Dictionary containing parsed JSON data
        anova_pair: Tuple containing the categorical and numeric variable for ANOVA
        dummy_vars: List of dummy variable names (without group)
        removed_categorical_info: Dictionary with original categorical data

    Returns:
        Dictionary of synthetic data with reconstructed categorical variables
    """
    if not parsed_data["has_anova"] or not anova_pair:
        return synthetic_data  # No reconstruction needed

    # Make a copy to avoid modifying the input directly
    synthetic_data = synthetic_data.copy()

    print("Reconstructing categorical variable from dummies...")
    categorical_var_to_reconstruct = anova_pair[0]

    for group in parsed_data["groups"]:
        # Reconstruct the original categorical variable from the generated dummies
        reconstructed_cat_data = dummies_to_categorical_with_marginals(
            synthetic_data,
            group,
            dummy_vars,  # List of base dummy names from expansion step
            removed_categorical_info,  # Original data saved during expansion
            categorical_var_to_reconstruct,
        )
        if reconstructed_cat_data is not None:
            synthetic_data[(categorical_var_to_reconstruct, group)] = (
                reconstructed_cat_data
            )
        else:
            print(
                f"Warning: Failed to reconstruct '{categorical_var_to_reconstruct}' for group '{group}'."
            )

    # Remove dummy variables from synthetic_data after collapsing
    print("Removing dummy variables...")
    keys_to_delete = []
    for group in parsed_data["groups"]:
        for dummy in dummy_vars:
            key = (dummy, group)
            if key in synthetic_data:
                keys_to_delete.append(key)

    for key in keys_to_delete:
        del synthetic_data[key]

    return synthetic_data


def dummies_to_categorical_with_marginals(
    synthetic_data,
    group,
    dummy_vars,
    removed_categorical_info,
    categorical_var,
):
    """
    Convert one-hot dummy variables back into a single categorical variable array,
    resolving conflicts to best match the original marginals for a specific group.

    Args:
        synthetic_data: Dictionary of synthetic data with (var, group) keys
        group: The group name to process
        dummy_vars: List of dummy variable names derived from categorical_var
        removed_categorical_info: Dictionary with original categorical data
        categorical_var: Name of the original categorical variable

    Returns:
        Numpy array with reconstructed categorical values
    """
    # Validate inputs and prepare data
    validation_result = _validate_dummy_reconstruction_inputs(
        synthetic_data,
        group,
        dummy_vars,
        removed_categorical_info,
        categorical_var,
    )

    if not validation_result["is_valid"]:
        return validation_result["default_return"]

    # Extract validated data
    n = validation_result["n_samples"]
    categories = validation_result["categories"]
    original_counts = validation_result["original_counts"]
    dummies_array = validation_result["dummies_array"]

    # Initialize arrays and counters
    categorical = np.full(n, None, dtype=object)
    current_counts = {cat: 0 for cat in categories}

    # Process samples in random order to avoid bias
    indices = np.arange(n)
    np.random.shuffle(indices)

    # First pass: Assign categories based on dummies and counts
    categorical = _assign_categories_first_pass(
        indices,
        dummies_array,
        categories,
        original_counts,
        current_counts,
        categorical,
    )

    # Second pass: Handle any remaining unassigned values
    categorical = _handle_unassigned_values(
        categorical, categories, original_counts, current_counts
    )

    # Final validation
    _validate_final_assignment(
        categorical, categories, original_counts, categorical_var, group
    )

    return categorical


def _validate_dummy_reconstruction_inputs(
    synthetic_data,
    group,
    dummy_vars,
    removed_categorical_info,
    categorical_var,
):
    """Validate inputs and prepare data for dummy-to-categorical reconstruction."""
    result = {"is_valid": False, "default_return": None}

    # Check if we have the required inputs
    key_for_group = (categorical_var, group)
    example_dummy_key = (
        (dummy_vars[0], group)
        if dummy_vars and (dummy_vars[0], group) in synthetic_data
        else None
    )

    if (
        not dummy_vars
        or key_for_group not in removed_categorical_info
        or not example_dummy_key
    ):
        print(
            f"Warning: Cannot reconstruct '{categorical_var}' for group '{group}'. Missing dummies, original info, or dummy data."
        )
        n = (
            len(synthetic_data.get(example_dummy_key, []))
            if example_dummy_key
            else 0
        )
        result["default_return"] = (
            np.full(n, None, dtype=object) if n > 0 else None
        )
        return result

    # Extract and prepare data
    n = len(synthetic_data[example_dummy_key])
    categories = [d.split(f"{categorical_var}_", 1)[1] for d in dummy_vars]

    # Get original distribution
    original_group_data = removed_categorical_info[key_for_group]
    orig_counts = Counter(original_group_data)
    for cat in categories:
        if cat not in orig_counts:
            orig_counts[cat] = 0

    # Get dummy arrays
    dummies_array = np.stack(
        [synthetic_data.get((d, group), np.zeros(n)) for d in dummy_vars],
        axis=1,
    )

    # Return validated and processed data
    result.update(
        {
            "is_valid": True,
            "n_samples": n,
            "categories": categories,
            "original_counts": orig_counts,
            "dummies_array": dummies_array,
        }
    )

    return result


def _calculate_deficits(categories, original_counts, current_counts):
    """Calculate deficits between original and current counts for each category."""
    return {
        cat: original_counts[cat] - current_counts[cat]
        for cat in categories
        if current_counts[cat] < original_counts[cat]
    }


def _handle_unassigned_values(
    categorical, categories, original_counts, current_counts
):
    """Handle any remaining unassigned values (second pass)."""
    unassigned_indices = np.where(categorical == None)[0]
    if len(unassigned_indices) > 0:
        print(
            f"  Info: Handling {len(unassigned_indices)} unassigned entries."
        )

        for i in unassigned_indices:
            # Calculate final deficits for remaining assignments
            final_deficits = _calculate_deficits(
                categories, original_counts, current_counts
            )

            if final_deficits:
                best_cat = max(final_deficits, key=final_deficits.get)
                categorical[i] = best_cat
                current_counts[best_cat] += 1
            else:
                # If no deficits left, assign randomly
                categorical[i] = np.random.choice(categories)

    return categorical


def _validate_final_assignment(
    categorical, categories, original_counts, categorical_var, group
):
    """Validate the final categorical assignment against original counts."""
    final_counts = Counter(categorical)
    count_mismatch = False

    for cat in categories:
        if final_counts.get(cat, 0) != original_counts.get(cat, 0):
            count_mismatch = True
            break

    if count_mismatch:
        print(
            f"  Warning for '{categorical_var}', group '{group}': Final counts {dict(final_counts)} != original {dict(original_counts)}"
        )


def _assign_categories_first_pass(
    indices,
    dummies_array,
    categories,
    original_counts,
    current_counts,
    categorical,
):
    """Assign categories based on active dummies and count constraints (first pass)."""
    for i in indices:
        active_indices = np.where(dummies_array[i] == 1)[0]
        num_active = len(active_indices)
        assigned_category = None

        # Calculate current deficits
        deficits = _calculate_deficits(
            categories, original_counts, current_counts
        )

        if num_active == 1:
            # Case 1: Exactly one dummy active
            assigned_category = _handle_single_active_dummy(
                active_indices, categories, deficits
            )
        elif num_active > 1:
            # Case 2: Multiple dummies active (conflict)
            assigned_category = _resolve_multiple_active_dummies(
                active_indices, categories, deficits
            )

        # Case 3: Zero dummies active OR conflict couldn't be resolved yet
        if assigned_category is None and deficits:
            assigned_category = max(deficits, key=deficits.get)

        # Fallback if still None
        if assigned_category is None:
            assigned_category = _select_fallback_category(
                categories, active_indices, num_active
            )

        # Assign the selected category
        if assigned_category is not None:
            categorical[i] = _finalize_category_assignment(
                assigned_category, deficits, current_counts, original_counts
            )

    return categorical


def _handle_single_active_dummy(active_indices, categories, deficits):
    """Handle case where exactly one dummy is active."""
    cat_idx = active_indices[0]
    cat = categories[cat_idx]
    return cat if cat in deficits else None


def _resolve_multiple_active_dummies(active_indices, categories, deficits):
    """Resolve conflicts when multiple dummies are active."""
    active_cats_with_deficit = [
        categories[idx]
        for idx in active_indices
        if categories[idx] in deficits
    ]
    if active_cats_with_deficit:
        # Assign the active category with the largest deficit
        return max(active_cats_with_deficit, key=lambda c: deficits[c])
    return None


def _select_fallback_category(categories, active_indices, num_active):
    """Select a fallback category when other approaches fail."""
    possible_cats = (
        [categories[idx] for idx in active_indices]
        if num_active > 0
        else categories
    )
    return np.random.choice(possible_cats) if possible_cats else None


def _finalize_category_assignment(
    assigned_category, deficits, current_counts, original_counts
):
    """Finalize category assignment considering count constraints."""
    if current_counts[assigned_category] < original_counts[assigned_category]:
        current_counts[assigned_category] += 1
        return assigned_category
    else:
        # Try finding another category with deficit
        remaining_deficits = {
            c: d for c, d in deficits.items() if c != assigned_category
        }
        if remaining_deficits:
            fallback_cat = max(remaining_deficits, key=remaining_deficits.get)
            current_counts[fallback_cat] += 1
            return fallback_cat
        else:
            # No deficits left, use original choice
            current_counts[assigned_category] += 1
            return assigned_category
