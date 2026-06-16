import numpy as np
from scipy.stats import norm
from collections import Counter
from text2tabular.reconstruction.utils.utils import make_positive_definite

np.seterr(all="ignore")


def apply_gaussian_copula(
    ordinal_data,
    continuous_data,
    categorical_data,
    binary_data,
    large_corr_matrix,
    combined_vars,  # List of (var, group) tuples in matrix order
    group_sizes,  # Dict mapping group name to size
):
    """
    Apply Gaussian copula using the large correlation matrix to generate
    correlated synthetic data for all groups simultaneously.
    Defined in Haugh2016 (https://www.columbia.edu/~mh2078/QRM/Copulas.pdf)

    Args:
        ordinal_data: Dictionary of ordinal data by (var, group) keys
        continuous_data: Dictionary of continuous data by (var, group) keys
        categorical_data: Dictionary of categorical data by (var, group) keys
        binary_data: Dictionary of binary data by (var, group) keys
        large_corr_matrix: Combined correlation matrix for all variables across groups
        combined_vars: List of (var, group) tuples in matrix order
        group_sizes: Dict mapping group name to size

    Returns:
        Dictionary of synthetic data with (var, group) keys
    """
    n_combined = large_corr_matrix.shape[0]
    total_study_size = sum(group_sizes.values())

    # Generate correlated normal samples
    correlated_Z = _generate_correlated_samples(
        large_corr_matrix, total_study_size, n_combined
    )
    if correlated_Z is None:
        return None

    # Transform to uniform using the standard normal CDF
    U = norm.cdf(correlated_Z)

    # Transform back to original distributions group by group
    return _transform_to_original_distributions(
        U,
        ordinal_data,
        continuous_data,
        categorical_data,
        binary_data,
        combined_vars,
        group_sizes,
    )


def _generate_correlated_samples(corr_matrix, n_samples, n_vars):
    """
    Generate correlated standard normal samples based on the correlation matrix.

    Args:
        corr_matrix: Correlation matrix
        n_samples: Number of samples to generate
        n_vars: Number of variables

    Returns:
        Correlated standard normal samples or None if matrix decomposition fails
    """
    try:
        # Add small jitter for numerical stability if needed
        min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
        if min_eig < 1e-8:
            print(
                f"  Warning: Matrix min eigenvalue {min_eig:.2e} < 1e-8. Adding jitter."
            )
            corr_matrix = _stabilize_correlation_matrix(corr_matrix)

        L = np.linalg.cholesky(corr_matrix)
        Z = np.random.normal(0, 1, (n_samples, n_vars))
        return Z @ L.T
    except np.linalg.LinAlgError as e:
        print(
            f"ERROR: Cholesky decomposition failed: {e}. Check matrix positive definiteness."
        )
        return None


def _stabilize_correlation_matrix(matrix):
    """Add jitter to make the correlation matrix numerically stable."""
    matrix = matrix.copy()  # Avoid modifying the original
    matrix += 1e-8 * np.eye(matrix.shape[0])
    np.fill_diagonal(matrix, 1.0)  # Re-normalize diagonal
    return make_positive_definite(matrix)


def _transform_to_original_distributions(
    uniform_samples,
    ordinal_data,
    continuous_data,
    categorical_data,
    binary_data,
    combined_vars,
    group_sizes,
):
    """Transform uniform samples back to the original distributions for each variable and group."""
    synthetic_data = {}
    current_row = 0

    for group, group_size in group_sizes.items():
        group_U = uniform_samples[current_row : current_row + group_size, :]

        for i, (var, g) in enumerate(combined_vars):
            if g != group:
                continue  # Skip if not current group

            key = (var, g)
            var_U = group_U[:, i]  # Uniform samples for this var/group

            # Transform based on data type
            if key in ordinal_data:
                synthetic_data[key] = _transform_ordinal(
                    var_U, ordinal_data[key], group_size
                )
            elif key in continuous_data:
                synthetic_data[key] = _transform_continuous(
                    var_U, continuous_data[key], group_size
                )
            elif key in categorical_data:
                synthetic_data[key] = _transform_categorical(
                    var_U, categorical_data[key], group_size
                )
            elif key in binary_data:
                synthetic_data[key] = _transform_binary(
                    var_U, binary_data[key], group_size
                )

        current_row += group_size

    return synthetic_data


def _transform_ordinal(uniform_samples, original_data, group_size):
    """Transform uniform samples to ordinal data."""

    result = _transform_continuous(uniform_samples, original_data, group_size)

    return np.round(result).astype(int)


def _transform_continuous(uniform_samples, original_data, group_size):
    """
    Transform uniform samples to continuous data using empirical CDF.
    This preserves the exact shape of the original distribution.
    """

    if len(original_data) == 0:
        # If original data is empty, just return the uniform samples as is
        return uniform_samples

    if len(np.unique(original_data)) <= 1:
        # If all values are the same, return the constant
        return np.full(group_size, original_data[0])

    # Sort the original data
    sorted_data = np.sort(original_data)

    # Use linear interpolation to map from uniform to the empirical distribution
    result = np.interp(
        uniform_samples, np.linspace(0, 1, len(sorted_data)), sorted_data
    )

    return result


def _transform_categorical(uniform_samples, original_data, group_size):
    """Transform uniform samples to categorical data."""
    counts = Counter(original_data)
    categories = sorted(counts.keys())
    counts_list = [counts[cat] for cat in categories]
    sorted_idx = np.argsort(uniform_samples)

    group_var_data = np.empty(group_size, dtype=object)
    start = 0

    for cat_idx, count in enumerate(counts_list):
        end = min(start + count, group_size)
        indices_to_assign = sorted_idx[start:end]
        if len(indices_to_assign) > 0:
            group_var_data[indices_to_assign] = categories[cat_idx]
        start = end

    if start < group_size:  # Handle leftovers if any
        group_var_data[sorted_idx[start:]] = categories[-1]

    return group_var_data


def _transform_binary(uniform_samples, original_data, group_size):
    """Transform uniform samples to binary data."""
    count_ones = np.sum(original_data == 1)
    sorted_idx = np.argsort(uniform_samples)

    group_var_data = np.zeros(group_size, dtype=int)
    indices_for_ones = sorted_idx[group_size - count_ones :]

    if len(indices_for_ones) > 0:
        group_var_data[indices_for_ones] = 1

    return group_var_data
