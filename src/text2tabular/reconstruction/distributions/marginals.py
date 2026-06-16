import numpy as np
from scipy.stats import skewnorm, gamma
from scipy import optimize
import warnings


def generate_distributions(variables, group_sizes):
    """
    Generates all types of distributions (categorical, binary, continuous, ordinal)
    based on the provided configuration and group sizes.
    """
    categorical_data = generate_categorical_distributions(
        variables, group_sizes
    )
    binary_data = generate_binary_distributions(variables, group_sizes)
    numeric_data_combined = generate_numeric_distributions(
        variables, group_sizes
    )

    continuous_data = {}
    ordinal_data = {}

    for key, data_array in numeric_data_combined.items():
        var_type, variable_name, group_name = key
        if var_type == "continuous":
            continuous_data[(variable_name, group_name)] = data_array
        elif var_type == "ordinal":
            ordinal_data[(variable_name, group_name)] = data_array

    return categorical_data, binary_data, continuous_data, ordinal_data


def generate_categorical_distributions(variables, group_sizes):
    """
    Generate categorical distributions based on specified categories and counts.

    Args:
        variables: Dictionary containing variable specifications
        group_sizes: Dictionary mapping groups to their sizes

    Returns:
        Dictionary mapping (variable, group) tuples to arrays of categorical values
    """
    categorical_data = {}
    if "categorical" in variables:
        for variable, groups in variables["categorical"].items():
            for group, stats in groups.items():
                try:
                    # stats is a dict: {category: count, ...}
                    counts = list(stats.values())
                    size = group_sizes[group]
                    if sum(counts) == size:
                        samples = []
                        for category, count in stats.items():
                            samples.extend([category] * count)
                        np.random.shuffle(samples)
                        categorical_data[(variable, group)] = np.array(samples)

                except KeyError:
                    warnings.warn(
                        f"Counts for {variable} in {group} must sum to group size {size}"
                    )
                    continue
    return categorical_data


def generate_binary_distributions(variables, group_sizes):
    """
    Generate binary distributions based on specified counts.
    Args:
        variables: Dictionary containing variable specifications
        group_sizes: Dictionary mapping groups to their sizes

    Returns:
        Dictionary mapping (variable, group) tuples to arrays of binary values (0/1)
    """
    binary_data = {}
    if "binary" in variables:
        for variable, groups in variables["binary"].items():
            for (
                group,
                count,
            ) in groups.items():  # count is now an int, not a dict
                try:
                    if isinstance(count, int):

                        size = group_sizes[group]
                        samples = np.zeros(size)
                        samples[:count] = 1
                        np.random.shuffle(samples)
                        binary_data[(variable, group)] = samples
                except KeyError:
                    warnings.warn(
                        f"Group '{group}' not found in group_sizes. Skipping binary generation for {variable}."
                    )
                    continue
    return binary_data


def generate_numeric_distributions(variables, group_sizes):
    """Generates numeric distributions (ordinal/continuous) based on variable stats.

    Args:
        variables (dict): Variable specifications, e.g., {"ordinal": {...}, "continuous": {...}}.
        group_sizes (dict): Mapping of groups to sizes.

    Returns:
        dict: Maps (variable_type, variable_name, group_name) to generated value arrays.
    """

    all_generated_data = {}
    variable_types = ["ordinal", "continuous"]

    for var_type in variable_types:
        if var_type not in variables:
            continue

        for variable_name, groups in variables[var_type].items():

            for group_name, stats in groups.items():
                if group_name not in group_sizes:
                    continue

                size = group_sizes[group_name]
                if size <= 0:
                    all_generated_data[
                        (var_type, variable_name, group_name)
                    ] = np.array([])
                    continue

                raw_samples = generate_samples(
                    var_type, variable_name, group_name, stats, size
                )

                final_samples = process_samples(var_type, raw_samples)

                all_generated_data[(var_type, variable_name, group_name)] = (
                    final_samples
                )

    return all_generated_data


def generate_samples(var_type, variable_name, group_name, stats, size):
    """Generates raw samples based on provided statistics."""
    if (
        "mean" in stats
        and "std" in stats
        and stats["std"] is not None
        and stats["mean"] is not None
    ):
        mean = stats["mean"]
        std = stats["std"]
        skew_param = stats.get("skew", 0)
        min_v = stats.get("min_val", None)
        max_v = stats.get("max_val", None)
        tail_w = stats.get("tail_weight", 0)
        tail_sf = stats.get("tail_std_factor", 0)

        if std <= 0:
            warnings.warn(
                f"Std is {std} for '{var_type}' variable '{variable_name}', group '{group_name}'. Skipping."
            )
            return np.array([])
        else:
            return generate_bounded_skewnorm(
                mean=mean,
                std=std,
                size=size,
                skew=skew_param,
                min_val=min_v,
                max_val=max_v,
                tail_weight=tail_w,
                tail_std_factor=tail_sf,
            )
    elif (
        "median" in stats
        and stats["median"] is not None
        and (
            "iqr" in stats
            and stats["iqr"] is not None
            or (
                "q1" in stats
                and stats["q1"] is not None
                and "q3" in stats
                and stats["q3"] is not None
            )
        )
    ):
        median_val = stats["median"]
        q1_val = stats.get("q1", None)
        q3_val = stats.get("q3", None)
        min_v = stats.get("min_val", None)
        max_v = stats.get("max_val", None)
        # if q1 and q3 are not provided, use iqr and parse with parse_iqr
        if q1_val is None or q3_val is None:
            iqr_val = stats.get("iqr", None)

            if iqr_val is not None and len(iqr_val) == 2:
                q1_val, q3_val = iqr_val
            elif isinstance(iqr_val, (float, int)):
                q1_val = median_val - iqr_val / 2.0
                q3_val = median_val + iqr_val / 2.0

        return generate_non_normal_sample(
            median=median_val,
            size=size,
            q1=q1_val,
            q3=q3_val,
            # iqr=iqr_val,
            min_val=min_v,
            max_val=max_v,
        )
    elif "ci95" in stats and stats["ci95"] is not None:
        ci = stats["ci95"]
        if isinstance(ci, (float, int)) and "median" in stats:
            lower_bound = ci - stats["median"]
            upper_bound = ci + stats["median"]
        elif "ci95_low" in stats and "ci95_high" in stats:
            lower_bound = stats["ci95_low"]
            upper_bound = stats["ci95_high"]

        else:
            return np.array([])

        mean_val = (lower_bound + upper_bound) / 2
        std_val = (upper_bound - lower_bound) / 3.14  # Approximate std from CI
        skew_param = stats.get("skew", 0)
        min_v = stats.get("min_val", None)
        max_v = stats.get("max_val", None)
        tail_w = stats.get("tail_weight", 0)
        tail_sf = stats.get("tail_std_factor", 0)
        if std_val <= 0:
            warnings.warn(
                f"Std derived from CI is {std_val} for '{var_type}' variable '{variable_name}', group '{group_name}'. Skipping."
            )
            return np.array([])
        else:
            return generate_bounded_skewnorm(
                mean=mean_val,
                std=std_val,
                size=size,
                skew=skew_param,
                min_val=min_v,
                max_val=max_v,
                tail_weight=tail_w,
                tail_std_factor=tail_sf,
            )

    elif "mean" or "median" in stats:
        if "mean" in stats and stats["mean"] is not None:
            mean_val = stats["mean"]
        elif "median" in stats and stats["median"] is not None:
            mean_val = stats["median"]
        else:
            warnings.warn(
                f"Neither mean nor median provided for '{var_type}' variable '{variable_name}', group '{group_name}'. Skipping."
            )
            return np.array([])

        skew_param = stats.get("skew", 0)
        min_v = stats.get("min_val", None)
        max_v = stats.get("max_val", None)
        tail_w = stats.get("tail_weight", 0)
        tail_sf = stats.get("tail_std_factor", 0)

        return generate_bounded_skewnorm(
            mean=mean_val,
            std=1.0,
            size=size,
            skew=skew_param,
            min_val=min_v,
            max_val=max_v,
            tail_weight=tail_w,
            tail_std_factor=tail_sf,
        )

    else:
        return np.array([])


def process_samples(var_type, raw_samples):
    """Processes raw samples based on variable type (ordinal/continuous)."""
    if raw_samples is not None and raw_samples.size > 0:
        if var_type == "ordinal":
            return np.round(raw_samples).astype(int)
        else:  # continuous
            return raw_samples
    elif raw_samples is not None:  # Empty array
        return np.array([], dtype=int if var_type == "ordinal" else float)
    else:
        return np.array([])


def generate_non_normal_sample(
    median, size, q1=None, q3=None, iqr=None, min_val=None, max_val=None
):
    """
    Generates a non-normal data sample given a median and either Q1 & Q3 or IQR,
    by fitting a gamma distribution. Values are bounded by min_val and max_val if provided.

    Parameters:
    median (float): The desired median of the sample (Q2).
    q1 (float, optional): The desired first quartile (25th percentile).
    q3 (float, optional): The desired third quartile (75th percentile).
    iqr (float, optional): The desired interquartile range (Q3 - Q1).
    size (int): The number of data points to generate.
    min_val (float, optional): Minimum allowed value for the distribution.
    max_val (float, optional): Maximum allowed value for the distribution.

    Returns:
    numpy.ndarray: A NumPy array containing the generated data sample.

    Raises:
    ValueError: If insufficient quartile information or inconsistent values are provided.
    RuntimeError: If fitting the gamma distribution fails.
    """

    target_q1, target_median, target_q3 = _calculate_target_quartiles(
        median, q1, q3, iqr
    )

    fitted_params = _fit_gamma_parameters(target_q1, target_median, target_q3)

    # Generate samples using a batch approach similar to generate_bounded_skewnorm
    samples_collected = []
    batch_size = 50
    # Determine batch sizes
    batch_size_base = int(size / batch_size) if size > 100 else size
    batch_size_base = max(
        batch_size_base, batch_size
    )  # Ensure a minimum batch size

    while len(samples_collected) < size:
        remaining_size = size - len(samples_collected)
        current_batch_size = min(
            batch_size_base * 2, remaining_size * 3
        )  # Generate more than needed
        current_batch_size = max(current_batch_size, batch_size)

        # Generate a batch of samples
        batch = gamma.rvs(
            fitted_params[0],
            loc=fitted_params[1],
            scale=fitted_params[2],
            size=current_batch_size,
        )

        # Apply bounds if specified
        if min_val is not None:
            batch = batch[batch >= min_val]
        if max_val is not None:
            batch = batch[batch <= max_val]

        # Add to collected samples
        samples_collected.extend(batch)

        if len(samples_collected) >= size:
            break

    # Convert to numpy array, shuffle, and return exactly 'size' samples
    final_samples = np.array(samples_collected)
    np.random.shuffle(final_samples)  # Shuffle samples
    return final_samples[:size]


def _calculate_target_quartiles(median, q1=None, q3=None, iqr=None):
    """Helper to determine Q1, median, and Q3 from inputs."""
    if q1 is not None and q3 is not None:
        if q1 > median:
            raise ValueError("Q1 cannot be greater than the median.")
        if q3 < median:
            raise ValueError("Q3 cannot be less than the median.")
        if q1 > q3:
            raise ValueError("Q1 cannot be greater than Q3.")
        return float(q1), float(median), float(q3)
    elif iqr is not None:
        if iqr < 0:
            raise ValueError("IQR cannot be negative.")
        current_q1 = median - (iqr / 2.0)
        current_q3 = median + (iqr / 2.0)
        if current_q1 > median:  # Should not happen if iqr >= 0
            raise ValueError(
                "Calculated Q1 is greater than median. Check inputs."
            )
        if current_q3 < median:  # Should not happen if iqr >= 0
            raise ValueError(
                "Calculated Q3 is less than median. Check inputs."
            )
        return current_q1, float(median), current_q3
    else:
        raise ValueError("Either (q1 and q3) or iqr must be provided.")


def _fit_gamma_parameters(target_q1, target_median, target_q3):
    """
    Fits gamma distribution parameters (shape, loc, scale) to match
    the target Q1, median, and Q3.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            def objective_function(params):
                a, loc, scale = params
                if a <= 0 or scale <= 0:  # Shape and scale must be positive
                    return np.inf  # Using np.inf for invalid parameters
                # Calculate squared differences from target quantiles
                err_q1 = (
                    gamma.ppf(0.25, a, loc=loc, scale=scale) - target_q1
                ) ** 2
                err_median = (
                    gamma.ppf(0.50, a, loc=loc, scale=scale) - target_median
                ) ** 2
                err_q3 = (
                    gamma.ppf(0.75, a, loc=loc, scale=scale) - target_q3
                ) ** 2
                return err_q1 + err_median + err_q3

            # Initial guess for parameters
            initial_scale_guess = (
                abs(target_q3 - target_q1)
                if abs(target_q3 - target_q1) > 1e-6
                else 1.0
            )
            initial_loc_guess = target_q1 - initial_scale_guess
            if target_q1 < 0 and initial_loc_guess > 0:
                initial_loc_guess = target_q1 - 1.0

            initial_params = [
                2.0,
                initial_loc_guess,
                initial_scale_guess,
            ]  # shape, loc, scale

            result = optimize.minimize(
                objective_function,
                initial_params,
                method="Nelder-Mead",
                options={"maxiter": 2500, "adaptive": True},
            )

            # if not result.success:
            #     warnings.warn(f"Optimization for gamma parameters failed to converge: {result.message}")

            fitted_params = result.x
            # Ensure shape and scale are positive after optimization
            if fitted_params[0] <= 0:
                fitted_params[0] = 1e-6  # Min positive shape
            if fitted_params[2] <= 0:
                fitted_params[2] = 1e-6  # Min positive scale

            return fitted_params

    except Exception as e:
        # Catching a broad exception here as optimize.minimize can raise various things
        # or the ppf function might fail under extreme parameter guesses.
        raise RuntimeError(
            f"Failed to fit gamma distribution parameters. Error: {e}"
        )


def generate_bounded_skewnorm(
    mean,
    std,
    size,
    skew=0,
    min_val=None,
    max_val=None,
    tail_weight=0,
    tail_std_factor=0,
):
    """
    Generate samples from a (possibly skewed) normal distribution,
    with optional heavier or lighter tails, and bounded by min_val and max_val.

    """
    if std <= 0:
        raise ValueError("Standard deviation (std) must be positive.")

    samples_collected = []

    # Determine sizes for main and tail distributions for each batch attempt
    batch_size_base = (
        int(size / 10) if size > 100 else size
    )  # Generate in batches
    batch_size_base = max(batch_size_base, 10)  # Ensure a minimum batch size

    while len(samples_collected) < size:
        remaining_size = size - len(samples_collected)
        current_batch_attempt_size = min(
            batch_size_base * 2, remaining_size * 3
        )  # Generate more than needed
        current_batch_attempt_size = max(current_batch_attempt_size, 10)

        if tail_weight <= 0 or tail_weight >= 1:
            main_target_this_batch = current_batch_attempt_size
            tail_target_this_batch = 0
        else:
            tail_target_this_batch = int(
                current_batch_attempt_size * tail_weight
            )
            main_target_this_batch = (
                current_batch_attempt_size - tail_target_this_batch
            )

        current_batch_samples = []

        # Generate main samples
        if main_target_this_batch > 0:
            batch_main = skewnorm.rvs(
                a=skew, loc=mean, scale=std, size=main_target_this_batch
            )
            if min_val is not None:
                batch_main = batch_main[batch_main >= min_val]
            if max_val is not None:
                batch_main = batch_main[batch_main <= max_val]
            current_batch_samples.extend(batch_main)

        # Generate tail samples
        if tail_target_this_batch > 0:
            current_tail_std = std * tail_std_factor
            if current_tail_std <= 0:
                current_tail_std = std  # fallback
            batch_tail = skewnorm.rvs(
                a=skew,
                loc=mean,
                scale=current_tail_std,
                size=tail_target_this_batch,
            )
            if min_val is not None:
                batch_tail = batch_tail[batch_tail >= min_val]
            if max_val is not None:
                batch_tail = batch_tail[batch_tail <= max_val]
            current_batch_samples.extend(batch_tail)

        samples_collected.extend(current_batch_samples)

        if len(samples_collected) >= size:
            break

    final_samples = np.array(samples_collected)
    np.random.shuffle(
        final_samples
    )  # Shuffle if samples from different distributions were mixed
    return final_samples[:size]
