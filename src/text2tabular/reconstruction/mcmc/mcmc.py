import numpy as np
import math
import random
from text2tabular.reconstruction.mcmc.mcmc_utils import (
    calculate_target_stats_and_corrs_and_tests,
    calculate_stats_and_corrs_and_tests,
    calculate_difference,
)


def mcmc_refinement(
    synthetic_data,
    variables,
    groups,
    correlations,
    tests,
    max_iterations=1000,
    tolerance=0.01,
    seed=42,
):
    """
    Refine synthetic data using MCMC to match target statistics and correlations.

    Args:
        synthetic_data: Dictionary of initial synthetic data
        variables: Dictionary containing variable specifications
        correlations: Dictionary specifying correlations between variables
        groups: List of group names
        max_iterations: Maximum number of MCMC iterations
        tolerance: Convergence tolerance

    Returns:
        Dictionary of refined synthetic data
    """

    def get_temperature(
        initial, final, iteration, max_iter, decay="quadratic"
    ):
        progress = iteration / max_iter

        if decay == "linear":
            return initial - (initial - final) * progress
        elif decay == "quadratic":
            return initial - (initial - final) * (progress**2)
        elif decay == "exponential":
            return initial * (final / initial) ** progress
        elif decay == "reciprocal":
            return initial / (1 + progress * (initial / final - 1))

    initial_temp = 1
    final_temp = 0.01
    step_size_base = 0.05
    temp_decay = "quadratic"
    # best_diff = np.inf

    # set seed
    np.random.seed(seed)

    # Calculate target statistics and correlations
    target_stats, target_corrs, target_tests = (
        calculate_target_stats_and_corrs_and_tests(
            variables, correlations, tests
        )
    )

    # Initialize current data
    current_data = {k: v.copy() for k, v in synthetic_data.items()}

    # MCMC iterations
    for iteration in range(max_iterations):
        temp = get_temperature(
            initial_temp,
            final_temp,
            iteration,
            max_iterations,
            decay=temp_decay,
        )
        # Calculate adaptive step size
        step_size = step_size_base * temp

        # Calculate current statistics and correlations
        current_stats, current_corrs, current_tests = (
            calculate_stats_and_corrs_and_tests(
                current_data, target_stats, target_corrs, groups, target_tests
            )
        )

        # Calculate current difference from targets
        total_diff = calculate_difference(
            current_stats,
            current_corrs,
            target_stats,
            target_corrs,
            target_tests,
            current_tests,
        )

        # Check if we've reached desired accuracy
        if total_diff < tolerance:
            print(f"MCMC converged after {iteration} iterations")
            break

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, difference: {total_diff:.4f}")

        # Propose a change
        var, group = random.choice(list(current_data.keys()))

        # Make a copy of the current data
        proposed_data = {k: v.copy() for k, v in current_data.items()}

        # Modify the proposed data
        if (var, group) in target_stats:

            idx1, idx2 = np.random.choice(
                len(current_data[(var, group)]), 2, replace=False
            )

            if "mean" in target_stats[(var, group)]:
                round_values = False
                if var in variables["ordinal"]:
                    if group in variables["ordinal"][var]:
                        round_values = True
                # For continuous and ordinal, swap values
                if iteration % 2 == 0:
                    (
                        proposed_data[(var, group)][idx1],
                        proposed_data[(var, group)][idx2],
                    ) = (
                        proposed_data[(var, group)][idx2],
                        proposed_data[(var, group)][idx1],
                    )
                else:
                    # Adjust the value to move closer to the target mean
                    mean = (
                        target_stats[(var, group)]["mean"]
                        if "mean" in target_stats[(var, group)]
                        else None
                    )
                    std = (
                        target_stats[(var, group)]["std"]
                        if "std" in target_stats[(var, group)]
                        else None
                    )
                    if mean is None:
                        continue  # Skip if mean is not defined
                    if std is None:
                        std = 1.0  # Default to 1 if std is not provided
                    adjustment = np.random.normal(
                        loc=(mean - np.mean(current_data[(var, group)]))
                        * 0.1,  # Move 10% closer to the target mean
                        scale=step_size
                        * std,  # Scale perturbation by target std
                    )
                    if round_values:
                        if adjustment > 0:
                            adjustment = math.ceil(
                                adjustment
                            )  # Use ceil for positive adjustments
                        else:
                            adjustment = math.floor(
                                adjustment
                            )  # Use floor for negative adjustments
                    proposed_data[(var, group)][idx1] += adjustment

            # Modified block to handle median, Q1, Q3, and use IQR for scaling
            elif "median" in target_stats[(var, group)]:  # Changed to elif
                round_values = False
                # Safer check for ordinal variable and group
                if var in variables.get(
                    "ordinal", {}
                ) and group in variables.get("ordinal", {}).get(var, {}):
                    round_values = True

                # For continuous and ordinal, swap values on even iterations
                if iteration % 2 == 0:
                    (
                        proposed_data[(var, group)][idx1],
                        proposed_data[(var, group)][idx2],
                    ) = (
                        proposed_data[(var, group)][idx2],
                        proposed_data[(var, group)][idx1],
                    )
                else:
                    # Adjust the value to move closer to the target median, scaled by IQR
                    target_var_stats = target_stats[(var, group)]
                    current_vals_for_stat = current_data[(var, group)]

                    target_median = target_var_stats["median"]
                    current_median = np.median(current_vals_for_stat)

                    diff_to_target_median = target_median - current_median
                    adjustment_loc = (
                        diff_to_target_median * 0.1
                    )  # Move 10% closer to target median

                    # Determine scale_basis using IQR (prefer target, then calculate from Q1/Q3, then current)
                    scale_basis = 1.0  # Default fallback
                    if (
                        "iqr" in target_var_stats
                        and target_var_stats["iqr"] > 0
                    ):
                        scale_basis = target_var_stats["iqr"]
                    elif "q1" in target_var_stats and "q3" in target_var_stats:
                        target_q1 = target_var_stats["q1"]
                        target_q3 = target_var_stats["q3"]
                        if (
                            target_q3 > target_q1
                        ):  # Ensure Q3 > Q1 for a valid IQR
                            scale_basis = target_q3 - target_q1

                    if (
                        scale_basis == 1.0
                    ):  # If target IQR wasn't found or valid
                        current_q1 = np.percentile(current_vals_for_stat, 25)
                        current_q3 = np.percentile(current_vals_for_stat, 75)
                        current_iqr_val = current_q3 - current_q1
                        if current_iqr_val > 0:
                            scale_basis = current_iqr_val
                        else:  # Fallback to STD if IQR is zero or not useful
                            current_std_val = np.std(current_vals_for_stat)
                            if current_std_val > 0:
                                scale_basis = current_std_val
                            # If all spread measures are zero/unavailable, scale_basis remains 1.0

                    adjustment_scale = step_size * scale_basis
                    adjustment = np.random.normal(
                        loc=adjustment_loc, scale=max(adjustment_scale, 1e-6)
                    )

                    if round_values:
                        adjustment = (
                            math.ceil(adjustment)
                            if adjustment > 0
                            else math.floor(adjustment)
                        )

                    current_value_to_perturb = proposed_data[(var, group)][
                        idx1
                    ]
                    perturbed_value = current_value_to_perturb + adjustment

                    # Clamp the perturbed value using percentiles
                    data_for_bounds = current_data[(var, group)]
                    if (
                        len(data_for_bounds) > 1
                    ):  # Percentiles need at least 2 data points
                        lower_bound = np.percentile(data_for_bounds, 1)
                        upper_bound = np.percentile(data_for_bounds, 99)
                        perturbed_value = np.clip(
                            perturbed_value, lower_bound, upper_bound
                        )
                    elif (
                        len(data_for_bounds) == 1
                    ):  # Fallback for single data point
                        perturbed_value = np.clip(
                            perturbed_value,
                            data_for_bounds[0],
                            data_for_bounds[0],
                        )

                    if (
                        round_values
                    ):  # Ensure value remains ordinal if specified
                        perturbed_value = round(perturbed_value)

                    proposed_data[(var, group)][idx1] = perturbed_value

            elif "categories" in target_stats[(var, group)]:
                # For categorical and binary, swap only if values are different
                if (
                    proposed_data[(var, group)][idx1]
                    == proposed_data[(var, group)][idx2]
                ):
                    (
                        proposed_data[(var, group)][idx1],
                        proposed_data[(var, group)][idx2],
                    ) = (
                        proposed_data[(var, group)][idx2],
                        proposed_data[(var, group)][idx1],
                    )
            elif "count" in target_stats[(var, group)]:
                (
                    proposed_data[(var, group)][idx1],
                    proposed_data[(var, group)][idx2],
                ) = (
                    proposed_data[(var, group)][idx2],
                    proposed_data[(var, group)][idx1],
                )

        # Calculate proposed statistics and correlations
        # Calculate current statistics and correlations
        proposed_stats, proposed_corrs, proposed_tests = (
            calculate_stats_and_corrs_and_tests(
                proposed_data, target_stats, target_corrs, groups, target_tests
            )
        )

        # Calculate proposed difference
        proposed_diff = calculate_difference(
            proposed_stats,
            proposed_corrs,
            target_stats,
            target_corrs,
            target_tests,
            proposed_tests,
        )

        # Accept or reject the proposal
        if proposed_diff < total_diff:
            current_data = proposed_data
        else:
            if iteration % 2 != 0:
                acceptance_prob = np.exp(
                    -10 * (proposed_diff - total_diff) / max(temp, 1e-10)
                )
                if np.random.random() < acceptance_prob:
                    current_data = proposed_data

    return current_data
