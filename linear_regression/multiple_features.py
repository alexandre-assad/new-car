from typing import Tuple
from numpy import empty, int64, sum, mean, dot
from numpy.typing import NDArray


def calc_slopes(
    dependent_variables: NDArray[int64],
    dependent_mean: float,
    independent_variables: NDArray[int64],
    independent_means: NDArray[int64],
) -> NDArray[int64]:

    independent_variable_size = independent_variables.shape[1]
    slopes = empty(independent_variable_size)
    for i in range(independent_variable_size):
        cross_deviation = (
            sum(dependent_variables * independent_variables[:, i])
            - independent_variables.shape[0] * dependent_mean * independent_means[i]
        )
        deviation_about_independent = (
            sum(independent_variables[:, i] * independent_variables[:, i])
            - independent_variables.shape[0]
            * independent_means[i]
            * independent_means[i]
        )
        slopes[i] = cross_deviation / deviation_about_independent
    return slopes


def calc_intercept(
    slopes: NDArray[int64], dependent_mean: float, independent_means: NDArray[int64]
) -> float:
    return dependent_mean - sum(slopes * independent_means)


def estimate_linear_relationship(
    independent_variables: NDArray[int64], dependent_variables: NDArray[int64]
) -> Tuple[float, NDArray[int64]]:
    independent_means = mean(independent_variables, axis=0)
    dependent_mean = mean(dependent_variables)
    slopes = calc_slopes(
        dependent_variables, dependent_mean, independent_variables, independent_means
    )
    intercept = calc_intercept(slopes, dependent_mean, independent_means)
    return intercept, slopes


def predict_linear_relationship(
    independent_variables: NDArray[int64], coefficients: Tuple[float, NDArray[int64]]
) -> NDArray[int64]:
    intercept, slopes = coefficients
    predicted_values = dot(independent_variables, slopes) + intercept
    return predicted_values
