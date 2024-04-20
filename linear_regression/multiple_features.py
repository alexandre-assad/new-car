from typing import Tuple
from numpy import empty, int64, sum, mean, dot
from numpy.typing import NDArray


def calc_slopes(
    dependent_variables: NDArray[int64],
    dependent_mean: float,
    independent_variables: NDArray[int64],
    independent_means: NDArray[int64],
) -> NDArray[int64]:
    """
    Calculate the slopes of the regression line for each independent variable.

    Parameters:
    dependent_variables (NDArray[int64]): The dependent variable (y)
    dependent_mean (float): The mean of the dependent variable
    independent_variables (NDArray[int64]): The independent variables (x)
    independent_means (NDArray[int64]): The means of the independent variables

    Returns:
    NDArray[int64]: The slopes of the regression line for each independent variable
    """
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
    """
    Calculate the intercept of the regression line.

    Parameters:
    slopes (NDArray[int64]): The slopes of the regression line for each independent variable
    dependent_mean (float): The mean of the dependent variable
    independent_means (NDArray[int64]): The means of the independent variables

    Returns:
    float: The intercept of the regression line
    """
    return dependent_mean - sum(slopes * independent_means)


def estimate_linear_relationship(
    independent_variables: NDArray[int64], dependent_variables: NDArray[int64]
) -> Tuple[float, NDArray[int64]]:
    """
    Estimate the coefficients of the regression line.

    Parameters:
    independent_variables (NDArray[int64]): The independent variables (x)
    dependent_variables (NDArray[int64]): The dependent variable (y)

    Returns:
    Tuple[float, NDArray[int64]]: The intercept and slopes of the regression line
    """
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
    """
    Estimate the coefficients of the regression line for multiple independent variables.

    Parameters:
    independent_variables (NDArray[int64]): The independent variables (x1, x2, ..., xn)
    dependent_variables (NDArray[int64]): The dependent variable (y)

    Returns:
    Tuple[float, NDArray[int64]]: The intercept and slopes of the regression line. The slopes are an array where each element corresponds to the slope for each independent variable.
    """
    intercept, slopes = coefficients
    predicted_values = dot(independent_variables, slopes) + intercept
    return predicted_values
