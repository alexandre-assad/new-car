from typing import Tuple
from numpy import float64, int64, size, mean
from numpy.typing import NDArray


def _calc_slope(
    dependent_variables: NDArray[int64],
    dependent_mean: float,
    independent_variables: NDArray[int64],
    independent_mean: float,
) -> float:
    """
    Calculate the slope of the regression line.

    Parameters:
    dependent_variables (NDArray[int64]): The dependent variable (y)
    dependent_mean (float): The mean of the dependent variable
    independent_variables (NDArray[int64]): The independent variable (x)
    independent_mean (float): The mean of the independent variable

    Returns:
    float: The slope of the regression line
    """
    independent_variable_size = size(independent_variables)

    cross_deviation = (
        sum(dependent_variables * independent_variables)
        - independent_variable_size * dependent_mean * independent_mean
    )
    independent_variation = (
        sum(independent_variables * independent_variables)
        - independent_variable_size * independent_mean * independent_mean
    )

    slope = cross_deviation / independent_variation
    return slope


def _calc_intercept(
    slope: float, dependent_mean: float, independent_mean: float
) -> float:
    """
    Calculate the intercept of the regression line.

    Parameters:
    slope (float): The slope of the regression line
    dependent_mean (float): The mean of the dependent variable
    independent_mean (float): The mean of the independent variable

    Returns:
    float: The intercept of the regression line
    """
    return dependent_mean - slope * independent_mean


def estimate_coefs(
    independent_variables: NDArray[int64], dependent_variables: NDArray[int64]
) -> Tuple[float, float]:
    """
    Estimate the coefficients of the regression line.

    Parameters:
    independent_variables (NDArray[int64]): The independent variable (x)
    dependent_variables (NDArray[int64]): The dependent variable (y)

    Returns:
    Tuple[float, float]: The intercept and slope of the regression line
    """
    independent_mean = mean(independent_variables)
    dependent_mean = mean(dependent_variables)

    slope = _calc_slope(
        dependent_variables, dependent_mean, independent_variables, independent_mean
    )
    intercept = _calc_intercept(slope, dependent_mean, independent_mean)

    return intercept, slope


def predict_linear_relationship(
    independent_variables: NDArray[int64], coefs: Tuple[float, float]
) -> NDArray[float64]:
    """
    Estimate the coefficients of the regression line.

    Parameters:
    independent_variables (NDArray[int64]): The independent variable (x)
    dependent_variables (NDArray[int64]): The dependent variable (y)

    Returns:
    Tuple[float, float]: The intercept and slope of the regression line
    """
    intercept, slope = coefs
    predicted_values = slope * independent_variables + intercept
    return predicted_values
