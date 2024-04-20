# NEW CAR

This project is a simple demonstration of how to use linear regression to predict car prices based on various features. The project is structured as follows:

- [`charts/`](charts/): Contains Jupyter notebooks for generating various charts.
- [`data/`](data/): Contains the raw data used in the project.
- [`linear_regression/`](linear_regression/): Contains Python scripts for running linear regression.
- [`notebooks/`](notebooks/): Contains Jupyter notebooks for exploratory data analysis and feature selection.

## How to Run

1. Install the required Python packages:

```sh
poetry install
```

2. Run the Jupyter notebooks in the [`notebooks/`](notebooks/) directory:

```sh   
jupyter notebook notebooks/exploratory-analysis.ipynb
jupyter notebook notebooks/feature_selection.ipynb
```

3. Run the Python scripts in the [`linear_regression/`](linear_regression/) directory:

```sh
python linear_regression/multiple_features.py
python linear_regression/single_feature.py
```

## How the Regression Works

The project uses linear regression, a statistical method that attempts to model the relationship between a dependent variable (in this case, the selling price of a car) and one or more independent variables (the features of the car, such as its year, transmission type, and fuel type).

The regression model is trained on a subset of the data (the training set), and its performance is evaluated on a different subset (the test set). The performance is measured using three metrics: mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE).

The project includes both univariate linear regression (which uses a single feature to predict the selling price) and multivariate linear regression (which uses multiple features).

## More Pretty Than Full!

This project is a great introduction to linear regression and data analysis with Python. It's more about understanding the concepts and getting a feel for the data than about building a complex, full-featured application. So dive in, explore the data, and have fun learning about linear regression!