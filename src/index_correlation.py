import numpy as np


def calculate_implied_correlation(index_volatility, stock_volatilities, weights):
    num_stocks = len(stock_volatilities)
    implied_correlation_numerator = index_volatility**2 - np.sum(
        weights**2 * stock_volatilities**2
    )

    covariance_sum = 0.0
    for i in range(num_stocks):
        for j in range(num_stocks):
            covariance_sum += (
                weights[i] * weights[j] * stock_volatilities[i] * stock_volatilities[j]
            )

    if covariance_sum != 0:
        implied_correlation = implied_correlation_numerator / (2 * covariance_sum)
    else:
        implied_correlation = 0.0  # Handle the case when denominator is zero

    return implied_correlation


if __name__ == "__main__":
    # Example data
    index_volatility = 0.09295784803270265
    stock_volatilities = np.array([0.1, 0.12, 0.15, 0.08, 0.09])
    weights = np.array([0.2, 0.3, 0.15, 0.2, 0.15])

    # Calculate implied correlation
    implied_correlation = calculate_implied_correlation(
        index_volatility, stock_volatilities, weights
    )
    print("Implied Correlation:", implied_correlation)
