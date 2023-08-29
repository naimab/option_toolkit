import yfinance as yf
import pandas as pd
import numpy as np
import cachetools

# Create a cache that expires items after 24 hours
cache = cachetools.TTLCache(maxsize=100, ttl=86400)  # 86400 seconds = 24 hours

# Set's display formatting to 3 decimals places
pd.options.display.float_format = "{:,.3f}".format


def get_cache_key(tkr, period, interval):
    """Generate a unique cache key based on function parameters."""
    return f"{tkr}_{period}_{interval}"


# Set Variables
trading_days = 252


def increment_date_string(date_str):
    if date_str[-1] in ["d"]:
        num = int(date_str[:-1]) + 2
        return f"{num}{date_str[-1]}"
    else:
        num = int(date_str[:-1])
        return f"{num}{date_str[-1]}"


def convert_window_to_days(window_str):
    window_str = str(window_str)
    conversion = {
        "d": 1,
        "w": 5,  # Assuming 5 trading days in a week
        "m": 20,  # Rough average of trading days in a month
        "y": 252,  # Rough average of trading days in a year
    }
    # Get the last character of the window string to determine the unit (d, w, m, y)
    unit = window_str[-1]
    # Get the numeric part of the window string
    number = int(window_str[:-1])
    return number * conversion[unit]


def get_historical_prices(tkr, period, interval="1d"):
    """
    Gets historical prices (OHLC) for an instrument yfinance and caches the result.
    tkr(string): Ticker for instrument.
    days(string): Currently must use "#d" for time period for analysis.
    interval (string): Optional interval parameter. '#m", "#h", "d"

    returns: Pandas dataframe
    """
    cache_key = get_cache_key(tkr, period, interval)

    period = increment_date_string(period)

    # Check if value is in cache
    if cache_key in cache:
        return cache[cache_key]

    try:
        # fetch historical prices
        ticker = yf.Ticker(tkr)
        price_data = ticker.history(period=period, interval=interval)

        # Store the result in the cache
        cache[cache_key] = price_data

        return price_data
    except Exception as e:
        print(f"Error fetching prices from yahoo finance: {e}")
        return None


def close_to_close_volatility(tkr, window, period="252d"):
    """Computes the Close to Close volatility estimator."""
    df = get_historical_prices(tkr, period)

    # drop today's data:
    df = df.iloc[:-1].copy()
    window_size = convert_window_to_days(window)

    # calculate log returns
    df["Log Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    col_name = f"{window_size}d_vol"
    df[col_name] = df["Log Returns"].rolling(window=window_size, min_periods=window_size).std()
    # return df[col_name].iloc[-1] * np.sqrt(trading_days)
    df[col_name] = df[col_name] * np.sqrt(trading_days)
    return df


def garman_klass(tkr, window, period="252d"):
    """Computes the Garman-Klass volatility estimator."""
    df = get_historical_prices(tkr, period)
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])

    # Garman - Klass formula
    sigma2 = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)

    window_size = convert_window_to_days(window)
    col_name = f"{window}_vol"

    # Rolling computation of Garman-Klass estimator and annualization
    df[col_name] = sigma2.rolling(window=window_size, min_periods=window_size).mean()
    df[col_name] = np.sqrt(df[col_name] * trading_days)
    return df


def historical_volatility(tkr, window, period="252d", method="close_to_close"):
    if method == "close_to_close":
        return close_to_close_volatility(tkr, window=window, period=period)
    elif method == "garman_klass":
        return garman_klass(tkr, window, period)
    else:
        raise ValueError(f"Unknown method {method}")


def volatility_cones_table(tkr, period="504d", method="close_to_close"):
    """
    Returns dataframe with at least 2 years of daily data (Open, High, Low, Close)
    Parameters: tkr (string), period(string), method(string)

    """
    df = get_historical_prices(tkr, period)

    # Calculate log returns for the entire period
    df["Log Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    # Set the windows you will compare
    windows = ["10d", "20d", "40d", "60d", "120d"]

    # Compute volatilities for each window and store in the dataframe
    for window in windows:
        vol_col_name = f"{window}_vol"
        df[vol_col_name] = historical_volatility(tkr, window, period, method=method)[vol_col_name]

    # Define the percentiles you want to compute
    percentiles = {
        "Maximum": np.max,
        "90% percentile": lambda x: np.percentile(x, 90),
        "75% percentile": lambda x: np.percentile(x, 75),
        "Median": np.median,
        "25% percentile": lambda x: np.percentile(x, 25),
        "10% percentile": lambda x: np.percentile(x, 10),
        "Minimum": np.min,
    }

    # Create the final table
    result = pd.DataFrame(index=percentiles.keys(), columns=[f"{window}_vol" for window in windows])

    for window in windows:
        vol_col_name = f"{window}_vol"
        for percentile_name, percentile_func in percentiles.items():
            result.at[percentile_name, f"{window}_vol"] = percentile_func(df[vol_col_name].dropna())
    print(method)
    return result


def historical_vol_snapshot(tkr):
    """
    Returns dataframe with most recent 10d, 20d, 60d vols for instrument
    Parameter: tkr(string)
    """
    # Creates a dictionary with keys as periods and values as computed volatilities
    data = {
        "10d": close_to_close_volatility(tkr, "10d")["10d_vol"].iloc[-1],
        "20d": close_to_close_volatility(tkr, "20d")["20d_vol"].iloc[-1],
        "60d": close_to_close_volatility(tkr, "60d")["60d_vol"].iloc[-1],
    }

    # Converts the dictionary into a DataFrame
    df = pd.DataFrame([data])

    # Names the row as 'Volatility'
    df.index = ["Volatility"]

    print("Today's vol snapshot")
    return df


if __name__ == "__main__":
    print(volatility_cones_table("aapl", method="close_to_close"))
    # historical_vol_snapshot("aapl")
