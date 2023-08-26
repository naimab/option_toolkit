from datetime import datetime, timedelta
from pandas.tseries.holiday import (
    USFederalHolidayCalendar,
    nearest_workday,
    AbstractHolidayCalendar,
    Holiday,
    USThanksgivingDay,
    USMartinLutherKingJr,
    USPresidentsDay,
    GoodFriday,
    USMemorialDay,
    USLaborDay,
)
from pandas.tseries.offsets import DateOffset, BDay
import pandas as pd
import QuantLib as ql
import yfinance as yf
import numpy as np
import cachetools
from fredapi import Fred


# Replace YOUR_API_KEY with your actual FRED API key
api_key = "cb0fa660da1ffc1743a352f26c28c387"


# Create a cache that expires items after 24 hours
cache = cachetools.TTLCache(maxsize=100, ttl=86400)  # 86400 seconds = 24 hours


def get_ff_rate():
    """
    Returns fed funds rate from FredApi. Returns float.
    """
    # I use a fixed key for this cache since there are no arguments that vary
    cache_key = "ff_rate"

    # Check if value is in cache
    if cache_key in cache:
        return cache[cache_key]

    try:
        # Replace YOUR_API_KEY with your actual FRED API key
        fred = Fred(api_key=api_key)

        # Fetch the Federal Funds rate series
        fed_funds_series = fred.get_series("FEDFUNDS")

        # Get the most recent rate
        current_fed_funds_rate = fed_funds_series.iloc[-1]

        rate = current_fed_funds_rate / 100

        # Store the result in the cache
        cache[cache_key] = rate

        return rate
    except Exception as e:
        print(f"Error fetching Federal Funds rate from FRED: {e}")
        return None


def get_ticker_object(tkr):
    """
    Returns a yf.Ticker object for a given ticker.
    """
    try:
        return yf.Ticker(tkr)
    except Exception as e:
        print(f"Error fetching data for {tkr}: {e}")
        return None


def div_cal(tkr):
    """
    Dividend count estimates with date index. Ticke is string. Returns dataframe.
    """

    cache_key = f"div_{tkr}"

    # check if value is in cache
    if cache_key in cache:
        return cache[cache_key]

    # grabs object from yfinance
    tkr = get_ticker_object(tkr)
    df = tkr.dividends.tail()
    # Extract the last 4 dates and their values
    last_4_dates = df.index[-4:]
    last_4_values = df.values[-4:]
    # Update the dates by adding 12 months
    next_4_dates = last_4_dates + DateOffset(months=12) - BDay(1)
    # for date in next_4_dates: if date after 10th subtract a day
    # Create new DataFrame
    new_df = pd.DataFrame({"Dividends": last_4_values}, index=next_4_dates)
    new_df.reset_index(inplace=True)
    new_df.rename(columns={"index": "date"}, inplace=True)
    # convert to regular date column and now have dataframe to work with and pass
    new_df["Date"] = new_df["Date"].dt.date

    # store the result in the cache
    cache[cache_key] = new_df

    return new_df


def days_until_expiration(expiration_date):
    """
    Calculates DTE. Takes datetime object.
    """
    today = datetime.today().date()
    delta = expiration_date - today
    return delta.days


def get_current_price(tkr):
    """
    Get's current price and returns float. Ticker is string.
    """

    stock = get_ticker_object(tkr)
    price = stock.history(period="1m")["Close"].iloc[-1]
    return price


def forward(tkr, expiration):
    """
    Calculates forward price of instrument and returns float. Ticker is string and expiration is datatime object.
    """
    price = get_current_price(tkr)
    rate = get_ff_rate()
    dte = days_until_expiration(expiration)
    forward = price * np.exp(rate * dte / 365)
    # count days til dividend and if days til div is less than or qual to dte include and present value
    df = div_cal("tkr")
    df["dtd"] = df["Date"] - datetime.today().date()
    filtered_df = df[df["dtd"] < pd.Timedelta(dte, "D")]
    div_pv = 0
    for _, row in filtered_df.iterrows():
        div_pv += row["Dividends"] * np.exp(-rate * row["dtd"].days / 365)
    forward = forward - div_pv
    return forward


def discrete_divs(tkr):
    """
    Forecast next 12 months dividends and puts in the format for quantlib

    Parameters:
    - tkr (string): Ticker for instrument.

    Returns:
    List of two list. amount(float) and Datetime objects
    """
    df = div_cal(tkr)
    dividend_dates = []
    # print(type(df.loc[1,'Date']))
    for div in df.loc[:, "Date"]:
        x = ql.Date(div.day, div.month, div.year)
        dividend_dates.append(x)
    amounts = []
    for amt in df.loc[:, "Dividends"]:
        value = amt
        amounts.append(value)
    return amounts, dividend_dates


def holiday_cal():
    """
    NYSE Holiday Dictionary for next 12 months
    """

    class USTradingHolidayCalendar(AbstractHolidayCalendar):
        rules = [
            Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday(
                "Juneteenth National Independence Day",
                month=6,
                day=19,
                start_date="2021-06-18",
                observance=nearest_workday,
            ),
            Holiday("USIndependenceDay", month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        ]

    delta = timedelta(days=365)
    sdt = datetime.today().date()
    edt = sdt + delta
    cal = USTradingHolidayCalendar()
    holiday_list = pd.DataFrame(columns=["Holiday Date"])
    for dt in cal.holidays(start=sdt, end=edt):
        new_data = pd.DataFrame({"Holiday Date": [dt]})
        holiday_list = holiday_list._append(new_data, ignore_index=True)

    holiday_list["Holiday Date"] = pd.to_datetime(holiday_list["Holiday Date"]).dt.date
    return holiday_list


def earnings_cal(tkr):
    """
    Earnings calendar estimates based of prior earnings dates in dictionary.
    """

    # Create a unique cache key using ticker
    cache_key = f"earnings_{tkr}"

    # Check if earnings data for the ticker is in cache
    if cache_key in cache:
        return cache[cache_key]

    # downloading the earnings calendar
    tkr_obj = get_ticker_object(tkr)
    # This pulls enough earnings dates to capture last 4
    df = tkr_obj.get_earnings_dates(limit=9)
    df["Date"] = df.index.date
    # creating a column to store date objects
    df = df[["Date"]]
    # filtering the returned object for just the last 4 quarters
    filtered_dates = df[df.index <= pd.Timestamp.now(tz="America/New_York").normalize()].index
    # counting the next 4 quarters for earnings dates
    next_4_quarters = [(date + DateOffset(months=12) - BDay(1)) for date in filtered_dates]
    # this creates a datetime.date object
    clean_dates = [timestamp.date() for timestamp in next_4_quarters]
    # print(filtered_dates) # Just a check
    clean_df = pd.DataFrame(clean_dates, columns=["Earnings Date"])
    clean_df["Earnings Date"] = pd.to_datetime(clean_df["Earnings Date"]).dt.date

    cache[cache_key] = clean_df

    return clean_df


def opt_exp(tkr):
    """
    Gets option expirations for next 6 months from yfinance library. Returns dataframe.

    """
    # Get option expirations
    tkr = get_ticker_object(tkr)
    df = pd.DatetimeIndex(tkr.options)
    cutoff_date = pd.Timestamp.now() + pd.DateOffset(months=6)
    filtered_exp = df[df < cutoff_date]

    df = pd.DataFrame(filtered_exp, columns=["Date"])

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def options_df(tkr):
    """
    Returns option chain from yfinance for given tkr (string). Returns dataframe with reduced columns.
    """
    maturities = opt_exp(tkr)
    all_opts = []
    tkr = get_ticker_object(tkr)
    for e in maturities["Date"]:
        opt = tkr.option_chain(e.strftime("%Y-%m-%d"))
        combined_opt = pd.concat([opt.calls, opt.puts])
        combined_opt["expirationDate"] = e
        all_opts.append(combined_opt)
    all_opts = pd.concat(all_opts, ignore_index=True)

    # Boolean column if the option is a CALL
    all_opts["CALL"] = all_opts["contractSymbol"].str[4:].apply(lambda x: "C" in x)

    all_opts[["bid", "ask", "strike"]] = all_opts[["bid", "ask", "strike"]].apply(pd.to_numeric)
    all_opts["midmark"] = (
        all_opts["bid"] + all_opts["ask"]
    ) / 2  # Calculate the midpoint of bid-ask

    # Drop unnecessary and meaningless columns
    all_opts = all_opts.drop(
        columns=[
            "contractSize",
            "currency",
            "change",
            "percentChange",
            "lastTradeDate",
            "lastPrice",
            "impliedVolatility",
        ]
    )

    return all_opts


def get_atm(tkr):
    stock = get_ticker_object(tkr)
    price = stock.history(period="1m")["Close"].iloc[-1]
    df = options_df(tkr)

    calls = df[df["CALL"] == True]

    def find_atm_strike(group):
        # Find the index of the option strike that's closest to the current stock price
        idx = (group["strike"] - price).abs().idxmin()
        return group.loc[idx]

    result = calls.groupby("expirationDate").apply(find_atm_strike).reset_index(drop=True)
    return result


def filter_NTM_options(tkr, window=5):
    """
    Reduces option strikes to window (num_strikes) below and above the current stock.
    Calculates the mid-market prices for the option row.
    Returns dataframe.

    """
    # Download the current stock price
    stock = get_ticker_object(tkr)
    current_price = stock.history(period="1d")["Close"].iloc[0]

    # Download the options chain
    df = options_df(tkr)

    # Function to filter for each expiration group
    def filter_for_group(group):
        def filter_by_type(df, is_call_value):
            subset = df[df["CALL"] == is_call_value]
            subset = subset.sort_values("strike")
            closest_idx = subset["strike"].sub(current_price).abs().idxmin()
            lower_idx = max(subset.index.min(), closest_idx - window)
            upper_idx = min(subset.index.max(), closest_idx + window)
            return subset.loc[lower_idx:upper_idx]

        calls = filter_by_type(group, True)
        puts = filter_by_type(group, False)
        return pd.concat([calls, puts])

    # Group by expiration and then apply the filter_for_group function
    filtered_df = df.groupby("expirationDate").apply(filter_for_group).reset_index(drop=True)
    return filtered_df

# This block is for demonstration purposes only.
if __name__ == "__main__":
    print(filter_NTM_options('aapl'))
