import pandas as pd
import numpy as np
import datetime
from option_data import (
    holiday_cal,
    earnings_cal,
    opt_exp,
    get_current_price,
    discrete_divs,
    get_ff_rate,
)
from option_data import get_atm
from impliedVolandValuation import calculate_implied_volatility


def set_clean_vol(tkr, clean_vol, earnings_var=1.5, normal_var=1, wknd_var=0.08, holiday_var=0):
    """
    Calculate and return the skew based on given parameters.

    Parameters:
    - tkr (string): The ticker for instrument.
    - clean_vol (float): Desired clean at the forward vol for stock.
    - earnings_var (float): Optional. the assigned variance weight to normal day. Default: 1.5.
    - normal_var (float): Optional. the assigned variance weight to normal day. Default: 1.
    - wknd_var (float): Optional. the assigned variance weight to normal day. Default: .08.
    - holiday_var (float): Optional. the assigned variance weight to normal day. Default: 0.

    Returns:
    Pandas dataframe
    """
    
    delta = datetime.timedelta(days=260)
    sdt = datetime.datetime.today()
    edt = sdt + delta
    date_range = pd.date_range(sdt, edt)
    date_range_dt = [date.date() for date in date_range]

    # get holiday and earnings calendars and option expiratons
    holidays = holiday_cal()
    earnings = earnings_cal(tkr)
    expirations = opt_exp(tkr)

    # Create empty DataFrame
    initial_col = ["my clean vol", "weights"]
    df = pd.DataFrame(columns=initial_col, index=date_range_dt)

    # Assign weights based on whether it's a weekend or workday or event
    for index, row in df.iterrows():
        if index.weekday() >= 5:
            df.at[index, "weights"] = wknd_var
        elif index in earnings["Earnings Date"].values:
            df.at[index, "weights"] = earnings_var
        elif index in holidays["Holiday Date"].values:
            df.at[index, "weights"] = holiday_var
        else:
            df.at[index, "weights"] = normal_var

        # Update clean vol
        df.at[index, "my clean vol"] = clean_vol * np.sqrt(df.at[index, "weights"])

    today = pd.Timestamp.today().date()
    yesterday = datetime.datetime.now().date() - datetime.timedelta(days=1)

    columns = ["my clean vol", "dirty vol", "ts vol", "nvd", "dte"]
    # Creates empty data frame
    new_df = pd.DataFrame(index=expirations["Date"], columns=columns)

    ann_ratio = 365 / 260
    for index, row in expirations.iterrows():
        expiration_date = row["Date"]

        if expiration_date in df.index:
            new_df.loc[expiration_date, "nvd"] = df.loc[today:expiration_date, "weights"].sum()
            new_df.loc[expiration_date, "dte"] = (expiration_date - yesterday).days
            new_df.loc[expiration_date, "my clean vol"] = df.loc[expiration_date, "my clean vol"]

    new_df["dirty vol"] = (
        ((new_df["nvd"] / new_df["dte"]) * ann_ratio) * new_df["my clean vol"] ** 2
    ) ** 0.5
    new_df["lagged_dirty_vol"] = new_df["dirty vol"].shift(1)
    new_df["lagged_dte"] = new_df["dte"].shift(1)

    numerator = (new_df["dte"] * new_df["dirty vol"] ** 2) - (
        new_df["lagged_dte"] * new_df["lagged_dirty_vol"] ** 2
    )
    denominator = new_df["dte"] - new_df["lagged_dte"]
    new_df["ts vol"] = (numerator / denominator) ** 0.5
    new_df.drop("lagged_dirty_vol", axis=1, inplace=True)
    new_df.drop("lagged_dte", axis=1, inplace=True)

    return new_df


def compare_vol(tkr, clean_vol, earnings_var=1.5, normal_var=1, wknd_var=0.08, holiday_var=0):
    """
    Get market implied ATM vol for each expiration and sets up board to compare to user vol and timespread vols.

    Parameters:
    - tkr (string): Ticker for instrument.
    - clean_vol (float)

    Returns:
    Pandas dataframe
    """

    # Option parameters
    options_data = get_atm(tkr)
    spot_price = get_current_price(tkr)
    dividend_rate = np.sum(discrete_divs(tkr)[0]) / spot_price
    rate = get_ff_rate()

    clean_df = set_clean_vol(tkr, clean_vol, earnings_var, normal_var, wknd_var, holiday_var)
    options_data["implied dirty vol"] = options_data.apply(
        lambda row: calculate_implied_volatility(row, dividend_rate, spot_price, rate),
        axis=1,
    )

    # Adding atm implied vols to clean vol dataframe
    clean_df["implied dirty vol"] = clean_df.index.map(
        options_data.set_index("expirationDate")["implied dirty vol"]
    )

    # creating lagged columns to calculate the ts vol
    clean_df["lagged_implied_dirty_vol"] = clean_df["implied dirty vol"].shift(1)
    clean_df["lagged_dte"] = clean_df["dte"].shift(1)

    numerator = (clean_df["dte"] * clean_df["implied dirty vol"] ** 2) - (
        clean_df["lagged_dte"] * clean_df["lagged_implied_dirty_vol"] ** 2
    )
    denominator = clean_df["dte"] - clean_df["lagged_dte"]
    clean_df["implied ts vol"] = (numerator / denominator) ** 0.5
    # dropping lagged columns
    clean_df.drop("lagged_implied_dirty_vol", axis=1, inplace=True)
    clean_df.drop("lagged_dte", axis=1, inplace=True)

    return clean_df


# This block is for demontration purposes
if __name__ == "__main__":
    print(compare_vol("aapl", 0.25))
