from datetime import datetime, timedelta
import numpy as np
import math
from option_data import filter_NTM_options, forward, get_ff_rate, days_until_expiration

"""
Creates a skewmaster function that outputs a vol surface based on user inputs
"""

# set for skew_board function. Currently model for american exercise does not handle well close to expiration.
current_date = datetime.now().date()
two_days_later = current_date + timedelta(days=2)


# should take in the forward price as well
def skew(
    stock, atfVol, psk, csk, pPlus, cPlus, min_c, min_p, strike, expiration, fwd, e_strike=0.5
):
    """
    Calculates and returns the skewed strike vol based on given parameters.
    
    Parameters:
    - stock (float)
    - atfVol (float)
    - psk (float)
    - csk (float)
    - pPlus (float)
    - min_c (float)
    - min_p (float)
    - strike (float)
    - expiration (datetime)
    - fwd (float)
    - e_strike (float): default 0.5
    
    
    Returns:
    float: Calculated skew value.
    """
    # Set variables and access nested dictionary created above
    rate = get_ff_rate()
    t = days_until_expiration(expiration) / 365
    fwd = forward(stock, expiration)

    atfVega = fwd * np.sqrt(t / (2 * math.pi)) * np.exp(-(rate + (atfVol**2) / 8) * t)
    # Check if atfVega is close to 0
    if np.isclose(atfVega, 0, atol=0.5):  # adjust atol as needed
        return np.nan

    A_1 = -psk / (500 * atfVega)
    A_2 = pPlus / (25 * atfVega)
    B_1 = -csk / (500 * atfVega)
    B_2 = -cPlus / (25 * atfVega)
    B_3 = -1 * (B_1 + 2 * e_strike * fwd * B_2) / (3 * e_strike**2 * fwd**2)

    if strike <= fwd:
        strike_vol = np.maximum(atfVol + A_1 * (strike - fwd) + A_2 * (strike - fwd) **2, min_p)

    elif strike >= (1 + e_strike) * fwd:
        strike_vol = (
            atfVol
            + B_1 * (e_strike * fwd)
            + B_2 * (e_strike * fwd) ** 2
            + B_3 * (e_strike * fwd) ** 3
        )

    else:
        strike_vol = np.maximum((
            atfVol + B_1 * (strike - fwd) + B_2 * (strike - fwd) ** 2 + B_3 * (strike - fwd) ** 3
        ), min_c)

    return strike_vol


def skew_board(stock, atfVol, psk, csk, pPlus, cPlus, min_c, min_p, e_strike=0.5):
    """
    Uses the skew function to apply to a dataframe of strikes and expirations.
    
    Parameters:
    - stock (float)
    - atfVol (float)
    - psk (float)
    - csk (float)
    - pPlus (float)
    - min_c (float)
    - min_p (float)
    - e_strike (float): default 0.5
    
    Returns:
    dataframe
    """
    
    # Brings in dataframe of filtered strikes around the ATM strike.
    df = filter_NTM_options(stock)

    # Cache forwards for unique expirations to avoid pinging yfinance on every iteration
    unique_expirations = df["expirationDate"].unique()
    fwd_dict = {exp: forward(stock, exp) for exp in unique_expirations}

    # Using a list comprehension to compute all values at once
    my_strike_vol_values = [
        skew(
            stock,
            atfVol,
            psk,
            csk,
            pPlus,
            cPlus,
            min_c,
            min_p,
            row["strike"],
            row["expirationDate"],
            fwd_dict[row["expirationDate"]],
            e_strike=0.5,
        )
        for _, row in df.iterrows()
    ]

    df["my_strike_vol"] = my_strike_vol_values

    # Replace skew values with NaN for rows with expirationDate less than two_days_later
    df.loc[df["expirationDate"] <= two_days_later, "my_strike_vol"] = np.nan

    return df

# this is for demonstration purposes only.
if __name__ == "__main__":
    print(skew_board("aapl", 0.25, 45, 25, 0, 0, 0.10, 0.1).head(50))
