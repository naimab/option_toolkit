import numpy as np
import QuantLib as ql
from option_data import (
    discrete_divs,
    filter_NTM_options,
    get_current_price,
    get_ff_rate,
)


# 1. Setup
calculation_date = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = calculation_date


def opt_board(tkr):
    """
    Uses the quantlib library to price an American option and the yfinance tool to bring in tradeable option strikes and expirations.
    tkr (string)
    Returns Dataframe.
    """
    calculation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = calculation_date
    options_data = filter_NTM_options(tkr, 5)
    spot_price = get_current_price(tkr)
    dividend_rate = np.sum(discrete_divs(tkr)[0]) / spot_price
    rate = get_ff_rate()

    # Pass in skewmaster derived vol_surface. Call dictionary for strike vol. Benefit of O(1) complexity

    def calc_my_price(row, dividend_rate, spot_price):
        strike = row["strike"]
        maturity_date = row["expirationDate"]
        risk_free_rate = rate
        option_type = ql.Option.Call if row["CALL"] else ql.Option.Put
        volatility = 0.30
        day_count = ql.Actual360()

        # Convert maturity_date from numpy object to appropriate QuantLib object
        ql_maturity_date = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)

        # Construct QuantLib option type and exercise
        payoff = ql.PlainVanillaPayoff(option_type, strike)
        exercise = ql.AmericanExercise(calculation_date, ql_maturity_date)
        american_option = ql.VanillaOption(payoff, exercise)

        # Set up other QuantLib objects variables
        # handles spot
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

        # handles rates
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, risk_free_rate, day_count)
        )

        # handles dividends
        flat_dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(
                calculation_date,
                ql.QuoteHandle(ql.SimpleQuote(dividend_rate)),
                ql.Actual360(),
            )
        )

        # handles vol curve
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(calculation_date, ql.NullCalendar(), volatility, day_count)
        )

        # summarizes variables for input into engine
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, flat_dividend_ts, flat_ts, flat_vol_ts
        )
        steps = 500
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        american_option.setPricingEngine(binomial_engine)
        price = american_option.NPV()
        return price

    options_data["my price"] = options_data.apply(
        lambda row: calc_my_price(row, dividend_rate, spot_price), axis=1
    )
    return options_data


def numerical_vega(option, volatility_quote, d_sigma=0.001):
    """
    Calculates an estimate of the vega of a particular option. Returns Float.
    """
    # Get the option price for the current volatility
    current_vol = volatility_quote.value()
    option_price1 = option.NPV()

    # Bump up the volatility and get the new option price
    volatility_quote.setValue(current_vol + d_sigma)
    option_price2 = option.NPV()

    # Reset the volatility back to its original value
    volatility_quote.setValue(current_vol)

    return (option_price2 - option_price1) / d_sigma


def newton_raphson_implied_vol(
    option,
    market_price,
    volatility_quote,
    process,
    initial_guess,
    accuracy=1.0e-4,
    max_iterations=100,
):
    sigma = initial_guess
    for i in range(max_iterations):
        volatility_quote.setValue(sigma)
        option_price = option.NPV()
        vega = numerical_vega(option, volatility_quote)

        # Protect against zero vega scenarios
        if abs(vega) < 1.0e-10:
            print("Vega close to zero. Aborting.")
            return None

        error = option_price - market_price
        sigma_change = error / vega

        # If change is very small, break
        if abs(sigma_change) < accuracy:
            break

        sigma -= sigma_change

    return sigma


def calculate_implied_volatility(row, dividend_rate, spot_price, rate):
    strike = row["strike"]
    maturity_date = row["expirationDate"]
    risk_free_rate = rate
    option_type = ql.Option.Call if row["CALL"] else ql.Option.Put
    initial_guess = 0.20

    # Convert maturity_date from numpy object to appropriate QuantLib object
    ql_maturity_date = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)
    # Construct the American option
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.AmericanExercise(calculation_date, ql_maturity_date)
    american_option = ql.VanillaOption(payoff, exercise)

    # 2. Setup the Pricing Engine
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(
            calculation_date,
            ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)),
            ql.Actual360(),
        )
    )
    flat_dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(
            calculation_date,
            ql.QuoteHandle(ql.SimpleQuote(dividend_rate)),
            ql.Actual360(),
        )
    )
    volatility_value = ql.SimpleQuote(initial_guess)  # using SimpleQuote for volatility
    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            calculation_date,
            ql.NullCalendar(),
            ql.QuoteHandle(volatility_value),
            ql.Actual360(),
        )
    )
    bsm_process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot_price)),
        flat_dividend_ts,
        flat_ts,
        flat_vol_ts,
    )
    steps = 750
    binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
    american_option.setPricingEngine(binomial_engine)
    # 3. Compute the Implied Volatility
    market_price = row["midmark"]
    # Handle near-zero market prices
    NEAR_ZERO_THRESHOLD = 0.05
    if market_price < NEAR_ZERO_THRESHOLD:
        return None

    try:
        implied_vol = newton_raphson_implied_vol(
            american_option, market_price, volatility_value, bsm_process, initial_guess
        )
    except Exception as e:
        print(f"Error for row {row}: {e}")
        implied_vol = None 

    return implied_vol


def calculate_implied_volatility_board(tkr):
    """
    Calculates the implied volatily for a range of options generated by yfinance.
    Returns a dataframe with a column of implied_vol appended.
    """
    # Option parameters
    options_data = filter_NTM_options(tkr, 3)
    spot_price = get_current_price(tkr)
    dividend_rate = np.sum(discrete_divs(tkr)[0]) / 365
    rate = get_ff_rate()

    options_data["implied_vol"] = options_data.apply(
        lambda row: calculate_implied_volatility(row, dividend_rate, spot_price, rate),
        axis=1,
    )
    return options_data


if __name__ == "__main__":
    print(calculate_implied_volatility_board("aapl").head(50))
