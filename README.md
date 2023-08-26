# Option Trader Toolkit

The Option Trader Toolkit is a collection of Python tools aimed at aiding option traders with a variety of functionalities from volatility calculation to option valuation. The repository contains multiple scripts, each serving a unique purpose in the world of option trading. More features are in development to make the toolkit even more robust.

## Table of Contents

- [Scripts and Their Functions](#scripts-and-their-functions)
  - [skewmaster.py](#skewmasterpy)
  - [impliedVolandValuationl.py](#impliedvolandvaluationlpy)
  - [opt_data.py](#opt_datapy)
  - [clean_vol.py](#clean_volpy)
  - [index_correlation.py](#index_correlationpy)
- [Contributing](#contributing)

## Scripts and Their Functions

### skewmaster.py

- **Imports**: yfinance, numpy, math, opt_chain, datetime, fredapi
- **Functions**:
  - `Days_until_expiration()`: Gives days until expiration for an option of a given maturity.
  - `Skew()`: Calculates the vol of a specific strike based on its distance from the ATM call.
  - `Skew_board()`: Provides a custom strike volatility across an entire dataframe of strikes and maturities.

### impliedVolandValuationl.py

- **Imports**: quantlib, option_data, numpy, fredapi, numpy, yfinance
- **Functions**:
  - `Numerical_vega()`: Estimates the vega of an option.
  - `Newton_raphson_implied_vol`: Calculates implied vol using the Newton method.
  - `Calculate_implied_volatility`: Calculates implied vol using QuantLib and Newton's method.
  - `Calc_implied_volatility_board()`: Computes implied vol across a dataframe of strikes and expirations.
  - `Opt_board()`: Creates an option valuation for a range of strikes and expirations.

### opt_data.py

- **Imports**: pandas, numpy, yfinance, datetime, pandas.tseries, quantlib, yfinance, fredapi
- **Functions**:
  - `Opt_exp()`: Fetches available expiration dates for a stock for the next 6 months.
  - `Options_df()`: Reformats the option chain to provide the necessary columns and maturities.
  - `Filter_NTM_options()`: Reduces the options dataframe for strikes close to the money.
  - `Get_current_price()`: Fetches the current price of a stock.
  - `Get_forward()`: Determines the forward value based on the FED rate.
  - `Holiday_cal()`: Lists NYST holidays for the upcoming 12 months.
  - `Earnings_cal()`: Retrieves earnings call dates for the next 12 months.
  - `Div_cal()`: Gets dividend amount and ex-div dates for the upcoming year.
  - `Discrete_divs()`: Provides a list of dividend amounts and dates for the next year.

### clean_vol.py

- **Imports**: pandas, numpy, datetime, opt_calendar, opt_chain, main, implied_vol
- **Functions**:
  - `set_clean_vol()`: Adjusts the 365-day volatility to a 260-day one based on trading, weekend, and holiday days.
  - `compare_vol()`: Sets clean vol and adds implied vols column for specific expirations and strikes.

### index_correlation.py

- **Imports**: numpy
- **Functions**:
  - `Calculate_implied_correlation()`: Determines the implied correlation for an index based on individual stock weights, correlations, and vols.

## Contributing

As of now, this repository is not open for contributions. However, feedback and suggestions are always welcome.

---

Feel free to adjust the content as per your requirements. This is a general structure to give you a good starting point.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
