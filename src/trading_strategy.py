"""
trading_strategy.py

Trading evaluation for 5-day forward return prediction.

Model:
- Predicts 5-day forward log return
- Holds trade for 5 days
- Long-only strategy

Risk controls:
- Stop loss
- Take profit
- Transaction cost
"""


import numpy as np


INITIAL_CAPITAL = 10000

TRANSACTION_COST = 0.001

HOLDING_PERIOD = 5


# Based on prediction distribution
BUY_THRESHOLD = 0.009

SELL_THRESHOLD = -0.001


STOP_LOSS = -0.05

TAKE_PROFIT = 0.05



def generate_signals(predictions):

    signals = []


    for prediction in predictions:


        if prediction > BUY_THRESHOLD:

            signals.append(1)


        elif prediction < SELL_THRESHOLD:

            signals.append(-1)


        else:

            signals.append(0)



    return np.array(signals)




def backtest(
        predictions,
        actual_returns
):


    signals = generate_signals(
        predictions
    )


    capital = INITIAL_CAPITAL


    portfolio = []


    trades = 0

    wins = 0

    losses = 0



    i = 0


    while i < len(signals):


        signal = signals[i]



        if signal == 1:


            trades += 1



            # convert log return

            trade_return = (

                np.exp(actual_returns[i])

                -

                1

            )



            # Risk management

            if trade_return < STOP_LOSS:

                trade_return = STOP_LOSS



            if trade_return > TAKE_PROFIT:

                trade_return = TAKE_PROFIT



            # Transaction cost

            trade_return -= TRANSACTION_COST

            POSITION_SIZE = 0.2

            capital += (

                POSITION_SIZE

                *

                capital

                *

                trade_return

            )



            if trade_return > 0:

                wins += 1

            else:

                losses += 1



            # skip overlapping predictions

            i += HOLDING_PERIOD



        else:

            i += 1



        portfolio.append(capital)



        # capital protection

        if capital < INITIAL_CAPITAL * 0.5:

            break



    return {

        "capital": capital,

        "portfolio": np.array(portfolio),

        "signals": signals,

        "trades": trades,

        "wins": wins,

        "losses": losses

    }




def calculate_metrics(portfolio):


    if len(portfolio) < 2:

        return 0, 0



    returns = (

        np.diff(portfolio)

        /

        portfolio[:-1]

    )



    if returns.std() != 0:


        sharpe = (

            returns.mean()

            /

            returns.std()

        ) * np.sqrt(252)



    else:

        sharpe = 0



    running_max = np.maximum.accumulate(
        portfolio
    )


    drawdown = (

        portfolio - running_max

    ) / running_max



    max_drawdown = (

        drawdown.min()

        *

        100

    )


    return sharpe, max_drawdown




def evaluate_strategy(
        predictions,
        actual_returns
):


    result = backtest(

        predictions,

        actual_returns

    )



    roi = (

        (

            result["capital"]

            -

            INITIAL_CAPITAL

        )

        /

        INITIAL_CAPITAL

    ) * 100




    sharpe, max_dd = calculate_metrics(

        result["portfolio"]

    )



    if result["trades"] > 0:


        win_rate = (

            result["wins"]

            /

            result["trades"]

        ) * 100


    else:

        win_rate = 0



    return {


        "final_capital":

            result["capital"],


        "roi":

            roi,


        "trades":

            result["trades"],


        "wins":

            result["wins"],


        "losses":

            result["losses"],


        "win_rate":

            win_rate,


        "sharpe":

            sharpe,


        "max_drawdown":

            max_dd,


        "signals":

            result["signals"]

    }