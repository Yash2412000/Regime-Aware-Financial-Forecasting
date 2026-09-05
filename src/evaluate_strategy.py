"""
evaluate_strategy.py

Runs trading evaluation
"""

import numpy as np

from trading_strategy import evaluate_strategy



def main():


    predictions = np.load(
        "results/predictions.npy"
    )


    actual_returns = np.load(
        "results/actual_returns.npy"
    )


    predictions = predictions.flatten()

    actual_returns = actual_returns.flatten()



    result = evaluate_strategy(

        predictions,

        actual_returns

    )



    signals = result["signals"]



    print("\n========== Trading Evaluation ==========")


    print(
        f"Final Capital   : ${result['final_capital']:.2f}"
    )


    print(
        f"Strategy ROI    : {result['roi']:.2f}%"
    )


    print("---------------------------------------")


    print(
        f"Trades          : {result['trades']}"
    )


    print(
        f"Winning Trades  : {result['wins']}"
    )


    print(
        f"Losing Trades   : {result['losses']}"
    )


    print(
        f"Win Rate        : {result['win_rate']:.2f}%"
    )


    print(
        f"Sharpe Ratio    : {result['sharpe']:.3f}"
    )


    print(
        f"Max Drawdown    : {result['max_drawdown']:.2f}%"
    )


    print("---------------------------------------")


    print(
        f"BUY signals     : {(signals==1).sum()}"
    )


    print(
        f"SELL signals    : {(signals==-1).sum()}"
    )


    print(
        f"HOLD signals    : {(signals==0).sum()}"
    )



if __name__ == "__main__":

    main()