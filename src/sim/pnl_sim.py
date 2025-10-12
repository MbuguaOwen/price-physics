import numpy as np
import pandas as pd

def probs_to_trades(probs: np.ndarray, threshold: float = 0.58):
    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    classes = np.array([1,0,-1])[idx]
    positions = np.where(conf >= threshold, classes, 0)
    positions = np.where(positions == 1, 1, np.where(positions == -1, -1, 0))
    return positions

def pnl_from_positions(positions: np.ndarray, prices: np.ndarray, fee_bps: float = 3.5, slippage_bps: float = 2.0):
    positions = positions.astype(float)
    rets = np.diff(prices, prepend=prices[0]) / prices
    pnl = positions * rets
    turnover = np.abs(np.diff(positions, prepend=0.0))
    costs = (fee_bps + slippage_bps) * 1e-4 * turnover
    pnl_after_costs = pnl - costs
    equity = (1 + pnl_after_costs).cumprod()
    dd = 1 - equity / equity.cummax()
    return pd.DataFrame({"pnl": pnl_after_costs, "equity": equity, "drawdown": dd})
