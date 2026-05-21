"""
v5.4 策略回测 - 高频精选版
核心洞察: 毛盈利 +560 元, 但成本 864 元 -> 需要减少交易频率
方案: 只在"完美信号"时入场 (3 信号 + 趋势确认 + 高偏离)
"""

import numpy as np
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_price_log(log_path):
    prices = []
    timestamps = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*([\d.]+)\s*\|", line
            )
            if match:
                timestamps.append(match.group(1))
                prices.append(float(match.group(2)))
    return np.array(prices, dtype=np.float64), timestamps


def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    diffs = np.diff(prices[-(period + 1) :])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain, avg_loss = np.mean(gains), np.mean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


def compute_bollinger(prices, period=60, std_mult=2.0):
    if len(prices) < period:
        return 0.0, 0.0, 0.0
    recent = prices[-period:]
    middle = float(np.mean(recent))
    std = float(np.std(recent))
    return middle + std_mult * std, middle, middle - std_mult * std


def compute_zscore(prices, period=60):
    if len(prices) < period:
        return 0.0
    recent = prices[-period:]
    mean, std = np.mean(recent), np.std(recent)
    if std < 1e-10:
        return 0.0
    return float((prices[-1] - mean) / std)


def backtest_selective(
    prices,
    tick_size=1.0,
    vol_mult=5,
    commission=3.0,
    slippage_ticks=1,
    sl=10,
    tp=15,
    zs=2.0,
    min_signals=3,
):
    """
    v5.4 精选版: 极高信号质量要求
    """
    ROUND_TRIP_COST = 2 * commission + 2 * slippage_ticks * tick_size * vol_mult
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70

    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    total_pnl = 0.0
    wins = 0
    losses = 0
    cooldown_until = 0
    signal_count = 0
    entry_count = 0

    for i in range(200, len(prices)):
        window = prices[: i + 1]
        current_price = prices[i]

        daily_range = (np.max(window[-200:]) - np.min(window[-200:])) / tick_size
        if daily_range < 15:
            continue

        sma200 = np.mean(window[-200:])
        price_vs_sma = (current_price - sma200) / tick_size

        if position != 0:
            if position == 1 and current_price <= stop_loss:
                pnl = (
                    current_price - entry_price
                ) * position * vol_mult - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "SL",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
                cooldown_until = i + 15
            elif position == -1 and current_price >= stop_loss:
                pnl = (
                    current_price - entry_price
                ) * position * vol_mult - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "SL",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
                cooldown_until = i + 15
            elif position == 1 and current_price >= take_profit:
                pnl = (
                    current_price - entry_price
                ) * position * vol_mult - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "TP",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
                cooldown_until = i + 15
            elif position == -1 and current_price <= take_profit:
                pnl = (
                    current_price - entry_price
                ) * position * vol_mult - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "TP",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
                cooldown_until = i + 15
            continue

        if i < cooldown_until:
            continue

        signals = []

        # 1. Z-score
        zscore = compute_zscore(window, 60)
        upper, mid, lower = compute_bollinger(window, 60, 2.0)
        if zscore < -zs:
            conf = min(0.95, abs(zscore) / 3.0)
            if current_price <= lower:
                conf = min(0.98, conf * 1.1)
            signals.append((1, conf, "Z"))
        elif zscore > zs:
            conf = min(0.95, abs(zscore) / 3.0)
            if current_price >= upper:
                conf = min(0.98, conf * 1.1)
            signals.append((-1, conf, "Z"))

        # 2. RSI
        rsi = compute_rsi(window, 14)
        if rsi < RSI_OVERSOLD:
            conf = (RSI_OVERSOLD - rsi) / RSI_OVERSOLD * 0.75
            signals.append((1, conf, "R"))
        elif rsi > RSI_OVERBOUGHT:
            conf = (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 0.75
            signals.append((-1, conf, "R"))

        # 3. VWAP 回归
        vwap = np.mean(window[-200:])
        dev_ticks = abs(current_price - vwap) / tick_size
        if dev_ticks > 10:  # 更严格
            direction = -1 if current_price > vwap else 1
            conf = min(0.9, dev_ticks / 12.0)
            signals.append((direction, conf, "V"))

        # 4. BB 位置信号
        if upper > 0 and lower > 0:
            bb_pos = (current_price - lower) / (upper - lower)
            if bb_pos < 0.1:  # 价格接近 BB 下轨
                signals.append((1, 0.5, "BB"))
            elif bb_pos > 0.9:  # 价格接近 BB 上轨
                signals.append((-1, 0.5, "BB"))

        signal_count += 1

        if len(signals) < min_signals:
            continue

        dir_counts = {}
        dir_conf_sum = {}
        for d, c, t in signals:
            dir_counts[d] = dir_counts.get(d, 0) + 1
            dir_conf_sum[d] = dir_conf_sum.get(d, 0.0) + c

        best_dir = max(dir_counts.keys(), key=lambda d: dir_conf_sum[d])
        n_signals = dir_counts[best_dir]
        avg_conf = dir_conf_sum[best_dir] / n_signals

        if n_signals < min_signals:
            continue

        # 趋势加成
        trend_bonus = 0.0
        if best_dir == 1 and price_vs_sma < -5:
            trend_bonus = 0.12
        elif best_dir == -1 and price_vs_sma > 5:
            trend_bonus = 0.12

        confidence = min(1.0, avg_conf + 0.05 * (n_signals - 1) + trend_bonus)
        if confidence < 0.50:
            continue

        entry_price = current_price
        position = best_dir
        entry_count += 1
        stop_loss = entry_price - best_dir * sl * tick_size
        take_profit = entry_price + best_dir * tp * tick_size

    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    net_pnls = [t["pnl"] for t in trades]
    gross_pnls = [(t["exit"] - t["entry"]) * t["dir"] * vol_mult for t in trades]

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_net_pnl": np.mean(net_pnls) if net_pnls else 0,
        "avg_gross_pnl": np.mean(gross_pnls) if gross_pnls else 0,
        "total_gross_pnl": sum(gross_pnls),
        "total_costs": ROUND_TRIP_COST * total_trades,
        "max_win": max(net_pnls) if net_pnls else 0,
        "max_loss": min(net_pnls) if net_pnls else 0,
        "signal_count": signal_count,
        "entry_count": entry_count,
        "trades": trades,
        "round_trip_cost": ROUND_TRIP_COST,
    }


def main():
    prices, _ = parse_price_log("/workspace/uploads/price_tracker.log")
    print(f"解析 {len(prices)} 个 tick 数据")

    # 测试不同最低信号数要求
    for min_sig in [2, 3, 4]:
        for zs in [1.8, 2.0, 2.2]:
            r = backtest_selective(prices, zs=zs, min_signals=min_sig)
            if r["total_trades"] >= 3:
                win_sum = sum(t["pnl"] for t in r["trades"] if t["pnl"] > 0)
                loss_sum = abs(sum(t["pnl"] for t in r["trades"] if t["pnl"] <= 0))
                pf = win_sum / loss_sum if loss_sum > 0 else 0
                print(
                    f"min_sig={min_sig}, zs={zs:.1f}: {r['total_trades']}笔, "
                    f"胜率={r['win_rate']:.1%}, 盈亏={r['total_pnl']:+.2f}, "
                    f"均净={r['avg_net_pnl']:+.2f}, PF={pf:.2f}, "
                    f"信号/入场={r['signal_count']}/{r['entry_count']}"
                )

    # 最终推荐
    print(f"\n{'=' * 60}")
    print(f"  推荐配置分析:")
    print(f"{'=' * 60}")

    for min_sig, zs in [(2, 2.0), (3, 1.8), (3, 2.0)]:
        r = backtest_selective(prices, zs=zs, min_signals=min_sig)
        print(f"\n  min_signals={min_sig}, ZS={zs}:")
        print(f"    交易数: {r['total_trades']}")
        print(f"    胜率: {r['win_rate']:.1%}")
        print(f"    毛盈亏: {r['total_gross_pnl']:+.2f} 元")
        print(f"    总成本: {r['total_costs']:.2f} 元")
        print(f"    净盈亏: {r['total_pnl']:+.2f} 元")
        print(
            f"    信号利用率: {r['total_trades']}/{r['signal_count']} = {r['total_trades'] / r['signal_count']:.1%}"
            if r["signal_count"] > 0
            else ""
        )


if __name__ == "__main__":
    main()
