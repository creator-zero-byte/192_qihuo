"""
v5 策略回测框架
从 price_tracker.log 提取价格数据, 模拟 v5 策略交易
"""

import numpy as np
import re
import sys
import os

# Add output directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_price_log(log_path):
    """从 price_tracker.log 提取 tick 价格数据"""
    prices = []
    timestamps = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # Match lines like: 2026-05-20 09:02:00.054 |    9404.00 |
            match = re.search(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*([\d.]+)\s*\|", line
            )
            if match:
                ts_str = match.group(1)
                price = float(match.group(2))
                timestamps.append(ts_str)
                prices.append(price)

    return np.array(prices, dtype=np.float64), timestamps


def compute_atr_prices(prices, period=30):
    """计算 ATR (价格单位)"""
    if len(prices) < period + 1:
        return 1.0
    return float(np.mean(np.abs(np.diff(prices[-(period + 1) :]))))


def compute_rsi(prices, period=14):
    """Wilder's RSI"""
    if len(prices) < period + 1:
        return 50.0
    diffs = np.diff(prices[-(period + 1) :])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def compute_bollinger(prices, period=40, std_mult=2.0):
    if len(prices) < period:
        return 0.0, 0.0, 0.0
    recent = prices[-period:]
    middle = float(np.mean(recent))
    std = float(np.std(recent))
    return middle + std_mult * std, middle, middle - std_mult * std


def compute_zscore(prices, period=40):
    if len(prices) < period:
        return 0.0
    recent = prices[-period:]
    mean, std = np.mean(recent), np.std(recent)
    if std < 1e-10:
        return 0.0
    return float((prices[-1] - mean) / std)


def backtest_v5(prices, tick_size=1.0):
    """
    v5 策略回测
    使用简化但准确的信号逻辑
    """
    # v5 参数
    BB_PERIOD = 40
    BB_STD_MULT = 2.0
    RSI_PERIOD = 14
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    ZSCORE_ENTRY = 1.5
    ATR_PERIOD = 30
    MIN_DAILY_RANGE_TICKS = 20
    CONFIDENCE_THRESHOLD = 0.50
    STOP_LOSS_TICKS = 5
    TAKE_PROFIT_ATR_MULT = 1.2
    COMMISSION_PER_HAND = 3.0
    SLIPPAGE_TICKS = 1
    VOL_MULT = 5  # eb2607 volume_multiple
    PROFIT_COST_RATIO = 1.5
    VWAP_DEVIATION_TICKS = 8

    # 交易成本
    ROUND_TRIP_COST = (
        2 * COMMISSION_PER_HAND + 2 * SLIPPAGE_TICKS * tick_size * VOL_MULT
    )

    # 状态
    position = 0  # 0, 1, -1
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_idx = 0

    # 统计
    trades = []
    total_pnl = 0.0
    wins = 0
    losses = 0
    signal_count = 0
    regime_low_count = 0
    regime_ok_count = 0

    # 滑动窗口
    for i in range(100, len(prices)):
        window = prices[: i + 1]
        current_price = prices[i]

        # 日波动范围
        if i > 100:
            daily_range = (np.max(window[-100:]) - np.min(window[-100:])) / tick_size
        else:
            daily_range = 0

        # ATR (ticks)
        atr_price = compute_atr_prices(window, ATR_PERIOD)
        atr_ticks = atr_price / tick_size

        # Regime 判断
        trading_allowed = atr_ticks >= 0.5 or daily_range >= MIN_DAILY_RANGE_TICKS

        if not trading_allowed:
            regime_low_count += 1
            continue
        regime_ok_count += 1

        # 管理持仓
        if position != 0:
            # 检查止损
            if position == 1 and current_price <= stop_loss:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "STOP_LOSS",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0

            elif position == -1 and current_price >= stop_loss:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "STOP_LOSS",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0

            # 检查止盈
            elif position == 1 and current_price >= take_profit:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "TAKE_PROFIT",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0

            elif position == -1 and current_price <= take_profit:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "TAKE_PROFIT",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0

            continue

        # 信号生成
        signals = []

        # Z-score
        zscore = compute_zscore(window, BB_PERIOD)
        if zscore < -ZSCORE_ENTRY:
            conf = min(0.85, abs(zscore) / (ZSCORE_ENTRY * 2.5))
            upper, mid, lower = compute_bollinger(window, BB_PERIOD, BB_STD_MULT)
            if current_price <= lower:
                conf *= 1.0
            else:
                conf *= 0.5
            signals.append((1, conf, "ZSCORE"))
        elif zscore > ZSCORE_ENTRY:
            conf = min(0.85, abs(zscore) / (ZSCORE_ENTRY * 2.5))
            upper, mid, lower = compute_bollinger(window, BB_PERIOD, BB_STD_MULT)
            if current_price >= upper:
                conf *= 1.0
            else:
                conf *= 0.5
            signals.append((-1, conf, "ZSCORE"))

        # RSI
        rsi = compute_rsi(window, RSI_PERIOD)
        if rsi < RSI_OVERSOLD:
            conf = (RSI_OVERSOLD - rsi) / RSI_OVERSOLD * 0.65
            signals.append((1, conf, "RSI"))
        elif rsi > RSI_OVERBOUGHT:
            conf = (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 0.65
            signals.append((-1, conf, "RSI"))

        # VWAP 均值回归
        vwap = np.mean(window[-200:]) if len(window) >= 200 else np.mean(window)
        dev_ticks = abs(current_price - vwap) / tick_size
        if dev_ticks > VWAP_DEVIATION_TICKS:
            direction = -1 if current_price > vwap else 1
            conf = min(0.7, dev_ticks / (VWAP_DEVIATION_TICKS * 3))
            signals.append((direction, conf, "VWAP"))

        # 信号融合
        if not signals:
            continue

        dir_counts = {}
        dir_conf_sum = {}
        for d, c, t in signals:
            dir_counts[d] = dir_counts.get(d, 0) + 1
            dir_conf_sum[d] = dir_conf_sum.get(d, 0.0) + c

        best_dir = max(dir_counts.keys(), key=lambda d: dir_conf_sum[d])
        n_signals = dir_counts[best_dir]
        avg_conf = dir_conf_sum[best_dir] / n_signals

        if n_signals < 2:
            continue

        signal_count += 1
        agreement_bonus = 0.1 if n_signals >= 3 else 0.05
        confidence = min(1.0, avg_conf + agreement_bonus)

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # 位移检查
        expected_disp = atr_ticks * tick_size * (0.8 + confidence * 0.8)
        min_disp = ROUND_TRIP_COST * PROFIT_COST_RATIO / VOL_MULT
        if expected_disp < min_disp:
            continue

        # 入场
        entry_price = current_price
        position = best_dir
        entry_idx = i
        stop_loss = entry_price - best_dir * STOP_LOSS_TICKS * tick_size
        take_profit = entry_price + best_dir * TAKE_PROFIT_ATR_MULT * atr_price

    # 最终统计
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    # 计算每笔交易的净盈亏
    gross_pnls = [(t["exit"] - t["entry"]) * t["dir"] * VOL_MULT for t in trades]
    net_pnls = [t["pnl"] for t in trades]

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / total_trades if total_trades > 0 else 0,
        "avg_gross_pnl": np.mean(gross_pnls) if gross_pnls else 0,
        "avg_net_pnl": np.mean(net_pnls) if net_pnls else 0,
        "max_win": max(net_pnls) if net_pnls else 0,
        "max_loss": min(net_pnls) if net_pnls else 0,
        "signal_count": signal_count,
        "regime_low_count": regime_low_count,
        "regime_ok_count": regime_ok_count,
        "trades": trades,
        "round_trip_cost": ROUND_TRIP_COST,
    }


def backtest_v4_original(prices, tick_size=1.0):
    """原 v4 策略回测 (用于对比)"""
    BB_PERIOD = 50
    BB_STD_MULT = 2.5
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    ZSCORE_ENTRY = 2.0
    ATR_PERIOD = 50
    MIN_DAILY_RANGE_TICKS = 30
    CONFIDENCE_THRESHOLD = 0.65
    STOP_LOSS_TICKS = 3
    TAKE_PROFIT_ATR_MULT = 1.5
    COMMISSION_PER_HAND = 3.0
    SLIPPAGE_TICKS = 1
    VOL_MULT = 5
    PROFIT_COST_RATIO = 1.0

    ROUND_TRIP_COST = (
        2 * COMMISSION_PER_HAND + 2 * SLIPPAGE_TICKS * tick_size * VOL_MULT
    )

    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    total_pnl = 0.0
    wins = 0
    losses = 0
    signal_count = 0
    regime_low_count = 0
    regime_ok_count = 0

    for i in range(100, len(prices)):
        window = prices[: i + 1]
        current_price = prices[i]

        if i > 100:
            daily_range = (np.max(window[-100:]) - np.min(window[-100:])) / tick_size
        else:
            daily_range = 0

        atr_price = compute_atr_prices(window, ATR_PERIOD)
        atr_ticks = atr_price / tick_size

        # v4 原始 regime: ATR < 1.0 就禁止交易
        trading_allowed = atr_price >= 1.0 and daily_range >= MIN_DAILY_RANGE_TICKS

        if not trading_allowed:
            regime_low_count += 1
            continue
        regime_ok_count += 1

        if position != 0:
            if position == 1 and current_price <= stop_loss:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "STOP_LOSS",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
            elif position == -1 and current_price >= stop_loss:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "STOP_LOSS",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
            elif position == 1 and current_price >= take_profit:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "TAKE_PROFIT",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
            elif position == -1 and current_price <= take_profit:
                pnl = (
                    current_price - entry_price
                ) * position * VOL_MULT - ROUND_TRIP_COST
                total_pnl += pnl
                trades.append(
                    {
                        "idx": i,
                        "dir": position,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "reason": "TAKE_PROFIT",
                    }
                )
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0
            continue

        # v4 信号
        signals = []
        if len(window) >= BB_PERIOD:
            zscore = compute_zscore(window, BB_PERIOD)
            if zscore < -ZSCORE_ENTRY:
                conf = min(0.9, abs(zscore) / (ZSCORE_ENTRY * 2))
                signals.append((1, conf, "ZSCORE"))
            elif zscore > ZSCORE_ENTRY:
                conf = min(0.9, abs(zscore) / (ZSCORE_ENTRY * 2))
                signals.append((-1, conf, "ZSCORE"))

        if len(window) >= RSI_PERIOD + 5:
            rsi = compute_rsi(window, RSI_PERIOD)
            if rsi < RSI_OVERSOLD:
                conf = (RSI_OVERSOLD - rsi) / RSI_OVERSOLD * 0.7
                signals.append((1, conf, "RSI"))
            elif rsi > RSI_OVERBOUGHT:
                conf = (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 0.7
                signals.append((-1, conf, "RSI"))

        if not signals:
            continue

        dir_counts = {}
        dir_conf_sum = {}
        dir_types = {}
        for d, c, t in signals:
            dir_counts[d] = dir_counts.get(d, 0) + 1
            dir_conf_sum[d] = dir_conf_sum.get(d, 0.0) + c
            dir_types[d] = dir_types.get(d, set())
            dir_types[d].add(t)

        best_dir = max(dir_counts.keys(), key=lambda d: dir_conf_sum[d])
        n_signals = dir_counts[best_dir]
        avg_conf = dir_conf_sum[best_dir] / n_signals
        n_types = len(dir_types[best_dir])

        # v4 要求 3 个信号确认
        if n_signals < 3:
            continue

        signal_count += 1
        agreement_bonus = 0.15 if n_signals >= 3 else 0.08
        type_bonus = 0.1 if n_types >= 2 else 0.05
        confidence = min(1.0, avg_conf + agreement_bonus + type_bonus)

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        expected_disp = atr_ticks * tick_size * (0.5 + confidence * 1.0)
        min_disp = ROUND_TRIP_COST * PROFIT_COST_RATIO / VOL_MULT
        if expected_disp < min_disp:
            continue

        entry_price = current_price
        position = best_dir
        stop_loss = entry_price - best_dir * STOP_LOSS_TICKS * tick_size
        take_profit = entry_price + best_dir * TAKE_PROFIT_ATR_MULT * atr_price

    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    gross_pnls = [(t["exit"] - t["entry"]) * t["dir"] * VOL_MULT for t in trades]
    net_pnls = [t["pnl"] for t in trades]

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / total_trades if total_trades > 0 else 0,
        "avg_gross_pnl": np.mean(gross_pnls) if gross_pnls else 0,
        "avg_net_pnl": np.mean(net_pnls) if net_pnls else 0,
        "max_win": max(net_pnls) if net_pnls else 0,
        "max_loss": min(net_pnls) if net_pnls else 0,
        "signal_count": signal_count,
        "regime_low_count": regime_low_count,
        "regime_ok_count": regime_ok_count,
        "trades": trades,
        "round_trip_cost": ROUND_TRIP_COST,
    }


def print_results(label, results):
    """打印回测结果"""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  总交易次数:       {results['total_trades']}")
    print(f"  盈利/亏损:        {results['wins']} / {results['losses']}")
    print(f"  胜率:             {results['win_rate']:.1%}")
    print(f"  总净盈亏:         {results['total_pnl']:+.2f} 元")
    print(f"  平均毛盈亏:       {results['avg_gross_pnl']:+.2f} 元")
    print(f"  平均净盈亏:       {results['avg_net_pnl']:+.2f} 元")
    print(f"  最大单笔盈利:     {results['max_win']:+.2f} 元")
    print(f"  最大单笔亏损:     {results['max_loss']:+.2f} 元")
    print(f"  信号总数:         {results['signal_count']}")
    print(f"  允许交易 ticks:   {results['regime_ok_count']}")
    print(f"  禁止交易 ticks:   {results['regime_low_count']}")
    print(f"  单笔交易成本:     {results['round_trip_cost']:.2f} 元")
    print(f"{'=' * 60}")


def main():
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "uploads", "price_tracker.log"
    )
    if not os.path.exists(log_path):
        log_path = "/workspace/uploads/price_tracker.log"

    print(f"解析日志文件: {log_path}")
    prices, timestamps = parse_price_log(log_path)
    print(f"共解析 {len(prices)} 个 tick 数据")
    print(f"价格范围: {prices.min():.2f} - {prices.max():.2f}")
    print(f"价格波动: {prices.max() - prices.min():.0f} ticks")

    # 回测 v4 原版
    print("\n>>> 开始 v4 原版回测...")
    v4_results = backtest_v4_original(prices, tick_size=1.0)
    print_results("v4 原版策略回测", v4_results)

    # 回测 v5 优化版
    print("\n>>> 开始 v5 优化版回测...")
    v5_results = backtest_v5(prices, tick_size=1.0)
    print_results("v5 优化版策略回测", v5_results)

    # 对比分析
    print(f"\n{'=' * 60}")
    print(f"  对比分析")
    print(f"{'=' * 60}")

    pnl_improvement = v5_results["total_pnl"] - v4_results["total_pnl"]
    trades_improvement = v5_results["total_trades"] - v4_results["total_trades"]
    regime_improvement = v5_results["regime_ok_count"] - v4_results["regime_ok_count"]

    print(f"  盈亏改善: {pnl_improvement:+.2f} 元")
    print(f"  交易次数变化: {trades_improvement:+d}")
    print(f"  允许交易时间增加: {regime_improvement} ticks")
    print(f"  v4 允许交易比例: {v4_results['regime_ok_count'] / len(prices):.1%}")
    print(f"  v5 允许交易比例: {v5_results['regime_ok_count'] / len(prices):.1%}")

    if v5_results["total_trades"] > 0:
        print(f"\n  v5 策略盈亏分布:")
        winning_trades = [t for t in v5_results["trades"] if t["pnl"] > 0]
        losing_trades = [t for t in v5_results["trades"] if t["pnl"] <= 0]
        print(
            f"    盈利交易: {len(winning_trades)} 笔, 总盈利 {sum(t['pnl'] for t in winning_trades):+.2f} 元"
        )
        print(
            f"    亏损交易: {len(losing_trades)} 笔, 总亏损 {sum(t['pnl'] for t in losing_trades):+.2f} 元"
        )
        if winning_trades:
            print(
                f"    盈利交易中平均盈利: {np.mean([t['pnl'] for t in winning_trades]):.2f} 元"
            )
        if losing_trades:
            print(
                f"    亏损交易中平均亏损: {np.mean([t['pnl'] for t in losing_trades]):.2f} 元"
            )
        if winning_trades and losing_trades:
            profit_factor = sum(t["pnl"] for t in winning_trades) / abs(
                sum(t["pnl"] for t in losing_trades)
            )
            print(f"    盈亏比: {profit_factor:.2f}")

    print(f"\n{'=' * 60}")
    if v5_results["total_pnl"] > 0:
        print("  v5 策略实现正收益, 优化成功!")
    else:
        print("  v5 策略仍需进一步优化")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
