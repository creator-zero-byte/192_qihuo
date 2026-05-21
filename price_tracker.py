"""
价格追踪器 (v5) —— 输出实际价格、预测价格、交易触发详情、波动率状态
"""

import time
import logging
from datetime import datetime

from config import (
    PRICE_TRACKER_LOG,
    PRICE_TRACKER_INTERVAL,
    CONFIDENCE_THRESHOLD,
    PROFIT_COST_RATIO,
    COMMISSION_PER_HAND,
    ESTIMATED_SLIPPAGE_TICKS,
    FLIP_COOLDOWN,
    tick_buffer,
    predicted_vector,
    system_state,
    repair_state,
    runtime_data,
    trade_stats,
)

logger = logging.getLogger("futures_trader.price_tracker")


class PriceTracker:
    def __init__(self):
        self._last_output_time = 0.0
        self._file = None
        self._line_count = 0
        self._open_file()

    def _open_file(self):
        try:
            self._file = open(PRICE_TRACKER_LOG, "a", encoding="utf-8", buffering=1)
            self._file.write("\n")
            self._file.write("=" * 140 + "\n")
            self._file.write(
                f"  价格追踪器 v5 启动  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self._file.write("=" * 140 + "\n")
            self._write_header()
        except Exception as e:
            logger.error(f"无法打开价格追踪文件: {e}")

    def _write_header(self):
        if self._file is None:
            return
        self._file.write(
            f"{'时间':^22} | {'实际价格':^10} | {'预测价格':^10} | "
            f"{'方向':^8} | {'置信度':^7} | {'位移':^8} | "
            f"{'持仓':^4} | {'模式':^8} | {'Regime':^8} | {'ATR':^6} | {'日波动':^6} | 备注\n"
        )
        self._file.write("-" * 140 + "\n")
        self._line_count = 0

    def update(self):
        now = time.time()
        if now - self._last_output_time < PRICE_TRACKER_INTERVAL:
            return
        self._last_output_time = now

        if self._file is None or len(tick_buffer) == 0:
            return

        latest = tick_buffer[-1]
        actual_price = latest["last_price"]
        pred_dir = predicted_vector["direction"]
        pred_conf = predicted_vector["confidence"]
        pred_disp = predicted_vector["expected_displacement"]
        pred_price = predicted_vector.get("predicted_price", 0.0)

        if pred_price == 0.0 and pred_dir != 0:
            pred_price = actual_price + pred_dir * pred_disp

        dir_map = {1: "LONG", -1: "SHORT", 0: "---"}
        dir_str = dir_map.get(pred_dir, "---")
        pos = system_state["current_pos"]
        pos_str = f"{pos:+d}" if pos != 0 else " 0"
        mode = system_state["mode"]
        regime = runtime_data.get("volatility_regime", "?")
        atr = runtime_data.get("current_atr_ticks", 0)
        daily_range = runtime_data.get("daily_range", 0)
        trading_ok = runtime_data.get("trading_allowed", False)
        signal_type = predicted_vector.get("signal_type", "NONE")

        time_str = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.")
            + f"{int(now * 1000) % 1000:03d}"
        )

        parts = []
        if not trading_ok:
            parts.append(f"REGIME={regime} 禁止交易")
        if pred_dir == 0:
            parts.append("无信号")
        elif pred_conf < CONFIDENCE_THRESHOLD:
            parts.append(f"置信度不足({pred_conf:.1%})")

        tick_size = runtime_data.get("tick_size", 1.0)
        vol_mult = latest.get("volume_multiple", 1) or 1
        spread = latest.get("spread", 0.0)
        transaction_cost = (
            spread
            + 2 * COMMISSION_PER_HAND / vol_mult
            + 2 * ESTIMATED_SLIPPAGE_TICKS * tick_size  # v5: 双边
        )
        profit_threshold = transaction_cost * PROFIT_COST_RATIO

        if pred_disp < profit_threshold:
            parts.append(f"位移不足({pred_disp:.2f}<{profit_threshold:.2f})")

        if not parts:
            parts.append(f"条件满足 [{signal_type}]")

        pred_price_str = f"{pred_price:.2f}" if pred_price > 0 else "---"

        line = (
            f"{time_str} | {actual_price:>10.2f} | {pred_price_str:>10} | "
            f"{dir_str:^8} | {pred_conf:>6.1%} | {pred_disp:>8.4f} | "
            f"{pos_str:^4} | {mode:^8} | {regime:^8} | {atr:>5.2f} | {daily_range:>5.0f} | {' '.join(parts)}\n"
        )

        try:
            self._file.write(line)
            self._line_count += 1
            if self._line_count % 50 == 0:
                self._file.write("-" * 140 + "\n")
                self._write_header()
        except Exception as e:
            logger.error(f"写入追踪数据异常: {e}")

    def record_trade_trigger(
        self,
        direction,
        price,
        confidence,
        displacement,
        spread,
        slippage,
        pnl,
        old_pos,
    ):
        if self._file is None:
            return

        dir_str = "LONG" if direction == 1 else "SHORT"
        old_str = {1: "多", -1: "空", 0: "空仓"}.get(old_pos, "?")
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # v5: 增加净盈亏和成本明细
        vol_mult = tick_buffer[-1].get("volume_multiple", 1) if tick_buffer else 1
        gross_pnl = (
            (price - (trade_stats.get("_last_entry", price) or price))
            * direction
            * vol_mult
        )

        self._file.write("\n")
        self._file.write("+" + "-" * 80 + "+\n")
        self._file.write(f"|  TRADE TRIGGER  {time_str:^52}     |\n")
        self._file.write("+" + "-" * 80 + "+\n")
        self._file.write(f"|  Flip: {old_str} -> {dir_str:<58}|\n")
        self._file.write(f"|  Price: {price:<60.2f}|\n")
        self._file.write(f"|  Confidence: {confidence:<55.2%}|\n")
        self._file.write(f"|  Displacement: {displacement:<52.4f}|\n")
        self._file.write(f"|  Spread: {spread:<59.2f}|\n")
        self._file.write(f"|  Slippage: {slippage:<57.2f}|\n")
        self._file.write(f"|  Net PnL: {pnl:<+60.2f}|\n")
        self._file.write(f"|  Cumulative PnL: {trade_stats['total_pnl']:<+49.2f}|\n")
        self._file.write(
            f"|  Signal Type: {predicted_vector.get('signal_type', '?'):<54}|\n"
        )
        self._file.write("+" + "-" * 80 + "+\n\n")
        self._write_header()

    def close(self):
        if self._file:
            self._file.write("\n" + "=" * 140 + "\n")
            self._file.write(
                f"  价格追踪器 v5 关闭  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self._file.write("=" * 140 + "\n")
            self._file.close()
            self._file = None
