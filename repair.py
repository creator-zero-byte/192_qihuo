"""
预测偏差自修复模块 (v5)
"""

import time
import logging
import numpy as np
from config import (
    SYMBOL,
    MAX_REPAIR_DURATION,
    FLIP_GRACE_PERIOD,
    tick_buffer,
    system_state,
    repair_state,
    runtime_data,
    trade_stats,
    REPAIR_SIGMA_MULTIPLIER,
    REPAIR_MIN_HOLD_TIME,
    REPAIR_ERROR_DECAY_RATE,
)

logger = logging.getLogger("futures_trader.repair")


class RepairModule:
    """预测偏差自修复中枢 v5"""

    def __init__(self, api, quote, account, prediction_engine, executor):
        self.api = api
        self.quote = quote
        self.account = account
        self.engine = prediction_engine
        self.executor = executor
        self._last_repair_exit_time = 0.0
        self._last_smooth_exit_time = 0.0
        self._consecutive_repair_count = 0
        self._last_repair_trigger_time = 0.0

    def check_deviation(self):
        if system_state["current_pos"] == 0:
            self._consecutive_repair_count = 0
            return
        if repair_state["active"]:
            return
        if len(tick_buffer) < 50:
            return

        now = time.time()
        time_since_flip = now - runtime_data["last_flip_time"]
        if time_since_flip < FLIP_GRACE_PERIOD:
            return

        entry_time = repair_state.get("entry_time", 0.0)
        if entry_time > 0 and (now - entry_time) < REPAIR_MIN_HOLD_TIME:
            return

        if now - self._last_repair_exit_time < FLIP_GRACE_PERIOD:
            return

        p_now = self.quote.last_price
        entry_price = repair_state["entry_price"]
        predict_target = repair_state["predict_target"]
        pos = system_state["current_pos"]

        if entry_price == 0 or predict_target == 0:
            return

        if pos == 1:
            raw_error = p_now - predict_target
        else:
            raw_error = predict_target - p_now

        time_in_position = now - runtime_data["last_flip_time"]
        if time_in_position > 0:
            price_velocity = (p_now - entry_price) * pos
            if price_velocity > 0:
                effective_error = raw_error * (
                    1.0 - REPAIR_ERROR_DECAY_RATE * min(time_in_position, 10)
                )
            else:
                effective_error = raw_error
        else:
            effective_error = raw_error

        prices = [t["last_price"] for t in list(tick_buffer)[-50:]]
        dp = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        sigma = np.std(dp) if len(dp) > 5 else 1.0
        if sigma == 0:
            sigma = 1.0

        sigma_mult = REPAIR_SIGMA_MULTIPLIER + self._consecutive_repair_count * 0.5
        error_threshold = sigma_mult * sigma

        if effective_error < -error_threshold:
            repair_state["active"] = True
            repair_state["start_time"] = time.time()
            system_state["mode"] = "REPAIR"
            trade_stats["repair_count"] += 1
            repair_state["error_count"] += 1

            if now - self._last_repair_trigger_time < 60:
                self._consecutive_repair_count += 1
            else:
                self._consecutive_repair_count = 1
            self._last_repair_trigger_time = now

            logger.warning(
                f"预测偏差超阈值! error={effective_error:.2f}, "
                f"阈值={error_threshold:.2f}, 启动自修复"
            )

            if repair_state["error_count"] >= 3:
                self.engine.recalibrate()
                repair_state["error_count"] = 0

    def run(self):
        if not repair_state["active"]:
            return

        elapsed = time.time() - repair_state["start_time"]
        if elapsed > MAX_REPAIR_DURATION:
            self._timeout_force_exit()
            return

        total_pnl = self._calc_total_pnl()
        if total_pnl >= 0:
            self._repair_success()
            return

        self._smooth_exit()

    def _calc_total_pnl(self):
        try:
            return self.account.float_profit
        except Exception:
            return 0.0

    def _smooth_exit(self):
        now = time.time()
        if now - self._last_smooth_exit_time < 1.0:
            return

        p_now = self.quote.last_price
        pos = system_state["current_pos"]

        if len(tick_buffer) < 5:
            return

        recent_prices = [t["last_price"] for t in list(tick_buffer)[-5:]]
        price_trend = recent_prices[-1] - recent_prices[0]

        favorable_move = (pos == 1 and price_trend > 0) or (
            pos == -1 and price_trend < 0
        )

        if favorable_move or abs(price_trend) <= 1:
            try:
                self.executor.set_target_position(0)
                self._last_smooth_exit_time = now
            except Exception as e:
                logger.error(f"平滑退场异常: {e}")

    def _repair_success(self):
        logger.info("自修复成功, 权益已回归零轴")
        self.executor.set_target_position(0)
        repair_state["active"] = False
        repair_state["start_time"] = None
        repair_state["hedge_pos"] = 0
        system_state["mode"] = "NORMAL"
        system_state["current_pos"] = 0
        self._last_repair_exit_time = time.time()
        self._consecutive_repair_count = 0

    def _timeout_force_exit(self):
        logger.warning("自修复超时, 强制平仓止损退出")
        try:
            self.executor.set_target_position(0)
        except Exception as e:
            logger.error(f"超时平仓异常: {e}")

        repair_state["active"] = False
        repair_state["start_time"] = None
        repair_state["hedge_pos"] = 0
        system_state["mode"] = "NORMAL"
        system_state["current_pos"] = 0
        self._last_repair_exit_time = time.time()
        self._consecutive_repair_count = 0
