"""
协程C: TqSdk 永续状态翻转器 (v5 - 实战优化版)
核心改进:
1. 盈亏计算修正: 正确计入双边手续费+滑点
2. 动态止盈止损: 基于 ATR 自适应调整
3. 移动止损优化: 更灵敏的跟踪
4. BB 突破策略专用止盈: 突破后快速止盈
5. VWAP 回归策略专用止盈: 回归 VWAP 即平仓
"""

import time
import logging
import config
from config import (
    SYMBOL,
    CONFIDENCE_THRESHOLD,
    FLIP_COOLDOWN,
    PROFIT_COST_RATIO,
    COMMISSION_PER_HAND,
    ESTIMATED_SLIPPAGE_TICKS,
    predicted_vector,
    system_state,
    repair_state,
    runtime_data,
    trade_stats,
    get_dynamic_confidence_threshold,
    get_consecutive_loss_cooldown,
    record_trade_to_history,
    save_persistent_state,
    CONSECUTIVE_LOSS_THRESHOLD,
    COOLING_PERIOD,
    STOP_LOSS_TICKS,
    TRAILING_STOP_TICKS,
    TAKE_PROFIT_ATR_MULT,
    MIN_HOLD_SECONDS,
    MAX_HOLD_SECONDS,
    MIN_DAILY_RANGE_TICKS,
    MIN_PROFIT_TICKS,
)
from database import record_flip

logger = logging.getLogger("futures_trader.executor")


class Executor:
    """永续状态翻转器 v5"""

    def __init__(self, api, quote):
        self.api = api
        self.quote = quote
        self._last_entry_price = 0.0
        self._tick_size = 1.0
        self._target_task = None
        self._price_tracker = None
        self._highest_pnl_since_entry = 0.0
        self._stop_loss_price = 0.0
        self._take_profit_price = 0.0
        self._entry_signal_type = "UNKNOWN"

    def set_price_tracker(self, tracker):
        self._price_tracker = tracker

    def init_tick_size(self):
        try:
            self._tick_size = self.quote.price_tick
            logger.info(f"最小变动价位: {self._tick_size}")
            runtime_data["tick_size"] = self._tick_size
        except Exception:
            self._tick_size = 1.0
            runtime_data["tick_size"] = 1.0

    def _reset_target_task(self):
        self._target_task = None

    def _is_task_alive(self):
        if self._target_task is None:
            return False
        try:
            task_obj = getattr(self._target_task, "_task", None)
            if task_obj is not None and hasattr(task_obj, "done") and task_obj.done():
                return False
        except Exception:
            return False
        return True

    def _ensure_target_task(self):
        if self._target_task is not None and not self._is_task_alive():
            self._target_task = None
        if self._target_task is None:
            try:
                from tqsdk.lib import TargetPosTask

                self._target_task = TargetPosTask(self.api, SYMBOL, price="ACTIVE")
                logger.info("TargetPosTask 实例已创建 (ACTIVE)")
            except Exception as e:
                logger.error(f"创建 TargetPosTask 失败: {e}")
        return self._target_task

    def _safe_set_target(self, volume):
        try:
            task = self._ensure_target_task()
            if task is None:
                logger.error("TargetPosTask 不可用")
                return False
            task.set_target_volume(volume)
            return True
        except Exception as e:
            err_msg = str(e)
            if "已经结束" in err_msg:
                self._target_task = None
                try:
                    task = self._ensure_target_task()
                    if task is not None:
                        task.set_target_volume(volume)
                        return True
                except Exception as e2:
                    logger.error(f"重建后仍然失败: {e2}")
            else:
                logger.error(f"设置目标仓位异常: {e}")
            return False

    def handle_bad_order_exception(self, exception):
        logger.error(f"处理错单异常: {exception}")
        self._reset_target_task()
        system_state["current_pos"] = 0
        runtime_data["last_flip_time"] = time.time()
        if repair_state["active"]:
            repair_state["active"] = False
            repair_state["start_time"] = None
            repair_state["hedge_pos"] = 0
            if system_state["mode"] == "REPAIR":
                system_state["mode"] = "NORMAL"

    # ---- v5: 盈亏计算修正 ----
    def _calc_round_trip_cost(self):
        """
        计算完整交易周期(开仓+平仓)的总成本
        包括: 双边手续费 + 双边滑点
        """
        vol_mult = self.quote.volume_multiple or 1
        # 双边手续费 (开仓 + 平仓)
        commission = 2 * COMMISSION_PER_HAND
        # 双边滑点 (开仓滑点 + 平仓滑点), 以 tick 计
        slippage_cost = 2 * ESTIMATED_SLIPPAGE_TICKS * self._tick_size * vol_mult
        return commission + slippage_cost

    def _calc_net_pnl(self, entry_price, exit_price, direction):
        """
        v5: 正确的净盈亏计算
        毛盈亏 = (exit - entry) * direction * volume_multiple
        净盈亏 = 毛盈亏 - 总交易成本
        """
        vol_mult = self.quote.volume_multiple or 1
        gross_pnl = (exit_price - entry_price) * direction * vol_mult
        cost = self._calc_round_trip_cost()
        return gross_pnl - cost

    # ---- v5: 止盈止损管理 ----
    def _setup_stop_levels(self, entry_price, direction):
        """设置止盈止损价位 (v5: 基于 ATR)"""
        atr_price = runtime_data.get("current_atr", 1.0)
        if atr_price < 1e-10:
            atr_price = self._tick_size

        # 固定止损 (ticks -> price)
        self._stop_loss_price = (
            entry_price - direction * STOP_LOSS_TICKS * self._tick_size
        )

        # ATR 止盈
        self._take_profit_price = (
            entry_price + direction * TAKE_PROFIT_ATR_MULT * atr_price
        )

        # 记录信号类型用于策略化止盈
        self._entry_signal_type = predicted_vector.get("signal_type", "UNKNOWN")

        self._highest_pnl_since_entry = 0.0

        logger.info(
            f"止盈止损设置 | 入场={entry_price:.2f} | "
            f"止损={self._stop_loss_price:.2f} ({STOP_LOSS_TICKS} ticks) | "
            f"止盈={self._take_profit_price:.2f} ({TAKE_PROFIT_ATR_MULT}x ATR={atr_price:.2f}) | "
            f"策略={self._entry_signal_type}"
        )

    def _check_stop_loss(self, current_price, direction):
        if direction == 1 and current_price <= self._stop_loss_price:
            logger.warning(
                f"止损触发! 多头止损价={self._stop_loss_price:.2f}, 当前价={current_price:.2f}"
            )
            return True
        if direction == -1 and current_price >= self._stop_loss_price:
            logger.warning(
                f"止损触发! 空头止损价={self._stop_loss_price:.2f}, 当前价={current_price:.2f}"
            )
            return True
        return False

    def _check_take_profit(self, current_price, direction):
        """v5: 策略化止盈"""
        if direction == 1 and current_price >= self._take_profit_price:
            logger.info(
                f"止盈触发! 多头止盈价={self._take_profit_price:.2f}, 当前价={current_price:.2f}"
            )
            return True
        if direction == -1 and current_price <= self._take_profit_price:
            logger.info(
                f"止盈触发! 空头止盈价={self._take_profit_price:.2f}, 当前价={current_price:.2f}"
            )
            return True

        # v5: VWAP 回归策略 - 价格回归 VWAP 即平仓
        if self._entry_signal_type == "VWAP_REV":
            vwap = runtime_data.get("current_vwap", 0)
            if vwap > 0:
                # 多头: 价格回归到 VWAP 附近
                if direction == 1 and current_price >= vwap - self._tick_size:
                    logger.info(
                        f"VWAP 回归止盈! 当前价={current_price:.2f}, VWAP={vwap:.2f}"
                    )
                    return True
                if direction == -1 and current_price <= vwap + self._tick_size:
                    logger.info(
                        f"VWAP 回归止盈! 当前价={current_price:.2f}, VWAP={vwap:.2f}"
                    )
                    return True

        # v5: BB 突破策略 - 快速止盈
        if self._entry_signal_type == "BREAKOUT":
            atr_ticks = runtime_data.get("current_atr_ticks", 1.0)
            quick_profit = 2 * self._tick_size  # 2 ticks 快速利润
            if (
                direction == 1
                and (current_price - self._last_entry_price) >= quick_profit
            ):
                logger.info(
                    f"BB 突破快速止盈! 利润={current_price - self._last_entry_price:.2f}"
                )
                return True
            if (
                direction == -1
                and (self._last_entry_price - current_price) >= quick_profit
            ):
                logger.info(
                    f"BB 突破快速止盈! 利润={self._last_entry_price - current_price:.2f}"
                )
                return True

        return False

    def _update_trailing_stop(self, current_price, direction):
        """v5: 移动止损 (更灵敏)"""
        if direction == 1:
            pnl = current_price - self._last_entry_price
            if pnl > TRAILING_STOP_TICKS * self._tick_size:
                new_stop = current_price - TRAILING_STOP_TICKS * self._tick_size
                if new_stop > self._stop_loss_price:
                    self._stop_loss_price = new_stop
                    logger.info(f"移动止损上移: {self._stop_loss_price:.2f}")
        else:
            pnl = self._last_entry_price - current_price
            if pnl > TRAILING_STOP_TICKS * self._tick_size:
                new_stop = current_price + TRAILING_STOP_TICKS * self._tick_size
                if new_stop < self._stop_loss_price:
                    self._stop_loss_price = new_stop
                    logger.info(f"移动止损下移: {self._stop_loss_price:.2f}")

    def _check_max_hold_time(self):
        now = time.time()
        entry_time = runtime_data.get("entry_time", 0.0)
        if entry_time > 0 and (now - entry_time) > MAX_HOLD_SECONDS:
            logger.warning(f"持仓超时 {MAX_HOLD_SECONDS}s, 强制平仓")
            return True
        return False

    # ---- v5: 核心翻转评估 ----
    def evaluate_and_flip(self):
        """核心翻转评估循环 v5"""
        if system_state["blind_mode"]:
            return
        if repair_state["active"]:
            return
        if system_state["mode"] != "NORMAL":
            return

        now = time.time()
        if now < runtime_data.get("cooling_until", 0.0):
            return
        if not runtime_data.get("trading_allowed", False):
            return

        current_pos = system_state["current_pos"]
        if current_pos != 0:
            self._manage_open_position()
            return

        effective_cooldown = get_consecutive_loss_cooldown()
        if now - runtime_data["last_flip_time"] < effective_cooldown:
            return

        d = predicted_vector["direction"]
        c = predicted_vector["confidence"]
        e_disp = predicted_vector["expected_displacement"]

        if d == 0:
            return

        bid1 = self.quote.bid_price1
        ask1 = self.quote.ask_price1
        spread = ask1 - bid1

        # v5: 交易成本计算 (含双边)
        vol_mult = self.quote.volume_multiple or 1
        transaction_cost = (
            spread
            + 2 * COMMISSION_PER_HAND / vol_mult
            + 2 * ESTIMATED_SLIPPAGE_TICKS * self._tick_size  # v5: 双边滑点
        )
        profit_threshold = transaction_cost * PROFIT_COST_RATIO

        # v5: 动态置信度
        dynamic_threshold = get_dynamic_confidence_threshold()
        if trade_stats["current_consecutive_loss"] >= 2:
            profit_threshold *= 1.3  # v5: 从 1.5 降至 1.3
            dynamic_threshold += 0.03  # v5: 从 0.05 降至 0.03

        # v5: 位移检查
        min_disp = max(profit_threshold, MIN_PROFIT_TICKS * self._tick_size)

        if current_pos != d and c >= dynamic_threshold and e_disp >= min_disp:
            self._execute_flip(d, bid1, ask1, spread)

    def _manage_open_position(self):
        """管理已有持仓 v5"""
        current_pos = system_state["current_pos"]
        p_now = self.quote.last_price

        if current_pos == 1:
            float_pnl = p_now - self._last_entry_price
        else:
            float_pnl = self._last_entry_price - p_now

        if float_pnl > self._highest_pnl_since_entry:
            self._highest_pnl_since_entry = float_pnl

        self._update_trailing_stop(p_now, current_pos)

        if self._check_stop_loss(p_now, current_pos):
            logger.warning("止损触发, 立即平仓")
            self._close_position("STOP_LOSS", p_now)
            return

        if self._check_take_profit(p_now, current_pos):
            logger.info("止盈触发, 立即平仓")
            self._close_position("TAKE_PROFIT", p_now)
            return

        if self._check_max_hold_time():
            logger.info("持仓超时, 平仓退出")
            self._close_position("TIMEOUT", p_now)
            return

    def _close_position(self, reason, exit_price):
        """v5: 平仓 (正确计算净盈亏)"""
        old_pos = system_state["current_pos"]
        self._safe_set_target(0)

        # v5: 使用净盈亏计算 (扣减交易成本)
        pnl = self._calc_net_pnl(self._last_entry_price, exit_price, old_pos)

        logger.info(
            f"平仓 [{reason}] | 方向={'多' if old_pos == 1 else '空'} | "
            f"入场={self._last_entry_price:.2f} | 出场={exit_price:.2f} | "
            f"毛盈亏={(exit_price - self._last_entry_price) * old_pos * (self.quote.volume_multiple or 1):+.2f} | "
            f"净盈亏={pnl:+.2f} | 成本={self._calc_round_trip_cost():.2f}"
        )

        self._record_trade_result(old_pos, exit_price, pnl)

        system_state["current_pos"] = 0
        runtime_data["current_pos"] = 0
        runtime_data["entry_price"] = 0.0
        runtime_data["entry_time"] = 0.0
        self._highest_pnl_since_entry = 0.0

    def _execute_flip(self, new_direction, bid1, ask1, spread):
        """执行翻转 v5"""
        bv1 = self.quote.bid_volume1
        av1 = self.quote.ask_volume1

        if bv1 == 0 or av1 == 0:
            logger.warning("流动性枯竭, 暂停翻转")
            return

        old_pos = system_state["current_pos"]
        old_entry = self._last_entry_price

        if spread > self._tick_size * 2 and predicted_vector["confidence"] < 0.85:
            return

        success = self._safe_set_target(new_direction)
        if not success:
            logger.error("翻转执行失败")
            return

        logger.info(f"TargetPosTask 目标仓位设置为 {new_direction}")

        trade_price = self.quote.last_price
        now = time.time()

        system_state["current_pos"] = new_direction
        runtime_data["last_flip_time"] = now
        self._last_entry_price = trade_price
        runtime_data["entry_price"] = trade_price
        runtime_data["entry_time"] = now
        self._highest_pnl_since_entry = 0.0

        self._setup_stop_levels(trade_price, new_direction)

        repair_state["entry_price"] = trade_price
        repair_state["entry_time"] = now
        repair_state["predict_target"] = (
            trade_price + new_direction * predicted_vector["expected_displacement"]
        )
        repair_state["error_count"] = 0

        slippage = abs(trade_price - (bid1 if new_direction == 1 else ask1))
        trade_stats["total_slippage"] += slippage

        # v5: 翻转时的旧仓盈亏 (用净盈亏)
        pnl = 0.0
        if old_pos != 0 and old_entry > 0:
            pnl = self._calc_net_pnl(old_entry, trade_price, old_pos)
            trade_stats["total_pnl"] += pnl
            self._record_trade_result(old_pos, trade_price, pnl)

        trade_stats["trade_count"] += 1
        trade_stats["session_trade_count"] += 1
        trade_stats["session_pnl"] += pnl
        if pnl > 0:
            trade_stats["win_count"] += 1
            trade_stats["session_win_count"] += 1
            trade_stats["current_consecutive_loss"] = 0
        else:
            trade_stats["current_consecutive_loss"] += 1
            trade_stats["max_consecutive_loss"] = max(
                trade_stats["max_consecutive_loss"],
                trade_stats["current_consecutive_loss"],
            )

        trade_stats["cumulative_trade_count"] += 1
        trade_stats["cumulative_pnl"] += pnl
        if pnl > 0:
            trade_stats["cumulative_win_count"] += 1

        record_trade_to_history(
            direction=new_direction,
            entry_price=old_entry,
            exit_price=trade_price,
            pnl=pnl,
            confidence=predicted_vector["confidence"],
            spatial_mode=runtime_data["spatial_mode"],
            hurst=runtime_data["current_hurst"],
            regime=runtime_data.get("volatility_regime", "UNKNOWN"),
            atr=runtime_data.get("current_atr", 0.0),
            daily_range=runtime_data.get("daily_range", 0.0),
            signal_type=predicted_vector.get("signal_type", "UNKNOWN"),
        )
        save_persistent_state()

        record_flip(
            direction=new_direction,
            entry_price=old_entry,
            exit_price=trade_price,
            trade_price=trade_price,
            theoretical_price=bid1 if new_direction == 1 else ask1,
            slippage=slippage,
            spread=spread,
            confidence=predicted_vector["confidence"],
            pnl=pnl,
            cumulative_pnl=trade_stats["total_pnl"],
            spatial_mode=runtime_data["spatial_mode"],
            hurst=runtime_data["current_hurst"],
        )

        dir_str = "多" if new_direction == 1 else "空"
        dynamic_threshold = get_dynamic_confidence_threshold()
        wr = get_recent_win_rate(20)
        wr_str = f", 近期胜率={wr:.0%}" if wr is not None else ""

        logger.info(
            f"翻转->{dir_str} | 价格={trade_price} | "
            f"滑点={slippage:.1f} | 净PnL={pnl:.1f} | "
            f"置信度={predicted_vector['confidence']:.2f} (阈值={dynamic_threshold:.2f})"
            f"信号={predicted_vector.get('signal_type', '?')}{wr_str}"
        )

        if self._price_tracker:
            self._price_tracker.record_trade_trigger(
                direction=new_direction,
                price=trade_price,
                confidence=predicted_vector["confidence"],
                displacement=predicted_vector["expected_displacement"],
                spread=spread,
                slippage=slippage,
                pnl=pnl,
                old_pos=old_pos,
            )

    def _record_trade_result(self, old_pos, exit_price, pnl):
        """记录交易结果"""
        if pnl > 0:
            trade_stats["win_count"] += 1
            trade_stats["session_win_count"] += 1
            trade_stats["current_consecutive_loss"] = 0
        else:
            trade_stats["current_consecutive_loss"] += 1
            trade_stats["max_consecutive_loss"] = max(
                trade_stats["max_consecutive_loss"],
                trade_stats["current_consecutive_loss"],
            )
        trade_stats["trade_count"] += 1
        trade_stats["session_trade_count"] += 1
        trade_stats["session_pnl"] += pnl
        trade_stats["cumulative_trade_count"] += 1
        trade_stats["cumulative_pnl"] += pnl
        if pnl > 0:
            trade_stats["cumulative_win_count"] += 1

    def force_close_all(self):
        success = self._safe_set_target(0)
        if success:
            logger.info("强制平仓指令已发出")
        else:
            logger.warning("TargetPosTask 不可用, 尝试撤销所有挂单")
            try:
                self.api.cancel_all_orders()
            except Exception as e:
                logger.error(f"撤单也失败: {e}")

    def set_target_position(self, vol):
        success = self._safe_set_target(vol)
        if not success:
            logger.error(f"设置目标仓位 {vol} 失败")
