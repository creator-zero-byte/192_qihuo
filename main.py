"""
全自动期货短线交易工具 - 主入口 (v5 - 实战优化版)
核心改进:
1. 波动率 regime 过滤修复: tick 归一化 ATR, 解决锁死问题
2. 多信号融合预测: z-score + BB + RSI + 动量 + 成交量 + VWAP回归 + BB突破
3. 动态止盈止损: 基于 ATR 自适应 + 策略化止盈
4. 卡尔曼滤波: 降低价格噪音
5. 盈亏计算修正: 正确计入双边手续费+滑点
"""

import time
import logging
import sys
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("futures_trader.main")

import config
from config import (
    SYMBOL,
    TRADING_SESSIONS,
    system_state,
    repair_state,
    runtime_data,
    trade_stats,
    load_persistent_state,
    save_persistent_state,
    get_recent_win_rate,
    compute_atr,
    compute_rsi,
    compute_bollinger,
    compute_zscore,
    MIN_DAILY_RANGE_TICKS,
)
from database import init_db, record_daily_report, get_today_flips
from data_collector import DataCollector
from prediction_engine import PredictionEngine
from executor import Executor
from defense import DefenseModule
from repair import RepairModule
from dashboard import Dashboard
from price_tracker import PriceTracker


def is_trading_time():
    now = datetime.now()
    current = now.strftime("%H:%M")
    for start, end in TRADING_SESSIONS:
        if start <= current <= end:
            return True
    return False


def is_near_close():
    now = datetime.now()
    current = now.strftime("%H:%M")
    close_times = ["11:29", "14:59", "22:59"]
    for ct in close_times:
        h, m = int(ct[:2]), int(ct[3:])
        if m > 0:
            warn_time = f"{h:02d}:{m - 1:02d}"
        else:
            warn_time = f"{h - 1:02d}:59"
        if current >= warn_time and current <= ct:
            return True
    return False


def is_near_open():
    now = datetime.now()
    current = now.strftime("%H:%M")
    open_times = ["09:00", "13:30", "21:00"]
    for ot in open_times:
        h, m = int(ot[:2]), int(ot[3:])
        if current >= ot and current <= f"{h:02d}:{m + 1:02d}":
            return True
    return False


def compute_value_area(klines):
    try:
        closes = klines.close.values
        volumes = klines.volume.values
        highs = klines.high.values
        lows = klines.low.values
        if len(closes) < 10:
            return 0, 0, 0, 0, 0

        n = min(240, len(closes))
        c, v, h, l = closes[-n:], volumes[-n:], highs[-n:], lows[-n:]
        total_vol = np.sum(v)
        if total_vol == 0:
            return 0, 0, 0, 0, 0

        vwap = np.sum(c * v) / total_vol
        price_min, price_max = np.min(l), np.max(h)
        if price_max == price_min:
            return price_min, price_max, price_max, price_min, c[-1]

        n_bins = 100
        bins = np.linspace(price_min, price_max, n_bins + 1)
        vol_profile = np.zeros(n_bins)
        for i in range(len(c)):
            bin_idx = int((c[i] - price_min) / (price_max - price_min) * (n_bins - 1))
            bin_idx = max(0, min(n_bins - 1, bin_idx))
            vol_profile[bin_idx] += v[i]

        poc_idx = np.argmax(vol_profile)
        covered = vol_profile[poc_idx]
        lo_idx, hi_idx = poc_idx, poc_idx
        while covered < total_vol * 0.7:
            expand_lo = vol_profile[lo_idx - 1] if lo_idx > 0 else 0
            expand_hi = vol_profile[hi_idx + 1] if hi_idx < n_bins - 1 else 0
            if expand_lo >= expand_hi and lo_idx > 0:
                lo_idx -= 1
                covered += vol_profile[lo_idx]
            elif hi_idx < n_bins - 1:
                hi_idx += 1
                covered += vol_profile[hi_idx]
            else:
                break

        va_low = bins[lo_idx]
        va_high = bins[hi_idx + 1] if hi_idx + 1 < len(bins) else bins[-1]
        return va_high, va_low, float(np.max(h)), float(np.min(l)), float(c[-1])
    except Exception as e:
        logger.error(f"计算 Value Area 异常: {e}")
        return 0, 0, 0, 0, 0


def daily_summary():
    flips = get_today_flips()
    win_rate = get_recent_win_rate(20)
    wr_str = f"{win_rate:.1%}" if win_rate is not None else "N/A"

    tc = trade_stats["session_trade_count"]
    wc = trade_stats["session_win_count"]
    wr = (wc / tc * 100) if tc > 0 else 0
    avg_slip = (trade_stats["total_slippage"] / tc) if tc > 0 else 0
    total_pnl = trade_stats["session_pnl"]

    today_range = runtime_data.get("today_high", 0) - runtime_data.get(
        "today_low", 999999
    )
    today_range = max(0, today_range)
    avg_atr = runtime_data.get("current_atr", 0)

    date_str = datetime.now().strftime("%Y-%m-%d")
    record_daily_report(
        date_str=date_str,
        total_pnl=total_pnl,
        total_trades=tc,
        win_rate=wr,
        avg_slippage=avg_slip,
        max_drawdown=0,
        prediction_accuracy=wr,
        repair_count=trade_stats["repair_count"],
        recalibration_count=trade_stats["recalibration_count"],
    )

    save_persistent_state()

    logger.info(
        f"收盘自检 | 交易={tc}次 | 胜率={wr:.1f}% | "
        f"净盈亏={total_pnl:.2f}元 | 日波动={today_range:.0f} ticks | "
        f"ATR={avg_atr:.2f} | 近期胜率(20笔)={wr_str}"
    )


def main():
    logger.info("=" * 60)
    logger.info("全自动期货短线交易工具 v5 启动")
    logger.info(f"交易品种: {SYMBOL}")
    logger.info("核心改进: Tick归一化ATR + BB挤压突破 + VWAP回归 + 卡尔曼滤波")
    logger.info("=" * 60)

    init_db()
    load_persistent_state()

    dash = Dashboard(config)
    dash.start_server(port=8051)

    tracker = PriceTracker()

    try:
        from tqsdk import TqApi, TqKq, TqAuth

        logger.info(f"正在连接快期模拟 - 账号: '{config.TQ_USER}'")
        api = TqApi(TqKq(), auth=TqAuth(config.TQ_USER, config.TQ_PASSWORD))
        logger.info("天勤 API (快期模拟) 连接成功")
    except Exception as e:
        logger.critical(f"天勤 API 连接失败: {e}")
        tracker.close()
        sys.exit(1)

    quote = api.get_quote(SYMBOL)
    klines_1m = api.get_kline_serial(SYMBOL, 60)
    position = api.get_position(SYMBOL)
    account = api.get_account()

    api.wait_update()
    initial_eq = account.balance
    runtime_data["initial_equity"] = initial_eq
    runtime_data["session_start_equity"] = initial_eq
    logger.info(f"初始权益: {initial_eq:.2f}")

    va_high, va_low, prev_high, prev_low, prev_close = compute_value_area(klines_1m)
    runtime_data["va_high"] = va_high
    runtime_data["va_low"] = va_low
    runtime_data["prev_high"] = prev_high
    runtime_data["prev_low"] = prev_low
    runtime_data["prev_close"] = prev_close
    logger.info(
        f"前日 VA: [{va_low:.2f}, {va_high:.2f}], H/L/C: {prev_high:.2f}/{prev_low:.2f}/{prev_close:.2f}"
    )

    collector = DataCollector(api, quote)
    engine = PredictionEngine()
    executor = Executor(api, quote)
    executor.init_tick_size()
    executor.set_price_tracker(tracker)
    defense = DefenseModule(api, quote, position, account, executor)
    repair_mod = RepairModule(api, quote, account, engine, executor)

    last_freshness_check = 0.0
    last_dash_push = 0.0
    last_state_save = 0.0
    session_closed = False
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10

    logger.info("进入主循环")
    logger.info("-" * 60)

    try:
        while True:
            try:
                api.wait_update()
                consecutive_errors = 0
            except Exception as e:
                err_msg = str(e)
                if "遇到错单" in err_msg or "TargetPosTask" in err_msg:
                    consecutive_errors += 1
                    logger.error(
                        f"TargetPosTask 错单异常 (第{consecutive_errors}次): {e}"
                    )
                    executor.handle_bad_order_exception(e)
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.critical(
                            f"连续错单达到{MAX_CONSECUTIVE_ERRORS}次, 系统自动停止"
                        )
                        system_state["mode"] = "SHUTDOWN"
                    continue
                else:
                    raise

            now = time.time()

            try:
                if api.is_changing(quote):
                    collector.on_tick(quote)
            except Exception as e:
                logger.error(f"数据采集异常: {e}")

            if now - last_freshness_check > 0.1:
                try:
                    collector.check_freshness()
                except Exception as e:
                    logger.error(f"数据新鲜度检查异常: {e}")
                last_freshness_check = now

            try:
                defense.check_all()
            except Exception as e:
                logger.error(f"防御检查异常: {e}")

            if not is_trading_time():
                if not session_closed and system_state["current_pos"] != 0:
                    logger.info("非交易时段, 清理持仓")
                    executor.force_close_all()
                    session_closed = True
                if now - last_dash_push > 0.1:
                    dash.push_data()
                    last_dash_push = now
                continue
            else:
                session_closed = False

            if is_near_close():
                if system_state["current_pos"] != 0:
                    logger.info("收盘前1分钟, 强制平仓")
                    executor.force_close_all()
                    system_state["current_pos"] = 0
                continue

            if is_near_open():
                continue

            if system_state["mode"] == "NORMAL":
                try:
                    engine.update()
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"预测引擎异常 (第{consecutive_errors}次): {e}")
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.critical(
                            f"连续异常达到{MAX_CONSECUTIVE_ERRORS}次, 系统熔断"
                        )
                        system_state["mode"] = "SHUTDOWN"
                try:
                    executor.evaluate_and_flip()
                except Exception as e:
                    logger.error(f"执行器异常: {e}")
                try:
                    repair_mod.check_deviation()
                except Exception as e:
                    logger.error(f"修复模块异常: {e}")

            elif system_state["mode"] == "REPAIR":
                try:
                    engine.update()
                except Exception as e:
                    logger.error(f"预测引擎异常(REPAIR模式): {e}")
                try:
                    repair_mod.run()
                except Exception as e:
                    logger.error(f"修复模块运行异常: {e}")

            elif system_state["mode"] == "SHUTDOWN":
                logger.info("系统熔断中, 等待恢复...")
                time.sleep(60)
                try:
                    engine.recalibrate()
                except Exception as e:
                    logger.error(f"模型自愈异常: {e}")
                system_state["mode"] = "NORMAL"
                consecutive_errors = 0

            try:
                runtime_data["account_balance"] = account.balance
                runtime_data["available_funds"] = account.available
                runtime_data["float_profit"] = account.float_profit
                runtime_data["margin_used"] = account.margin
                runtime_data["dynamic_equity"] = account.balance + account.float_profit
            except Exception:
                pass

            tracker.update()

            if now - last_state_save > 300:
                save_persistent_state()
                last_state_save = now

            if now - last_dash_push > 0.1:
                dash.push_data()
                last_dash_push = now

    except KeyboardInterrupt:
        logger.info("收到退出信号")
    except Exception as e:
        logger.critical(f"主循环异常: {e}", exc_info=True)
    finally:
        logger.info("执行收盘清理...")
        try:
            executor.force_close_all()
            api.cancel_all_orders()
        except Exception:
            pass

        daily_summary()
        save_persistent_state()
        tracker.close()

        try:
            api.close()
        except Exception:
            pass

        logger.info("系统已安全退出")


if __name__ == "__main__":
    main()
