"""
全局配置与共享数据结构 (v5 - 最终优化版)

回测验证结论:
- v4 原版: 0 笔交易 (REGIME=LOW 锁死 96.9% 时间)
- v5 优化: REGIME 解锁 91.6% 时间
- 最佳参数: 3 信号确认 + ZS=1.8 + SL=10 + TP=15
  -> 49.1% 胜率, 毛盈亏 +830 元, 成本 -848 元, 净盈亏 -18 元 (几乎打平)
- 核心发现: 策略毛盈利有效, 主要障碍是交易成本 (16元/笔)

优化要点:
1. ATR 用 tick 归一化 (解决锁死)
2. 3 信号确认 + ZS=1.8 (提高胜率到 49.1%)
3. SL=10, TP=15 (盈亏比 1.5:1)
4. BB 挤压突破 + VWAP 均值回归 (增加信号源)
5. 卡尔曼滤波 (降低噪音)
6. 盈亏计算修正 (双边成本)
"""

import collections
import os
import json
import sqlite3
import time
import logging
import numpy as np

logger = logging.getLogger("futures_trader.config")

SYMBOL = "DCE.eb2607"
TICK_BUFFER_SIZE = 500
EQUITY_FLOOR_RATIO = 0.85
MAX_REPAIR_DURATION = 60
POLY_DEGREE = 3
HURST_WINDOW = 200
SIGMA_THRESHOLD = 6
FLIP_COOLDOWN = 3.0
FLIP_GRACE_PERIOD = 10.0
DATA_STALE_THRESHOLD = 2.0
SPREAD_MAX_WAIT = 2.0
PRICE_STEP_DELAY = 0.3
COMMISSION_PER_HAND = 3.0
ESTIMATED_SLIPPAGE_TICKS = 1
POSITION_SYNC_INTERVAL = 5.0
PROFIT_COST_RATIO = 1.5
MIN_PROFIT_TICKS = 3

# v5: 波动率 regime (tick 归一化)
MIN_DAILY_RANGE_TICKS = 15
ATR_PERIOD = 60
VOLATILITY_EXPANSION_RATIO = 1.3
ATR_TICK_THRESHOLD = 0.5

# v5: BB 参数
BB_PERIOD = 60
BB_STD_MULT = 2.0
BB_SQUEEZE_WIDTH = 0.004
BB_BREAKOUT_CONFIRM = 3

# v5: RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# v5: Z-score
ZSCORE_ENTRY = 1.8  # 回测最优

# v5: 止盈止损 (回测最优) - v6 优化：降低门槛提高交易频率
STOP_LOSS_TICKS = 8       # v6: 从 10 降至 8，减少被扫损概率
TRAILING_STOP_TICKS = 4   # v6: 从 5 降至 4，更灵敏锁定利润
TAKE_PROFIT_ATR_MULT = 1.2  # v6: 从 1.5 降至 1.2，更快止盈
MIN_HOLD_SECONDS = 5      # v6: 从 10 降至 5，减少持仓时间风险
MAX_HOLD_SECONDS = 180    # v6: 从 240 降至 180

# v6 新增：动态位移系数
DISPLACEMENT_SCALE_FACTOR = 0.6  # v6: 降低位移要求到原来的 60%

# v5: 置信度
CONFIDENCE_THRESHOLD = 0.50
DYNAMIC_CONFIDENCE_WINDOW = 20
DYNAMIC_CONFIDENCE_MIN = 0.40
DYNAMIC_CONFIDENCE_MAX = 0.75
DYNAMIC_CONFIDENCE_SCALE = 0.15

# v5: 连亏保护
CONSECUTIVE_LOSS_COOLDOWN_ADD = 1.5
MAX_CONSECUTIVE_LOSS_COOLDOWN = 10.0
CONSECUTIVE_LOSS_THRESHOLD = 3
COOLING_PERIOD = 30.0

REPAIR_SIGMA_MULTIPLIER = 3.0
REPAIR_MIN_HOLD_TIME = 8.0
REPAIR_ERROR_DECAY_RATE = 0.03

VWAP_DEVIATION_TICKS = 10
VWAP_LOOKBACK_TICKS = 200

KALMAN_PROCESS_NOISE = 0.0001
KALMAN_MEASUREMENT_NOISE = 1.0

PRICE_TRACKER_LOG = "price_tracker.log"
PRICE_TRACKER_INTERVAL = 1.0
TQ_USER = "19224304249"
TQ_PASSWORD = "tankai520"
TQ_KQ_NUMBER = 0

TRADING_SESSIONS = [
    ("09:01", "11:29"),
    ("13:31", "14:59"),
    ("21:01", "22:59"),
]

STATE_DB_PATH = os.path.join(os.path.dirname(__file__), "system_state.db")


def _get_state_db():
    conn = sqlite3.connect(STATE_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS persistent_state (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trade_history (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, direction INTEGER, entry_price REAL, exit_price REAL, pnl REAL, confidence REAL, spatial_mode TEXT, hurst REAL, regime TEXT, atr REAL, daily_range REAL, signal_type TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_stats (session_date TEXT PRIMARY KEY, initial_equity REAL, final_equity REAL, total_trades INTEGER, win_count INTEGER, total_pnl REAL, total_slippage REAL, repair_count INTEGER, recalibration_count INTEGER, daily_range REAL, avg_atr REAL, volatility_regime TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS daily_volatility (date TEXT PRIMARY KEY, high REAL, low REAL, daily_range REAL, avg_atr REAL, total_volume REAL, trades_taken INTEGER, trades_skipped INTEGER, skip_reason TEXT)"
    )
    conn.commit()
    return conn


def save_state(key, value):
    try:
        conn = _get_state_db()
        conn.execute(
            "INSERT OR REPLACE INTO persistent_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"保存状态失败 [{key}]: {e}")


def load_state(key, default=None):
    try:
        conn = _get_state_db()
        cursor = conn.execute(
            "SELECT value FROM persistent_state WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return default
    except Exception as e:
        logger.error(f"加载状态失败 [{key}]: {e}")
        return default


def record_trade_to_history(
    direction,
    entry_price,
    exit_price,
    pnl,
    confidence,
    spatial_mode,
    hurst,
    regime="UNKNOWN",
    atr=0.0,
    daily_range=0.0,
    signal_type="UNKNOWN",
):
    try:
        conn = _get_state_db()
        conn.execute(
            "INSERT INTO trade_history (timestamp, direction, entry_price, exit_price, pnl, confidence, spatial_mode, hurst, regime, atr, daily_range, signal_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                direction,
                entry_price,
                exit_price,
                pnl,
                confidence,
                spatial_mode,
                hurst,
                regime,
                atr,
                daily_range,
                signal_type,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"记录交易历史失败: {e}")


def get_recent_trade_results(n=20):
    try:
        conn = _get_state_db()
        cursor = conn.execute(
            "SELECT pnl FROM trade_history ORDER BY timestamp DESC LIMIT ?", (n,)
        )
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        results.reverse()
        return results
    except Exception as e:
        logger.error(f"获取近期交易结果失败: {e}")
        return []


def get_recent_win_rate(n=20):
    results = get_recent_trade_results(n)
    if not results:
        return None
    return sum(1 for r in results if r > 0) / len(results)


tick_buffer = collections.deque(maxlen=TICK_BUFFER_SIZE)
predicted_vector = {
    "direction": 0,
    "confidence": 0.0,
    "expected_displacement": 0.0,
    "predicted_price": 0.0,
    "signal_time": 0.0,
    "signal_type": "NONE",
}
system_state = {"mode": "NORMAL", "current_pos": 0, "blind_mode": False}
repair_state = {
    "active": False,
    "start_time": None,
    "hedge_pos": 0,
    "error_count": 0,
    "entry_price": 0.0,
    "predict_target": 0.0,
    "entry_time": 0.0,
    "highest_pnl": 0.0,
}
runtime_data = {
    "last_data_time": 0.0,
    "initial_equity": 0.0,
    "va_high": 0.0,
    "va_low": 0.0,
    "prev_high": 0.0,
    "prev_low": 0.0,
    "prev_close": 0.0,
    "today_open": 0.0,
    "today_high": 0.0,
    "today_low": 999999.0,
    "last_volume": 0,
    "aggressive_buy_vol": 0,
    "aggressive_sell_vol": 0,
    "last_flip_time": 0.0,
    "anomaly_suppress_until": 0.0,
    "spatial_mode": "MOMENTUM",
    "current_hurst": 0.5,
    "current_atr": 0.0,
    "current_atr_ticks": 0.0,
    "current_rsi": 50.0,
    "current_zscore": 0.0,
    "bb_upper": 0.0,
    "bb_lower": 0.0,
    "bb_middle": 0.0,
    "bb_squeeze": False,
    "bb_squeeze_active": False,
    "bb_squeeze_start": 0.0,
    "volatility_regime": "LOW",
    "daily_range": 0.0,
    "trading_allowed": False,
    "account_balance": 0.0,
    "available_funds": 0.0,
    "float_profit": 0.0,
    "margin_used": 0.0,
    "dynamic_equity": 0.0,
    "cooling_until": 0.0,
    "session_start_equity": 0.0,
    "entry_price": 0.0,
    "entry_time": 0.0,
    "highest_pnl_since_entry": 0.0,
    "current_vwap": 0.0,
    "vwap_deviation": 0.0,
    "kalman_price": 0.0,
    "tick_size": 1.0,
}
trade_stats = {
    "trade_count": 0,
    "win_count": 0,
    "total_slippage": 0.0,
    "total_pnl": 0.0,
    "max_consecutive_loss": 0,
    "current_consecutive_loss": 0,
    "repair_count": 0,
    "recalibration_count": 0,
    "cumulative_trade_count": 0,
    "cumulative_win_count": 0,
    "cumulative_pnl": 0.0,
    "session_trade_count": 0,
    "session_win_count": 0,
    "session_pnl": 0.0,
    "trades_skipped_today": 0,
}


def load_persistent_state():
    global trade_stats
    saved = load_state("trade_stats")
    if saved:
        trade_stats["cumulative_trade_count"] = saved.get("cumulative_trade_count", 0)
        trade_stats["cumulative_win_count"] = saved.get("cumulative_win_count", 0)
        trade_stats["cumulative_pnl"] = saved.get("cumulative_pnl", 0.0)
    logger.info(
        f"加载持久化状态: 累计交易={trade_stats['cumulative_trade_count']}, 累计盈亏={trade_stats['cumulative_pnl']:.2f}"
    )


def save_persistent_state():
    save_state(
        "trade_stats",
        {
            "cumulative_trade_count": trade_stats["cumulative_trade_count"],
            "cumulative_win_count": trade_stats["cumulative_win_count"],
            "cumulative_pnl": trade_stats["cumulative_pnl"],
        },
    )


def get_dynamic_confidence_threshold():
    win_rate = get_recent_win_rate(DYNAMIC_CONFIDENCE_WINDOW)
    if win_rate is None:
        return CONFIDENCE_THRESHOLD
    adjustment = (0.5 - win_rate) * DYNAMIC_CONFIDENCE_SCALE * 2
    dynamic = CONFIDENCE_THRESHOLD + adjustment
    return max(DYNAMIC_CONFIDENCE_MIN, min(DYNAMIC_CONFIDENCE_MAX, dynamic))


def get_consecutive_loss_cooldown():
    n = trade_stats["current_consecutive_loss"]
    if n <= 1:
        return FLIP_COOLDOWN
    extra = min(n * CONSECUTIVE_LOSS_COOLDOWN_ADD, MAX_CONSECUTIVE_LOSS_COOLDOWN)
    return FLIP_COOLDOWN + extra


def compute_atr(prices, period=None, tick_size=1.0):
    """v5: ATR 以 tick 为单位"""
    if period is None:
        period = ATR_PERIOD
    if len(prices) < period + 1:
        return 1.0
    return float(np.mean(np.abs(np.diff(prices[-(period + 1) :])))) / tick_size


def compute_atr_price(prices, period=None):
    """返回以价格为单位的 ATR"""
    if period is None:
        period = ATR_PERIOD
    if len(prices) < period + 1:
        return 1.0
    return float(np.mean(np.abs(np.diff(prices[-(period + 1) :]))))


def compute_rsi(prices, period=None):
    """Wilder's RSI"""
    if period is None:
        period = RSI_PERIOD
    if len(prices) < period + 1:
        return 50.0
    diffs = np.diff(prices[-(period + 1) :])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain, avg_loss = np.mean(gains), np.mean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


def compute_bollinger(prices, period=None, std_mult=None):
    if period is None:
        period = BB_PERIOD
    if std_mult is None:
        std_mult = BB_STD_MULT
    if len(prices) < period:
        return 0.0, 0.0, 0.0
    recent = prices[-period:]
    middle = float(np.mean(recent))
    std = float(np.std(recent))
    return middle + std_mult * std, middle, middle - std_mult * std


def compute_zscore(prices, period=None):
    if period is None:
        period = BB_PERIOD
    if len(prices) < period:
        return 0.0
    recent = prices[-period:]
    mean, std = np.mean(recent), np.std(recent)
    if std < 1e-10:
        return 0.0
    return float((prices[-1] - mean) / std)


class KalmanFilter:
    """v5: 卡尔曼滤波器"""

    def __init__(self, process_noise=0.0001, measurement_noise=1.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.x = None
        self.P = 1.0
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.P = 1.0
            self.initialized = True
            return measurement
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x

    def reset(self):
        self.initialized = False
        self.x = None
        self.P = 1.0
