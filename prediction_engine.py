"""
协程B: 多信号融合预测引擎 (v5 - 实战优化版)
核心改进:
1. 修复 ATR 单位: tick 归一化, 解决 REGIME=LOW 锁死问题
2. BB 挤压突破策略: squeeze 期间等待, expansion 时入场
3. VWAP 均值回归信号: 价格偏离 VWAP 过多时回归
4. 卡尔曼滤波: 降低价格噪音, 提高信号质量
5. 信号融合: 2 个确认即可 (原为 3 个)
6. 优化 RSI: 使用 Wilder's 平滑算法
7. 降低 Z-score 阈值: 1.5 (原为 2.0)
8. 优化预期位移计算: 基于实际波动率
"""

import time
import math
import logging
import numpy as np

from config import (
    tick_buffer,
    predicted_vector,
    system_state,
    runtime_data,
    POLY_DEGREE,
    HURST_WINDOW,
    SIGMA_THRESHOLD,
    BB_PERIOD,
    BB_STD_MULT,
    BB_SQUEEZE_WIDTH,
    BB_BREAKOUT_CONFIRM,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    ZSCORE_ENTRY,
    MIN_DAILY_RANGE_TICKS,
    ATR_PERIOD,
    ATR_TICK_THRESHOLD,
    VOLATILITY_EXPANSION_RATIO,
    STOP_LOSS_TICKS,
    TRAILING_STOP_TICKS,
    TAKE_PROFIT_ATR_MULT,
    MIN_HOLD_SECONDS,
    MAX_HOLD_SECONDS,
    VWAP_DEVIATION_TICKS,
    VWAP_LOOKBACK_TICKS,
    get_dynamic_confidence_threshold,
    get_consecutive_loss_cooldown,
    get_recent_win_rate,
    load_state,
    save_state,
    compute_atr,
    compute_atr_price,
    compute_rsi,
    compute_bollinger,
    compute_zscore,
    KalmanFilter,
)

logger = logging.getLogger("futures_trader.prediction")

# ============ Numba 加速 ============
try:
    from numba import njit

    NUMBA_AVAILABLE = True
    logger.info("Numba JIT 加速已启用")

    @njit(cache=True)
    def _weighted_polyfit(t, prices, weights, degree):
        n = len(t)
        cols = degree + 1
        A = np.zeros((cols, cols), dtype=np.float64)
        b = np.zeros(cols, dtype=np.float64)
        for k in range(n):
            w = weights[k]
            for i in range(cols):
                ti = t[k] ** (degree - i)
                for j in range(i, cols):
                    tj = t[k] ** (degree - j)
                    A[i, j] += w * ti * tj
                b[i] += w * ti * prices[k]
        for i in range(cols):
            for j in range(i + 1, cols):
                A[j, i] = A[i, j]
        Ab = np.zeros((cols, cols + 1), dtype=np.float64)
        for i in range(cols):
            for j in range(cols):
                Ab[i, j] = A[i, j]
            Ab[i, cols] = b[i]
        for i in range(cols):
            max_row = i
            max_val = abs(Ab[i, i])
            for k in range(i + 1, cols):
                if abs(Ab[k, i]) > max_val:
                    max_val = abs(Ab[k, i])
                    max_row = k
            if max_row != i:
                for j in range(cols + 1):
                    tmp = Ab[i, j]
                    Ab[i, j] = Ab[max_row, j]
                    Ab[max_row, j] = tmp
            pivot = Ab[i, i]
            if abs(pivot) < 1e-15:
                continue
            for k in range(i + 1, cols):
                factor = Ab[k, i] / pivot
                for j in range(i, cols + 1):
                    Ab[k, j] -= factor * Ab[i, j]
        x = np.zeros(cols, dtype=np.float64)
        for i in range(cols - 1, -1, -1):
            s = Ab[i, cols]
            for j in range(i + 1, cols):
                s -= Ab[i, j] * x[j]
            if abs(Ab[i, i]) < 1e-15:
                x[i] = 0.0
            else:
                x[i] = s / Ab[i, i]
        return x

    @njit(cache=True)
    def _evaluate_poly(coeffs, degree, t_eval):
        val = 0.0
        deriv1 = 0.0
        deriv2 = 0.0
        for i in range(degree + 1):
            power = degree - i
            val += coeffs[i] * (t_eval**power)
            if power > 0:
                deriv1 += coeffs[i] * power * (t_eval ** (power - 1))
            if power > 1:
                deriv2 += coeffs[i] * power * (power - 1) * (t_eval ** (power - 2))
        return val, deriv1, deriv2

    @njit(cache=True)
    def _hurst_rs(log_returns, max_k):
        n = len(log_returns)
        if n < 20:
            return 0.5
        rs_list = np.empty(max_k)
        ns_list = np.empty(max_k)
        count = 0
        for k in range(10, min(n // 2, max_k + 10)):
            sub_len = n // k
            if sub_len < 2:
                break
            rs_sum = 0.0
            valid = 0
            for i in range(k):
                start = i * sub_len
                end = start + sub_len
                sub = log_returns[start:end]
                mean_s, std_s = np.mean(sub), np.std(sub)
                if std_s == 0:
                    continue
                cumdev = np.cumsum(sub - mean_s)
                r = np.max(cumdev) - np.min(cumdev)
                rs_sum += r / std_s
                valid += 1
            if valid > 0:
                rs_list[count] = np.log(rs_sum / valid)
                ns_list[count] = np.log(sub_len)
                count += 1
        if count < 2:
            return 0.5
        x, y = ns_list[:count], rs_list[:count]
        mx, my = np.mean(x), np.mean(y)
        num = np.sum((x - mx) * (y - my))
        den = np.sum((x - mx) ** 2)
        if den == 0:
            return 0.5
        return num / den

except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba 不可用, 使用纯 Python 实现")

    def _weighted_polyfit(t, prices, weights, degree):
        n = len(t)
        cols = degree + 1
        V = np.zeros((n, cols))
        for k in range(n):
            for j in range(cols):
                V[k, j] = t[k] ** (degree - j)
        W = np.diag(weights)
        A = V.T @ W @ V
        b_vec = V.T @ W @ prices
        try:
            return np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            return np.polyfit(t, prices, degree)[::-1]

    def _evaluate_poly(coeffs, degree, t_eval):
        poly = np.poly1d(coeffs[::-1])
        return (
            float(poly(t_eval)),
            float(poly.deriv(1)(t_eval)),
            float(poly.deriv(2)(t_eval)),
        )

    def _hurst_rs(log_returns, max_k):
        n = len(log_returns)
        if n < 20:
            return 0.5
        rs_vals, ns_vals = [], []
        for k in range(10, min(n // 2, max_k + 10)):
            sub_len = n // k
            if sub_len < 2:
                break
            rs_sum, valid = 0.0, 0
            for i in range(k):
                sub = log_returns[i * sub_len : (i + 1) * sub_len]
                mean_s, std_s = np.mean(sub), np.std(sub)
                if std_s == 0:
                    continue
                cumdev = np.cumsum(sub - mean_s)
                rs_sum += (np.max(cumdev) - np.min(cumdev)) / std_s
                valid += 1
            if valid > 0:
                rs_vals.append(math.log(rs_sum / valid))
                ns_vals.append(math.log(sub_len))
        if len(rs_vals) < 2:
            return 0.5
        x, y = np.array(ns_vals), np.array(rs_vals)
        den = np.sum((x - np.mean(x)) ** 2)
        if den == 0:
            return 0.5
        return np.sum((x - np.mean(x)) * (y - np.mean(y))) / den


class PredictionEngine:
    """多信号融合预测引擎 v5"""

    def __init__(self):
        self.poly_degree = POLY_DEGREE
        self._anomaly_until = 0.0
        self._last_meaningful_update = 0.0
        self._signal_history = []
        self._bb_squeeze_detected = False
        self._squeeze_start_time = 0.0
        self._kalman = KalmanFilter()
        self._last_kalman_reset = 0.0

    # ---- v5: 波动率 regime 检测 (修复版) ----
    def detect_volatility_regime(self, prices, tick_size=1.0):
        """v5: 使用 tick 归一化的 ATR, 解决锁死问题"""
        if len(prices) < ATR_PERIOD + 10:
            return "LOW", 0.0

        # ATR 以 tick 为单位
        atr_ticks = compute_atr(prices, ATR_PERIOD, tick_size)
        atr_price = compute_atr_price(prices, ATR_PERIOD)
        runtime_data["current_atr"] = atr_price  # 价格单位 (用于止损计算)
        runtime_data["current_atr_ticks"] = atr_ticks  # tick 单位 (用于 regime 判断)

        # 平均 ATR
        avg_atr = (
            compute_atr(prices, ATR_PERIOD * 3, tick_size)
            if len(prices) >= ATR_PERIOD * 3 + 1
            else atr_ticks
        )
        if avg_atr < 1e-10:
            avg_atr = atr_ticks

        ratio = atr_ticks / avg_atr if avg_atr > 0 else 1.0

        # 日波动范围 (ticks)
        if len(prices) > 100:
            daily_range = (np.max(prices[-100:]) - np.min(prices[-100:])) / tick_size
            runtime_data["daily_range"] = daily_range
        else:
            daily_range = 0.0

        # v5: 使用 tick 归一化的阈值
        if atr_ticks < ATR_TICK_THRESHOLD and daily_range < MIN_DAILY_RANGE_TICKS:
            regime = "LOW"
            runtime_data["trading_allowed"] = False
        elif ratio > VOLATILITY_EXPANSION_RATIO:
            regime = "HIGH"
            runtime_data["trading_allowed"] = True
        elif daily_range >= MIN_DAILY_RANGE_TICKS:
            # v5: 日波动足够就允许交易 (即使 ATR 偏低)
            regime = "NORMAL"
            runtime_data["trading_allowed"] = True
        else:
            regime = "LOW"
            runtime_data["trading_allowed"] = False

        runtime_data["volatility_regime"] = regime

        # BB 挤压检测
        if len(prices) >= BB_PERIOD * 2:
            squeeze = self._check_bb_squeeze(prices)
            runtime_data["bb_squeeze"] = squeeze
            if squeeze and not self._bb_squeeze_detected:
                self._bb_squeeze_detected = True
                self._squeeze_start_time = time.time()
                runtime_data["bb_squeeze_active"] = True
                runtime_data["bb_squeeze_start"] = self._squeeze_start_time
                logger.info("BB 挤压检测: 波动率收缩, 准备突破")
            elif not squeeze:
                if self._bb_squeeze_detected:
                    logger.info("BB 挤压结束, 波动率扩张")
                self._bb_squeeze_detected = False
                runtime_data["bb_squeeze_active"] = False

        return regime, atr_price

    def _check_bb_squeeze(self, prices):
        """BB 挤压检测"""
        upper, middle, lower = compute_bollinger(prices, BB_PERIOD, BB_STD_MULT)
        price = prices[-1]
        if price < 1e-10:
            return False
        width_ratio = (upper - lower) / price
        return width_ratio < BB_SQUEEZE_WIDTH

    # ---- v5: z-score 均值回归信号 ----
    def zscore_signal(self, prices):
        if len(prices) < BB_PERIOD:
            return 0, 0.0
        zscore = compute_zscore(prices, BB_PERIOD)
        runtime_data["current_zscore"] = zscore
        upper, middle, lower = compute_bollinger(prices, BB_PERIOD, BB_STD_MULT)
        runtime_data["bb_upper"] = upper
        runtime_data["bb_middle"] = middle
        runtime_data["bb_lower"] = lower

        direction = 0
        confidence = 0.0

        if zscore < -ZSCORE_ENTRY:
            direction = 1  # 超卖做多
            confidence = min(0.85, abs(zscore) / (ZSCORE_ENTRY * 2.5))
        elif zscore > ZSCORE_ENTRY:
            direction = -1  # 超买做空
            confidence = min(0.85, abs(zscore) / (ZSCORE_ENTRY * 2.5))

        # BB 确认: 价格必须触及或穿越轨道
        if direction == 1 and prices[-1] > lower:
            confidence *= 0.5
        elif direction == -1 and prices[-1] < upper:
            confidence *= 0.5

        return direction, confidence

    # ---- v5: RSI 信号 ----
    def rsi_signal(self, prices):
        if len(prices) < RSI_PERIOD + 5:
            return 0, 0.0
        rsi = compute_rsi(prices, RSI_PERIOD)
        runtime_data["current_rsi"] = rsi

        direction = 0
        confidence = 0.0

        if rsi < RSI_OVERSOLD:
            direction = 1
            confidence = (RSI_OVERSOLD - rsi) / RSI_OVERSOLD * 0.65
        elif rsi > RSI_OVERBOUGHT:
            direction = -1
            confidence = (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT) * 0.65

        return direction, confidence

    # ---- v5: 动量信号 ----
    def momentum_signal(self, prices):
        n = min(150, len(prices))
        if n < 30:
            return 0, 0.0

        weights = np.exp(-0.03 * np.arange(n - 1, -1, -1))
        weights /= weights.sum()
        t_arr = np.arange(n, dtype=np.float64)

        try:
            coeffs = _weighted_polyfit(t_arr, prices, weights, self.poly_degree)
        except Exception:
            coeffs = np.polyfit(t_arr, prices, self.poly_degree)[::-1]

        t_last = float(n - 1)
        delta_t = max(2.0, min(8.0, np.std(np.diff(prices[-20:])) * 100))
        t_future = t_last + delta_t

        pred_price, velocity, acceleration = _evaluate_poly(
            coeffs, self.poly_degree, t_future
        )
        price_std = np.std(np.diff(prices[-20:])) if n >= 20 else 1.0
        velocity_threshold = price_std * 0.4  # v5: 降低阈值

        if abs(velocity) < velocity_threshold:
            return 0, 0.0

        direction = 1 if velocity > 0 else -1
        accel_factor = 1.0
        if acceleration * velocity > 0:
            accel_factor = 1.1
        elif acceleration * velocity < 0:
            accel_factor = 0.8

        confidence = (
            min(0.75, abs(velocity) / (velocity_threshold * 2.5)) * accel_factor
        )
        return direction, confidence

    # ---- v5: VWAP 均值回归信号 ----
    def vwap_reversion_signal(self, prices):
        """价格偏离 VWAP 过多时, 预期回归"""
        n = min(VWAP_LOOKBACK_TICKS, len(prices))
        if n < 50:
            return 0, 0.0

        vwap = np.mean(prices[-n:])  # 简化 VWAP (用均价代替)
        runtime_data["current_vwap"] = vwap

        deviation = prices[-1] - vwap
        tick_size = runtime_data.get("tick_size", 1.0)
        deviation_ticks = abs(deviation) / tick_size
        runtime_data["vwap_deviation"] = deviation_ticks

        if deviation_ticks < VWAP_DEVIATION_TICKS:
            return 0, 0.0

        # 偏离过大 -> 回归
        direction = -1 if deviation > 0 else 1
        confidence = min(0.7, deviation_ticks / (VWAP_DEVIATION_TICKS * 3))

        # 如果同时有 BB 确认, 增加置信度
        bb_upper, bb_mid, bb_lower = compute_bollinger(prices, BB_PERIOD, BB_STD_MULT)
        if direction == 1 and prices[-1] < bb_lower:
            confidence *= 1.2
        elif direction == -1 and prices[-1] > bb_upper:
            confidence *= 1.2
        confidence = min(0.85, confidence)

        return direction, confidence

    # ---- v5: BB 挤压突破信号 ----
    def bb_breakout_signal(self, prices):
        """BB 挤压后的突破信号"""
        if not runtime_data.get("bb_squeeze_active", False):
            return 0, 0.0

        # 检查挤压持续时间
        squeeze_duration = time.time() - runtime_data.get("bb_squeeze_start", 0)
        if squeeze_duration < 30:  # 至少挤压 30 秒
            return 0, 0.0

        # 检查是否发生突破
        upper, middle, lower = compute_bollinger(prices, BB_PERIOD, BB_STD_MULT)
        price = prices[-1]
        tick_size = runtime_data.get("tick_size", 1.0)
        confirm = BB_BREAKOUT_CONFIRM * tick_size

        # 突破 BB 上轨 -> 做多
        if price > upper + confirm:
            # 确认突破: 检查最近几个 tick 是否持续在轨道上方
            recent = prices[-5:]
            if all(p > upper for p in recent[-3:]):
                return 1, min(0.8, (price - upper) / (tick_size * 10))

        # 突破 BB 下轨 -> 做空
        if price < lower - confirm:
            recent = prices[-5:]
            if all(p < lower for p in recent[-3:]):
                return -1, min(0.8, (lower - price) / (tick_size * 10))

        return 0, 0.0

    # ---- v5: 成交量流信号 ----
    def volume_flow_signal(self):
        buf = list(tick_buffer)
        n = min(50, len(buf))
        if n < 15:
            return 0, 0.0
        recent = buf[-n:]

        changes = []
        for i in range(-5, 0):
            d1 = recent[i]["aggressive_buy_vol"] - recent[i]["aggressive_sell_vol"]
            d2 = (
                recent[i - 3]["aggressive_buy_vol"]
                - recent[i - 3]["aggressive_sell_vol"]
            )
            changes.append(d1 - d2)

        delta_rate = np.mean(changes)
        if abs(delta_rate) < 1:
            return 0, 0.0

        last_tick = recent[-1]
        bv, av = last_tick["bid_volume1"], last_tick["ask_volume1"]
        total = bv + av
        if total == 0:
            return 0, 0.0

        buy_pressure = bv / total
        direction = 0
        confidence = 0.0

        if delta_rate > 0 and buy_pressure > 0.52:
            direction = 1
            confidence = min(0.65, abs(delta_rate) / 120)
        elif delta_rate < 0 and buy_pressure < 0.48:
            direction = -1
            confidence = min(0.65, abs(delta_rate) / 120)

        return direction, confidence

    # ---- v5: 赫斯特指数 ----
    def compute_hurst(self):
        buf = list(tick_buffer)
        n = min(HURST_WINDOW, len(buf))
        if n < 30:
            return 0.5
        prices = np.array([t["last_price"] for t in buf[-n:]], dtype=np.float64)
        log_ret = np.diff(np.log(prices + 1e-10))
        log_ret = log_ret[np.isfinite(log_ret)]
        if len(log_ret) < 20:
            return 0.5
        h = _hurst_rs(log_ret, 50)
        h = max(0.0, min(1.0, h))
        runtime_data["current_hurst"] = h
        return h

    # ---- v5: 5-sigma 畸变屏蔽 (提高阈值) ----
    def check_anomaly(self):
        buf = list(tick_buffer)
        n = min(50, len(buf))
        if n < 10:
            return False
        prices = [t["last_price"] for t in buf[-n:]]
        dp = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        if len(dp) < 5:
            return False
        sigma = np.std(dp)
        if sigma == 0:
            return False
        if abs(dp[-1]) > SIGMA_THRESHOLD * sigma:
            self._anomaly_until = time.time() + 3.0  # v5: 从 5.0 降至 3.0
            predicted_vector["confidence"] = 0.0
            logger.warning(
                f"畸变屏蔽激活, DeltaP={dp[-1]:.2f}, {SIGMA_THRESHOLD}sigma={SIGMA_THRESHOLD * sigma:.2f}"
            )
            return True
        return False

    def is_anomaly_active(self):
        return time.time() < self._anomaly_until

    # ---- v5: 综合预测输出 ----
    def update(self):
        """多信号融合预测主循环 v5"""
        if system_state["blind_mode"]:
            return
        if self.is_anomaly_active():
            return
        if len(tick_buffer) < 30:
            return
        if self.check_anomaly():
            return

        buf = list(tick_buffer)
        prices = np.array([t["last_price"] for t in buf], dtype=np.float64)
        p_now = prices[-1]
        tick_size = runtime_data.get("tick_size", 1.0)

        # v5: 卡尔曼滤波更新
        kalman_price = self._kalman.update(p_now)
        runtime_data["kalman_price"] = kalman_price

        if p_now > runtime_data.get("today_high", 0):
            runtime_data["today_high"] = p_now
        if p_now < runtime_data.get("today_low", 999999):
            runtime_data["today_low"] = p_now

        # Step 1: 波动率 regime
        regime, atr = self.detect_volatility_regime(prices, tick_size)

        if not runtime_data["trading_allowed"]:
            if predicted_vector["confidence"] > 0:
                predicted_vector["confidence"] *= 0.85
                if predicted_vector["confidence"] < 0.01:
                    predicted_vector["direction"] = 0
                    predicted_vector["confidence"] = 0.0
                    predicted_vector["expected_displacement"] = 0.0
                    predicted_vector["signal_type"] = "NONE"
            return

        # Step 2: 多信号生成
        dir_zscore, conf_zscore = self.zscore_signal(prices)
        dir_rsi, conf_rsi = self.rsi_signal(prices)
        dir_momentum, conf_momentum = self.momentum_signal(prices)
        dir_volume, conf_volume = self.volume_flow_signal()
        dir_vwap, conf_vwap = self.vwap_reversion_signal(prices)
        dir_breakout, conf_breakout = self.bb_breakout_signal(prices)

        signals = []
        if dir_zscore != 0:
            signals.append((dir_zscore, conf_zscore, "MEAN_REV"))
        if dir_rsi != 0:
            signals.append((dir_rsi, conf_rsi, "MEAN_REV"))
        if dir_momentum != 0:
            signals.append((dir_momentum, conf_momentum, "MOMENTUM"))
        if dir_volume != 0:
            signals.append((dir_volume, conf_volume, "VOLUME"))
        if dir_vwap != 0:
            signals.append((dir_vwap, conf_vwap, "VWAP_REV"))
        if dir_breakout != 0:
            signals.append((dir_breakout, conf_breakout, "BREAKOUT"))

        if not signals:
            if predicted_vector["confidence"] > 0:
                predicted_vector["confidence"] *= 0.88
                if predicted_vector["confidence"] < 0.01:
                    predicted_vector["direction"] = 0
                    predicted_vector["confidence"] = 0.0
                    predicted_vector["expected_displacement"] = 0.0
                    predicted_vector["signal_type"] = "NONE"
            return

        # v5: 信号一致性 (2 个即可, 原为 3 个)
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

        # v5: 2 个信号确认即可 (降低门槛)
        if n_signals < 2:
            if predicted_vector["confidence"] > 0:
                predicted_vector["confidence"] *= 0.88
                if predicted_vector["confidence"] < 0.01:
                    predicted_vector["direction"] = 0
                    predicted_vector["confidence"] = 0.0
                    predicted_vector["expected_displacement"] = 0.0
                    predicted_vector["signal_type"] = "NONE"
            return

        # 一致性加成
        agreement_bonus = 0.1 if n_signals >= 3 else 0.05
        type_bonus = 0.08 if n_types >= 2 else 0.0

        confidence = min(1.0, avg_conf + agreement_bonus + type_bonus)

        # 赫斯特修正
        h = self.compute_hurst()
        if h < 0.45 and best_dir in [
            d for d, c, t in signals if t in ("MEAN_REV", "VWAP_REV")
        ]:
            confidence += 0.05
        elif h > 0.55 and best_dir in [d for d, c, t in signals if t == "MOMENTUM"]:
            confidence += 0.05

        # v5: 预期位移基于实际波动率
        atr_ticks = runtime_data.get("current_atr_ticks", 1.0)
        if atr_ticks < 0.5:
            atr_ticks = 0.5
        expected_displacement = atr_ticks * tick_size * (0.8 + confidence * 0.8)

        predicted_price = p_now + best_dir * expected_displacement

        now = time.time()
        if best_dir != predicted_vector.get("direction", 0):
            predicted_vector["signal_time"] = now

        predicted_vector["direction"] = best_dir
        predicted_vector["confidence"] = confidence
        predicted_vector["expected_displacement"] = expected_displacement
        predicted_vector["predicted_price"] = predicted_price
        predicted_vector["signal_time"] = now

        if n_types >= 2:
            predicted_vector["signal_type"] = "HYBRID"
        else:
            predicted_vector["signal_type"] = list(dir_types[best_dir])[0]

        self._last_meaningful_update = now
        self._signal_history.append(
            {
                "time": now,
                "direction": best_dir,
                "confidence": confidence,
                "price": p_now,
                "type": predicted_vector["signal_type"],
            }
        )
        if len(self._signal_history) > 100:
            self._signal_history = self._signal_history[-100:]

    # ---- 模型自愈 ----
    def recalibrate(self):
        buf = list(tick_buffer)
        n = min(300, len(buf))
        if n < 50:
            return
        prices = np.array([t["last_price"] for t in buf[-n:]], dtype=np.float64)
        t_arr = np.arange(n, dtype=np.float64)
        best_aic = float("inf")
        best_k = self.poly_degree
        for k in [2, 3, 4]:
            coeffs = np.polyfit(t_arr, prices, k)
            poly = np.poly1d(coeffs)
            residuals = prices - poly(t_arr)
            rss = np.sum(residuals**2)
            aic = n * math.log(rss / n + 1e-10) + 2 * (k + 1)
            if aic < best_aic:
                best_aic = aic
                best_k = k
        old_k = self.poly_degree
        self.poly_degree = best_k
        h = self.compute_hurst()
        logger.info(f"模型自愈: 拟合阶数 {old_k}->{best_k}, H={h:.3f}")
        runtime_data["current_hurst"] = h
        from config import trade_stats

        trade_stats["recalibration_count"] += 1
