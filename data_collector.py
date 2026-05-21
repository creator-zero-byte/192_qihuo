"""
协程A: TqSdk 毫秒级数据采集器 (v5)
新增: VWAP 追踪, 卡尔曼滤波输入, 盘口深度指标
"""

import time
import logging
from config import (
    tick_buffer,
    system_state,
    runtime_data,
    DATA_STALE_THRESHOLD,
)

logger = logging.getLogger("futures_trader.data_collector")


class DataCollector:
    """Tick 数据采集与衍生指标计算 v5"""

    def __init__(self, api, quote):
        self.api = api
        self.quote = quote
        self._last_volume = 0
        self._initialized = False
        self._vwap_sum = 0.0
        self._vwap_vol = 0
        self._same_dir_count = 0
        self._last_price_dir = 0

    def on_tick(self, quote):
        try:
            last_price = quote.last_price
            bid1 = quote.bid_price1
            ask1 = quote.ask_price1
            bid_vol1 = quote.bid_volume1
            ask_vol1 = quote.ask_volume1
            volume = quote.volume
            dt = quote.datetime

            if last_price > runtime_data.get("today_high", 0):
                runtime_data["today_high"] = last_price
            if last_price < runtime_data.get("today_low", 999999):
                runtime_data["today_low"] = last_price

            p_mid = (bid1 + ask1) / 2.0
            spread = ask1 - bid1

            if not self._initialized:
                self._last_volume = volume
                runtime_data["today_open"] = last_price
                self._initialized = True
                delta_volume = 0
            else:
                delta_volume = volume - self._last_volume
                self._last_volume = volume

            if last_price >= ask1:
                direction_flag = 1
            elif last_price <= bid1:
                direction_flag = -1
            else:
                direction_flag = 0

            if direction_flag == 1:
                runtime_data["aggressive_buy_vol"] += delta_volume
            elif direction_flag == -1:
                runtime_data["aggressive_sell_vol"] += delta_volume

            if len(tick_buffer) > 0:
                prev_price = tick_buffer[-1]["last_price"]
                if last_price > prev_price:
                    current_dir = 1
                elif last_price < prev_price:
                    current_dir = -1
                else:
                    current_dir = 0

                if current_dir == self._last_price_dir and current_dir != 0:
                    self._same_dir_count += 1
                else:
                    self._same_dir_count = 1
                    self._last_price_dir = current_dir

            # v5: VWAP 追踪 (不重置)
            if delta_volume > 0:
                self._vwap_sum += last_price * delta_volume
                self._vwap_vol += delta_volume

            vol_mult = getattr(quote, "volume_multiple", 1) or 1
            price_tick = getattr(quote, "price_tick", 1.0)
            runtime_data["tick_size"] = price_tick

            snapshot = {
                "datetime": dt,
                "last_price": last_price,
                "bid_price1": bid1,
                "ask_price1": ask1,
                "bid_volume1": bid_vol1,
                "ask_volume1": ask_vol1,
                "volume": volume,
                "p_mid": p_mid,
                "spread": spread,
                "delta_volume": delta_volume,
                "direction_flag": direction_flag,
                "aggressive_buy_vol": runtime_data["aggressive_buy_vol"],
                "aggressive_sell_vol": runtime_data["aggressive_sell_vol"],
                "timestamp": time.time(),
                "volume_multiple": vol_mult,
                "price_tick": price_tick,
                "vwap": self._vwap_sum / self._vwap_vol
                if self._vwap_vol > 0
                else last_price,
                "same_dir_count": self._same_dir_count,
                "order_imbalance": (bid_vol1 - ask_vol1) / max(bid_vol1 + ask_vol1, 1),
            }
            tick_buffer.append(snapshot)
            runtime_data["last_data_time"] = time.time()
            runtime_data["current_vwap"] = (
                self._vwap_sum / self._vwap_vol if self._vwap_vol > 0 else last_price
            )

        except Exception as e:
            logger.error(f"Tick 采集异常: {e}")

    def check_freshness(self):
        now = time.time()
        dt = now - runtime_data["last_data_time"]
        if runtime_data["last_data_time"] == 0:
            return
        if dt > DATA_STALE_THRESHOLD:
            if not system_state["blind_mode"]:
                system_state["blind_mode"] = True
                logger.warning(f"警报: 行情中断, 延迟={dt * 1000:.0f}ms, 进入盲飞模式")
                try:
                    for order in self.api.get_order().values():
                        if order.status == "ALIVE":
                            self.api.cancel_order(order)
                except Exception as e:
                    logger.error(f"撤单异常: {e}")
        elif dt < 0.2:
            if system_state["blind_mode"]:
                system_state["blind_mode"] = False
                logger.info(f"行情恢复, 延迟={dt * 1000:.0f}ms")
