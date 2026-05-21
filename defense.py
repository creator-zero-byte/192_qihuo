"""
协程D：TqSdk 容错与防御中枢 (v5)
"""

import time
import logging
from config import (
    SYMBOL,
    EQUITY_FLOOR_RATIO,
    POSITION_SYNC_INTERVAL,
    system_state,
    repair_state,
    runtime_data,
    trade_stats,
)

logger = logging.getLogger("futures_trader.defense")


class DefenseModule:
    """多层防御体系 v5"""

    def __init__(self, api, quote, position, account, executor):
        self.api = api
        self.quote = quote
        self.position = position
        self.account = account
        self.executor = executor
        self._last_pos_check = 0.0
        self._pos_mismatch_since = 0.0

    def check_all(self):
        self._check_equity_circuit_breaker()
        self._check_position_sync()

    def _check_equity_circuit_breaker(self):
        if system_state["mode"] == "SHUTDOWN":
            return
        try:
            dynamic_equity = self.account.float_profit + self.account.balance
            initial_equity = runtime_data["initial_equity"]
            if initial_equity <= 0:
                return
            equity_ratio = dynamic_equity / initial_equity
            if equity_ratio < EQUITY_FLOOR_RATIO:
                logger.critical(
                    f"触发系统性熔断! 权益比={equity_ratio:.4f}, "
                    f"动态权益={dynamic_equity:.2f}, 初始权益={initial_equity:.2f}"
                )
                self.executor.force_close_all()
                try:
                    for order in self.api.get_order().values():
                        if order.status == "ALIVE":
                            self.api.cancel_order(order)
                except Exception:
                    pass
                system_state["mode"] = "SHUTDOWN"
                repair_state["error_count"] += 1
                trade_stats["repair_count"] += 1
        except Exception as e:
            logger.error(f"权益检查异常: {e}")

    def _check_position_sync(self):
        now = time.time()
        if now - self._last_pos_check < POSITION_SYNC_INTERVAL:
            return
        self._last_pos_check = now

        try:
            physical_long = self.position.pos_long
            physical_short = self.position.pos_short
            physical_net = physical_long - physical_short
            logical_net = system_state["current_pos"]

            if physical_net != logical_net:
                if self._pos_mismatch_since == 0:
                    self._pos_mismatch_since = now
                elif now - self._pos_mismatch_since > POSITION_SYNC_INTERVAL:
                    logger.warning(
                        f"仓位不一致! 物理={physical_net}, 逻辑={logical_net}"
                    )
                    try:
                        for order in self.api.get_order().values():
                            if order.status == "ALIVE":
                                self.api.cancel_order(order)
                    except Exception:
                        pass

                    self.executor._reset_target_task()
                    system_state["current_pos"] = physical_net
                    self._pos_mismatch_since = 0

                    if repair_state["active"]:
                        repair_state["active"] = False
                        repair_state["start_time"] = None
                        repair_state["hedge_pos"] = 0
                        if system_state["mode"] == "REPAIR":
                            system_state["mode"] = "NORMAL"

                    runtime_data["last_flip_time"] = now
                    logger.info(f"仓位同步完成, 当前净仓={physical_net}")
            else:
                self._pos_mismatch_since = 0
        except Exception as e:
            logger.error(f"仓位对账异常: {e}")

    @staticmethod
    def wrap_network_safe(api, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TimeoutError, ConnectionError, OSError) as e:
            logger.error(f"网络异常: {e}, 进入重连模式")
            system_state["blind_mode"] = True
            for i in range(60):
                try:
                    api.wait_update(deadline=time.time() + 3)
                    system_state["blind_mode"] = False
                    logger.info("网络重连成功")
                    return None
                except Exception:
                    time.sleep(5)
            logger.critical("网络重连失败，系统将关闭")
            system_state["mode"] = "SHUTDOWN"
            return None
