"""
SQLite 时序数据库模块 (v5)
"""

import sqlite3
import time
import os
import logging

logger = logging.getLogger("futures_trader.database")

DB_PATH = os.path.join(os.path.dirname(__file__), "trades.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS flip_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL, datetime_str TEXT, direction INTEGER,
            entry_price REAL, exit_price REAL, trade_price REAL,
            theoretical_price REAL, slippage REAL, spread_at_time REAL,
            confidence REAL, pnl REAL, cumulative_pnl REAL,
            spatial_mode TEXT, hurst REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE, total_pnl REAL, total_trades INTEGER,
            win_rate REAL, avg_slippage REAL, max_drawdown REAL,
            prediction_accuracy REAL, repair_count INTEGER,
            recalibration_count INTEGER
        )
    """)
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")


def record_flip(
    direction,
    entry_price,
    exit_price,
    trade_price,
    theoretical_price,
    slippage,
    spread,
    confidence,
    pnl,
    cumulative_pnl,
    spatial_mode,
    hurst,
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = time.time()
    dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    c.execute(
        """INSERT INTO flip_records
        (timestamp, datetime_str, direction, entry_price, exit_price,
         trade_price, theoretical_price, slippage, spread_at_time,
         confidence, pnl, cumulative_pnl, spatial_mode, hurst)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            dt_str,
            direction,
            entry_price,
            exit_price,
            trade_price,
            theoretical_price,
            slippage,
            spread,
            confidence,
            pnl,
            cumulative_pnl,
            spatial_mode,
            hurst,
        ),
    )
    conn.commit()
    conn.close()


def record_daily_report(
    date_str,
    total_pnl,
    total_trades,
    win_rate,
    avg_slippage,
    max_drawdown,
    prediction_accuracy,
    repair_count,
    recalibration_count,
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT OR REPLACE INTO daily_reports
        (date, total_pnl, total_trades, win_rate, avg_slippage,
         max_drawdown, prediction_accuracy, repair_count, recalibration_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            date_str,
            total_pnl,
            total_trades,
            win_rate,
            avg_slippage,
            max_drawdown,
            prediction_accuracy,
            repair_count,
            recalibration_count,
        ),
    )
    conn.commit()
    conn.close()


def get_today_flips():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = time.strftime("%Y-%m-%d")
    c.execute(
        "SELECT * FROM flip_records WHERE datetime_str LIKE ? ORDER BY timestamp",
        (f"{today}%",),
    )
    rows = c.fetchall()
    conn.close()
    return rows
