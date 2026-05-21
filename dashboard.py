"""
协程E：TqSdk 交互式诊断仪表盘 (v5)
"""

import time
import threading
import logging
from collections import deque

logger = logging.getLogger("futures_trader.dashboard")

_dash_data = {
    "prices": deque(maxlen=2000),
    "timestamps": deque(maxlen=2000),
    "predictions": deque(maxlen=2000),
    "confidence_upper": deque(maxlen=2000),
    "confidence_lower": deque(maxlen=2000),
    "flip_marks": [],
    "equity_actual": deque(maxlen=2000),
    "equity_theoretical": deque(maxlen=2000),
    "equity_ts": deque(maxlen=2000),
    "deviation_points": [],
    "bid_depths": deque(maxlen=100),
    "ask_depths": deque(maxlen=100),
}


class Dashboard:
    def __init__(self, config_module):
        self.config = config_module
        self._app = None
        self._server_thread = None

    def push_data(self):
        from config import (
            tick_buffer,
            predicted_vector,
            system_state,
            runtime_data,
            trade_stats,
        )

        if len(tick_buffer) == 0:
            return

        latest = tick_buffer[-1]
        now = time.time()
        _dash_data["prices"].append(latest["last_price"])
        _dash_data["timestamps"].append(now)

        pred_dir = predicted_vector["direction"]
        pred_disp = predicted_vector["expected_displacement"]
        p = latest["last_price"]
        pred_price = p + pred_dir * pred_disp
        _dash_data["predictions"].append(pred_price)
        _dash_data["confidence_upper"].append(p + pred_disp)
        _dash_data["confidence_lower"].append(p - pred_disp)
        _dash_data["bid_depths"].append(latest.get("bid_volume1", 0))
        _dash_data["ask_depths"].append(latest.get("ask_volume1", 0))
        _dash_data["equity_ts"].append(now)

    def add_flip_mark(self, timestamp, price, direction):
        _dash_data["flip_marks"].append((timestamp, price, direction))

    def add_deviation_point(self, trade_no, normalized_dev):
        _dash_data["deviation_points"].append((trade_no, normalized_dev))

    def start_server(self, port=8050):
        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output
            import plotly.graph_objs as go
        except ImportError:
            logger.warning("Dash/Plotly 未安装, 仪表盘功能不可用")
            return

        from config import (
            system_state,
            predicted_vector,
            runtime_data,
            trade_stats,
            repair_state,
        )

        app = dash.Dash(__name__)
        self._app = app

        app.layout = html.Div(
            [
                html.H1(
                    "期货短线交易工具 v5 - 实时诊断仪表盘",
                    style={"textAlign": "center", "color": "#2c3e50"},
                ),
                html.Div(
                    id="status-panel",
                    style={
                        "display": "flex",
                        "justifyContent": "space-around",
                        "padding": "10px",
                        "backgroundColor": "#ecf0f1",
                        "borderRadius": "8px",
                        "margin": "10px",
                    },
                ),
                html.Div(
                    id="account-panel",
                    style={
                        "display": "flex",
                        "justifyContent": "space-around",
                        "padding": "10px",
                        "backgroundColor": "#fdf2e9",
                        "borderRadius": "8px",
                        "margin": "10px",
                    },
                ),
                html.Div(
                    id="trigger-panel",
                    style={
                        "padding": "10px",
                        "backgroundColor": "#eaf2f8",
                        "borderRadius": "8px",
                        "margin": "10px",
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="price-chart",
                            style={"height": "400px"},
                            config={"scrollZoom": True},
                        ),
                    ]
                ),
                dcc.Interval(id="interval", interval=500, n_intervals=0),
            ]
        )

        @app.callback(
            [
                Output("status-panel", "children"),
                Output("account-panel", "children"),
                Output("trigger-panel", "children"),
                Output("price-chart", "figure"),
            ],
            [Input("interval", "n_intervals")],
        )
        def update_dashboard(n):
            from config import (
                CONFIDENCE_THRESHOLD,
                PROFIT_COST_RATIO,
                COMMISSION_PER_HAND,
                ESTIMATED_SLIPPAGE_TICKS,
                FLIP_COOLDOWN,
                tick_buffer,
            )

            mode = system_state.get("mode", "UNKNOWN")
            pos = system_state.get("current_pos", 0)
            conf = predicted_vector.get("confidence", 0)
            blind = system_state.get("blind_mode", False)
            hurst = runtime_data.get("current_hurst", 0.5)
            atr = runtime_data.get("current_atr_ticks", 0)
            regime = runtime_data.get("volatility_regime", "LOW")
            trading_allowed = runtime_data.get("trading_allowed", False)
            daily_range = runtime_data.get("daily_range", 0)
            rsi = runtime_data.get("current_rsi", 50)
            zscore = runtime_data.get("current_zscore", 0)
            last_dt = runtime_data.get("last_data_time", 0)
            freshness = (time.time() - last_dt) * 1000 if last_dt > 0 else 9999

            fresh_color = (
                "#27ae60"
                if freshness < 200
                else ("#f39c12" if freshness < 500 else "#e74c3c")
            )
            regime_color = {
                "LOW": "#e74c3c",
                "NORMAL": "#f39c12",
                "HIGH": "#27ae60",
            }.get(regime, "#95a5a6")

            status_children = [
                html.Div([html.B("MODE"), html.Div(mode)]),
                html.Div([html.B("POS"), html.Div(f"{pos:+d}")]),
                html.Div([html.B("CONF"), html.Div(f"{conf:.1%}")]),
                html.Div(
                    [html.B("REGIME"), html.Div(regime, style={"color": regime_color})]
                ),
                html.Div(
                    [
                        html.B("TRADE"),
                        html.Div(
                            "允许" if trading_allowed else "禁止",
                            style={
                                "color": "#27ae60" if trading_allowed else "#e74c3c"
                            },
                        ),
                    ]
                ),
                html.Div([html.B("ATR"), html.Div(f"{atr:.2f} ticks")]),
                html.Div([html.B("RANGE"), html.Div(f"{daily_range:.0f} ticks")]),
                html.Div([html.B("RSI"), html.Div(f"{rsi:.1f}")]),
                html.Div([html.B("Z"), html.Div(f"{zscore:.2f}")]),
                html.Div(
                    [
                        html.B("FRESH"),
                        html.Div(f"{freshness:.0f}ms", style={"color": fresh_color}),
                    ]
                ),
            ]

            acct_balance = runtime_data.get("account_balance", 0)
            float_pnl = runtime_data.get("float_profit", 0)
            dynamic_eq = runtime_data.get("dynamic_equity", 0)
            init_eq = runtime_data.get("initial_equity", 0)
            total_pnl = trade_stats.get("total_pnl", 0)
            pnl_color = "#27ae60" if float_pnl >= 0 else "#e74c3c"

            account_children = [
                html.Div([html.B("初始权益"), html.Div(f"{init_eq:,.2f}")]),
                html.Div([html.B("动态权益"), html.Div(f"{dynamic_eq:,.2f}")]),
                html.Div(
                    [
                        html.B("浮动盈亏"),
                        html.Div(f"{float_pnl:+,.2f}", style={"color": pnl_color}),
                    ]
                ),
                html.Div([html.B("累计盈亏"), html.Div(f"{total_pnl:+,.2f}")]),
                html.Div(
                    [
                        html.B("交易次数"),
                        html.Div(f"{trade_stats.get('trade_count', 0)}"),
                    ]
                ),
                html.Div(
                    [
                        html.B("胜率"),
                        html.Div(
                            f"{trade_stats.get('win_count', 0)}/{trade_stats.get('trade_count', 1)}"
                        ),
                    ]
                ),
            ]

            e_disp = predicted_vector.get("expected_displacement", 0)
            direction = predicted_vector.get("direction", 0)
            dir_str = {1: "LONG", -1: "SHORT", 0: "无信号"}.get(direction, "无信号")
            signal_type = predicted_vector.get("signal_type", "NONE")

            trigger_children = [
                html.B("交易条件: "),
                html.Span(
                    f"模式={mode} ",
                    style={"color": "#27ae60" if mode == "NORMAL" else "#e74c3c"},
                ),
                html.Span(
                    f"信号={dir_str} ({signal_type}) ",
                    style={"color": "#27ae60" if direction != 0 else "#e74c3c"},
                ),
                html.Span(
                    f"置信度={conf:.1%} ",
                    style={
                        "color": "#27ae60"
                        if conf >= CONFIDENCE_THRESHOLD
                        else "#e74c3c"
                    },
                ),
                html.Span(
                    f"波动率={regime} ",
                    style={"color": "#27ae60" if trading_allowed else "#e74c3c"},
                ),
            ]

            prices = list(_dash_data["prices"])
            ts = list(range(len(prices)))
            preds = list(_dash_data["predictions"])

            price_fig = go.Figure()
            price_fig.add_trace(
                go.Scatter(x=ts, y=prices, name="实时价格", line=dict(color="#2c3e50"))
            )
            if preds:
                price_fig.add_trace(
                    go.Scatter(
                        x=ts,
                        y=preds,
                        name="预测路径",
                        line=dict(color="#e74c3c", dash="dash"),
                    )
                )

            va_h = runtime_data.get("va_high", 0)
            va_l = runtime_data.get("va_low", 0)
            if va_h > 0:
                price_fig.add_hline(
                    y=va_h,
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="VA High",
                )
                price_fig.add_hline(
                    y=va_l, line_dash="dot", line_color="blue", annotation_text="VA Low"
                )

            bb_upper = runtime_data.get("bb_upper", 0)
            bb_lower = runtime_data.get("bb_lower", 0)
            if bb_upper > 0:
                price_fig.add_hline(
                    y=bb_upper,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="BB Upper",
                )
                price_fig.add_hline(
                    y=bb_lower,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="BB Lower",
                )

            for fm in _dash_data["flip_marks"][-50:]:
                idx = len(prices) - 1
                marker = "UP" if fm[2] == 1 else "DN"
                color = "#27ae60" if fm[2] == 1 else "#e74c3c"
                price_fig.add_annotation(
                    x=idx,
                    y=fm[1],
                    text=marker,
                    font=dict(color=color, size=16),
                    showarrow=False,
                )

            price_fig.update_layout(
                title=f"Tick 价格流 | Regime: {regime} | 日波动: {daily_range:.0f} ticks",
                margin=dict(l=40, r=20, t=40, b=20),
                uirevision="price-chart-stable",
            )

            return (status_children, account_children, trigger_children, price_fig)

        def _run():
            try:
                logging.getLogger("werkzeug").setLevel(logging.ERROR)
                app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
            except Exception as e:
                logger.error(f"仪表盘服务异常退出: {e}", exc_info=True)

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        logger.info(f"仪表盘已启动: http://127.0.0.1:{port}")
