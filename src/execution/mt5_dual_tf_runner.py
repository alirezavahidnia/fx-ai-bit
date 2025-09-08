# MT5 runner for low-equity FX on M5.
# - Combines technical, fundamental (events), and sentiment (news) analysis.
# - Data fetching (news, events) runs in a non-blocking background thread.
# - ATR-based SL/TP, risk-based lot sizing, daily DD kill-switch.

from __future__ import annotations
import os, time, argparse, datetime as dt
from typing import Dict, Tuple, List, Any
import threading
from copy import deepcopy

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
import MetaTrader5 as mt5

# Load env for API keys, Telegram tokens, etc.
load_dotenv()

from ..utils.config import load_config
from ..utils.trade_logger import TradeLogger
from ..features.technical import rsi, macd, realized_vol, atr
from ..features import fundamental, alpha_vantage_fetcher

# ================= Globals for Background Data & Threading =================
shared_data_lock = threading.Lock()
shared_data = {
    "news_sentiment": {},
    "economic_events": [],
    "last_updated": None
}

# ================= Default tunables (used as fallbacks; overridden by config inside main) =================
DRY_RUN        = False
ENTER_THRESH   = 0.62
EXIT_THRESH    = 0.55
MIN_HOLD_MIN   = 15
COOLDOWN_MIN   = 10
REQUIRE_CONSENSUS = False
SKIP_IF_MINLOTS_EXCEEDS_RISK = True
RISK_TOLERANCE_MULTIPLIER = 1.10
MAGIC_NUMBER   = 202501

# ================= Time helpers =================
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def today_utc_start() -> dt.datetime:
    n = now_utc()
    return n.replace(hour=0, minute=0, second=0, microsecond=0)

def within_trading_window(cfg: dict) -> bool:
    w = cfg.get("trading_window", {})
    if not w.get("enabled", False):
        return True
    hour = now_utc().hour
    return w["start_hour_utc"] <= hour < w["end_hour_utc"]

# ================= Background Data Fetching =================
def background_data_updater(cfg: dict, symbols: list[str]):
    """
    Runs in a background thread to periodically fetch news and economic data
    without blocking the main trading loop.
    """
    global shared_data

    api_cfg = cfg.get("api_keys", {})
    av_api_key_env_name = api_cfg.get("alpha_vantage_api_key_env")
    av_api_key = os.getenv(av_api_key_env_name) if av_api_key_env_name else None

    news_provider = cfg.get("news_provider", "alphavantage" if av_api_key else None)

    while True:
        try:
            logger.info("BACKGROUND: Fetching news and economic data...")
            sentiments = {}
            if news_provider == "alphavantage":
                sentiments = alpha_vantage_fetcher.fetch_news_sentiments(api_key=av_api_key, symbols=symbols)

            events = fundamental.get_economic_calendar()

            with shared_data_lock:
                shared_data["news_sentiment"] = sentiments
                shared_data["economic_events"] = events
                shared_data["last_updated"] = now_utc()

            logger.info("BACKGROUND: Data updated successfully.")

        except Exception as e:
            logger.exception(f"BACKGROUND: Data fetch failed: {e}")

        # Update every 15 minutes
        time.sleep(15 * 60)

# ================= MT5 data =================
TF_MAP = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
}

def fetch_bars(symbol: str, tf_code, count: int = 600) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, count)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.rename(columns={"open":"open","high":"high","low":"low","close":"close"}) \
             .set_index("time")[["open","high","low","close"]]

# ================= Features & Signal Generation =================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi14"] = rsi(out["close"], 14)
    m = macd(out["close"])
    out = out.join(m)
    out["rv96"] = realized_vol(out["close"], 96)
    out["atr14"] = atr(out, 14)
    out["ema50"]  = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out.dropna(inplace=True)
    return out

def prob_up_from_feats(feats: pd.DataFrame) -> float:
    den = float(feats["rv96"].iloc[-1]) + 1e-9
    macd_norm  = float(feats["macd_hist"].iloc[-1]) / den
    trend_norm = float(feats["ema50"].iloc[-1] - feats["ema200"].iloc[-1]) / den
    macd_term  = np.tanh(2.0 * macd_norm)
    trend_term = np.tanh(1.2 * trend_norm)
    rsi_dist   = (float(feats["rsi14"].iloc[-1]) - 50.0) / 20.0
    rsi_term   = np.clip(rsi_dist, -2.0, 2.0)
    logit = (0.8 * trend_term) + (0.6 * macd_term) - (0.6 * rsi_term)
    price, ema50, ema200 = float(feats["close"].iloc[-1]), float(feats["ema50"].iloc[-1]), float(feats["ema200"].iloc[-1])
    if price < min(ema50, ema200): logit -= 0.10
    elif price > max(ema50, ema200): logit += 0.10
    return float(1.0 / (1.0 + np.exp(-logit)))

def get_final_trade_decision(
    p_up_technical: float,
    symbol: str,
    current_sentiment: dict,
    current_events: list
) -> Tuple[float, str]:
    """
    Combines technical probability with sentiment and economic events.
    - Blocks trades if a high-impact event is imminent.
    - Adjusts probability based on news sentiment.
    """
    # 1. Check for blocking economic events
    now = now_utc()
    event_window_seconds = 3600  # Block trades 1h before a high-impact event
    for event in current_events:
        event_time = event.get('datetime_utc')
        if event.get('impact') == 'High' and event_time:
            time_diff_seconds = (event_time - now).total_seconds()
            # Block if event is in the next hour or happened in the last 15 mins
            if -900 < time_diff_seconds < event_window_seconds:
                reason = f"BLOCK: High-impact event '{event.get('event')}' at {event_time.strftime('%H:%M')} UTC"
                return 0.5, reason # Return neutral prob to prevent trade

    # 2. Adjust based on sentiment
    final_p_up = p_up_technical
    symbol_sentiment = current_sentiment.get(symbol)
    if symbol_sentiment and symbol_sentiment.get("relevance_sum", 0) > 0.5: # Require some relevance
        # Map sentiment score [-1, 1] to a sentiment probability [0, 1]
        sentiment_score = symbol_sentiment.get('sentiment_score', 0.0)
        sentiment_p_up = 0.5 + (sentiment_score / 2.0)

        # Weighted average: 70% technical, 30% sentiment
        final_p_up = (0.7 * p_up_technical) + (0.3 * sentiment_p_up)
        reason = f"Tech({p_up_technical:.2f}) + Sent({sentiment_p_up:.2f})"
    else:
        reason = f"Tech({p_up_technical:.2f}) only"

    return final_p_up, reason

# ================= Account / execution =================
def account_equity() -> float:
    acc = mt5.account_info()
    if not acc: raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
    return float(acc.equity)

def symbol_info_or_raise(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None: raise RuntimeError(f"symbol_info({symbol}) failed")
    if not info.visible: mt5.symbol_select(symbol, True)
    return info

def close_symbol_positions(symbol: str) -> dict:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f"positions_get({symbol}) failed: {mt5.last_error()}")
        return {"ok": False, "msg": str(mt5.last_error()), "closed_tickets": []}
    if not positions: return {"ok": True, "msg": "no positions", "closed_tickets": []}

    tick = mt5.symbol_info_tick(symbol)
    results, closed_tickets = [], [pos.ticket for pos in positions]
    for pos in positions:
        deal_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if deal_type == mt5.ORDER_TYPE_SELL else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": pos.volume,
            "type": deal_type, "position": pos.ticket, "price": price, "deviation": 20,
            "magic": MAGIC_NUMBER, "comment": "close_by_python", "type_filling": mt5.ORDER_FILLING_FOK,
        }
        r = mt5.order_send(req)
        results.append(f"{pos.ticket}:{r.retcode}")
    return {"ok": True, "msg": ";".join(results), "closed_tickets": closed_tickets}

def calc_sl_tp(side: int, price: float, atr_val: float, sl_mult: float, tp_mult: float) -> Tuple[float, float]:
    if not atr_val or atr_val <= 0: atr_val = 0.0001
    return (price - sl_mult * atr_val, price + tp_mult * atr_val) if side > 0 else (price + sl_mult * atr_val, price - tp_mult * atr_val)

def enforce_min_distance(symbol: str, side: int, entry: float, sl: float, tp: float) -> Tuple[float, float]:
    info = symbol_info_or_raise(symbol)
    point = getattr(info, 'point', 0.00001)

    # Use getattr for safety, as some brokers might not provide this field.
    stops_level_pips = getattr(info, 'stops_level', 0)
    min_dist = stops_level_pips * point

    if min_dist <= 0:
        return sl, tp

    if side > 0: # Long
        sl = min(sl, entry - min_dist)
        tp = max(tp, entry + min_dist)
    else: # Short
        sl = max(sl, entry + min_dist)
        tp = min(tp, entry - min_dist)

    # Snap to grid, ensuring we don't divide by zero
    if point > 0:
        sl = round(sl / point) * point
        tp = round(tp / point) * point

    return sl, tp

def place_market_order(symbol: str, side: int, lots: float, sl: float | None=None, tp: float | None=None) -> dict:
    info = symbol_info_or_raise(symbol)
    tick = mt5.symbol_info_tick(symbol)
    order_type = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
    price = tick.ask if side > 0 else tick.bid
    point = info.point
    if point > 0:
        if sl: sl = round(sl / point) * point
        if tp: tp = round(tp / point) * point

    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lots), "type": order_type,
        "price": price, "deviation": 20, "magic": MAGIC_NUMBER, "comment": "python_entry",
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    if sl: req["sl"] = sl
    if tp: req["tp"] = tp

    if DRY_RUN: return {"ok": True, "retcode": "DRY", "comment": "simulated"}

    result = mt5.order_send(req)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"ok": True, "retcode": result.retcode, "comment": result.comment, "deal_id": result.deal}
    return {"ok": False, "retcode": result.retcode, "comment": result.comment}

# ---------- Telegram: DAILY SUMMARY ONLY ----------
def _tg_creds(cfg):
    a = cfg.get("alerts", {}) or {}
    if not a.get("telegram_enabled", False): return None, None
    token = os.getenv(a.get("telegram_bot_token_env", ""))
    chat_id = os.getenv(a.get("telegram_chat_id_env", ""))
    return (token, chat_id) if token and chat_id else (None, None)

def send_telegram_alert(cfg: dict, text: str) -> bool:
    token, chat_id = _tg_creds(cfg)
    if not token: return False
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/sendMessage",
            params={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        return r.ok
    except Exception: return False

def human_pl(v: float) -> str:
    return f"{'🟢' if v >= 0 else '🔴'} ${abs(v):.2f}"

def build_daily_summary(start_ts: dt.datetime, end_ts: dt.datetime, symbols: list[str]) -> dict:
    deals = mt5.history_deals_get(start_ts, end_ts) or []
    summary = {"total": 0.0, "trades": 0, "wins": 0, "per_symbol": {s: 0.0 for s in symbols}}
    for d in deals:
        if d.entry != mt5.DEAL_ENTRY_OUT or d.symbol not in summary["per_symbol"]: continue
        net = d.profit + d.commission + d.swap
        summary["total"] += net
        summary["per_symbol"][d.symbol] += net
        summary["trades"] += 1
        if net > 0: summary["wins"] += 1
    summary["winrate"] = (summary["wins"] / summary["trades"] * 100.0) if summary["trades"] else 0.0
    return summary

def format_daily_summary(start_ts: dt.datetime, end_ts: dt.datetime, sym_summary: dict) -> str:
    lines = [
        "<b>📊 Daily Summary</b>",
        f"<b>Period:</b> {start_ts:%Y-%m-%d %H:%M} → {end_ts:%Y-%m-%d %H:%M} UTC",
        f"<b>Trades:</b> {sym_summary['trades']}   <b>Win rate:</b> {sym_summary['winrate']:.0f}%",
        f"<b>Net P/L:</b> {human_pl(sym_summary['total'])}", "<b>By symbol:</b>"
    ] + [f"• {s}: {human_pl(v)}" for s, v in sym_summary["per_symbol"].items()]
    return "\n".join(lines)

def format_trade_close_alert(deal: mt5.TradeDeal) -> str:
    pnl = deal.profit + deal.commission + deal.swap
    return "\n".join([
        f"<b>🔔 Trade Closed: {deal.symbol}</b>",
        f"<b>Direction:</b> {'BUY' if deal.type == mt5.DEAL_TYPE_SELL else 'SELL'}",
        f"<b>Profit/Loss:</b> {human_pl(pnl)}",
        f"<b>Volume:</b> {deal.volume:.2f}",
        f"<b>Close Price:</b> {deal.price:.5f}",
    ])

# ================= Robust lot sizing (FX-friendly) =================
def first_non_none(*vals):
    return next((v for v in vals if v is not None), None)

def risk_dollars_from_bps(equity: float, bps: float) -> float:
    return equity * (bps / 10_000.0)

def clamp_to_step(x: float, step: float, min_v: float, max_v: float) -> float:
    if step <= 0: step = 0.01
    return round(max(min_v, min(max_v, x)) / step) * step

def lots_for_risk(symbol: str, equity: float, stop_distance_price: float, bps: float) -> float:
    info = symbol_info_or_raise(symbol)

    # Safely get attributes that might be missing from the broker's data
    point = getattr(info, 'point', 0.00001)
    trade_tick_size = getattr(info, 'trade_tick_size', None)
    tick_size_val = getattr(info, 'tick_size', None)

    trade_tick_value = getattr(info, 'trade_tick_value', None)
    tick_value_val = getattr(info, 'tick_value', None)
    contract_size = getattr(info, 'trade_contract_size', 0)

    tick_size = first_non_none(trade_tick_size, tick_size_val, point)
    tick_value = first_non_none(trade_tick_value, tick_value_val, (point * contract_size) or None)

    vol_min = getattr(info, 'volume_min', 0.01)
    vol_step = getattr(info, 'volume_step', 0.01)
    vol_max = getattr(info, 'volume_max', 100.0)

    if not all([tick_size, tick_value, stop_distance_price > 0]):
        logger.warning(f"[{symbol}] Sizing fallback due to missing info: tick_size={tick_size}, tick_value={tick_value}, stop_dist={stop_distance_price}")
        return clamp_to_step(0.01, vol_step, vol_min, vol_max)

    risk_per_lot = (stop_distance_price / tick_size) * tick_value
    if risk_per_lot <= 0:
        logger.warning(f"[{symbol}] risk_per_lot<=0 fallback; tick_size={tick_size}, tick_value={tick_value}")
        return clamp_to_step(0.01, vol_step, vol_min, vol_max)

    risk_target = risk_dollars_from_bps(equity, bps)
    lots_raw = risk_target / risk_per_lot
    lots = clamp_to_step(lots_raw, vol_step, vol_min, vol_max)
    logger.info(f"[{symbol}] Sizing: stop_dist={stop_distance_price:.5f}, risk_per_lot=${risk_per_lot:.2f}, target=${risk_target:.2f} -> lots={lots}")
    return lots

def estimate_risk_dollars(symbol: str, lots: float, stop_distance_price: float) -> float:
    info = symbol_info_or_raise(symbol)

    # Safely get attributes
    point = getattr(info, 'point', 0.00001)
    trade_tick_size = getattr(info, 'trade_tick_size', None)
    tick_size_val = getattr(info, 'tick_size', None)

    trade_tick_value = getattr(info, 'trade_tick_value', None)
    tick_value_val = getattr(info, 'tick_value', None)
    contract_size = getattr(info, 'trade_contract_size', 0)

    tick_size = first_non_none(trade_tick_size, tick_size_val, point)
    tick_value = first_non_none(trade_tick_value, tick_value_val, (point * contract_size) or None)

    if not all([tick_size, tick_value, stop_distance_price > 0]):
        return float("inf")

    return (stop_distance_price / tick_size) * tick_value * lots

# ================= Main Execution Logic =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None, help="Comma-separated override (e.g., EURUSD,GBPUSD)")
    args = ap.parse_args()

    cfg = load_config()
    symbols = (args.symbols.split(",") if args.symbols else cfg.get("symbols", []))
    tfs = cfg.get("timeframes", ["M5"])
    tf_codes = {tf: TF_MAP[tf] for tf in tfs}

    strat = cfg.get("strategy") or {}
    enter_thresh, exit_thresh = strat.get("enter_thresh", ENTER_THRESH), strat.get("exit_thresh", EXIT_THRESH)
    min_hold_min, cooldown_min = strat.get("min_hold_min", MIN_HOLD_MIN), strat.get("cooldown_min", COOLDOWN_MIN)
    require_entry_confirmation = strat.get("require_entry_confirmation", False)

    risk_cfg = cfg["risk"]
    sl_mult, tp_mult = risk_cfg.get("stop_atr_mult", 2.0), risk_cfg.get("takeprofit_atr_mult", 3.0)
    trade_bps = risk_cfg.get("max_risk_per_trade_bps", 80.0)
    kill = risk_cfg.get("kill_switch", {})
    daily_dd_max, max_trades_day = kill.get("max_intraday_drawdown_pct", 4.0), kill.get("max_daily_trades_per_symbol", 6)

    if not mt5.initialize(): raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    logger.info(f"MT5 initialized. Symbols={symbols}, TFs={tfs}")

    # ---- State ----
    last_bar_time = {s: {tf: pd.Timestamp(0, tz="UTC") for tf in tfs} for s in symbols}
    tf_side = {s: {tf: 0 for tf in tfs} for s in symbols}
    open_side = {s: 0 for s in symbols}
    trades_today = {s: 0 for s in symbols}
    last_open_time = {s: now_utc() - dt.timedelta(days=99) for s in symbols}
    last_close_time = {s: now_utc() - dt.timedelta(days=99) for s in symbols}
    pending_alerts = set()
    pending_signal: Dict[str, Dict[str, Any]] = {} # For entry confirmation feature
    trade_logger = TradeLogger()

    for p in mt5.positions_get() or []:
        if p.symbol in symbols:
            open_side[p.symbol] = +1 if p.type == mt5.POSITION_TYPE_BUY else -1
            last_open_time[p.symbol] = dt.datetime.fromtimestamp(p.time, tz=dt.timezone.utc)
            logger.info(f"Detected live {p.symbol} {'LONG' if open_side[p.symbol]>0 else 'SHORT'} from {last_open_time[p.symbol]}")

    day_start, equity_day_open, paused_for_dd = today_utc_start(), account_equity(), False

    # ---- Start background data thread ----
    data_thread = threading.Thread(target=background_data_updater, args=(cfg, symbols), daemon=True)
    data_thread.start()
    logger.info("Background data fetching thread started. Waiting 10s for initial data...")
    time.sleep(10)

    logger.info("🟢 Bot started.")

    while True:
        nowt_loop = now_utc()

        if pending_alerts:
            deals = mt5.history_deals_get(day_start, nowt_loop) or []
            processed = {d.position_id for d in deals if d.position_id in pending_alerts and d.entry == mt5.DEAL_ENTRY_OUT and (
                trade_logger.log_close_trade(d.position_id, d.profit + d.commission + d.swap),
                send_telegram_alert(cfg, format_trade_close_alert(d)) if cfg.get("alerts",{}).get("alert_on_trade_close") else True
            )}
            pending_alerts.difference_update(processed)

        if nowt_loop >= day_start + dt.timedelta(days=1):
            summary = build_daily_summary(day_start, nowt_loop, symbols)
            send_telegram_alert(cfg, format_daily_summary(day_start, nowt_loop, summary))
            day_start, equity_day_open = today_utc_start(), account_equity()
            trades_today, paused_for_dd = {s: 0 for s in symbols}, False
            logger.info("🔄 New UTC day: counters reset")

        dd_pct = (account_equity() - equity_day_open) / max(equity_day_open, 1) * 100.0
        if dd_pct <= -abs(daily_dd_max) and not paused_for_dd:
            if not DRY_RUN:
                for s in symbols:
                    res = close_symbol_positions(s)
                    if res.get("closed_tickets"): pending_alerts.update(res["closed_tickets"])
            paused_for_dd = True
            logger.error(f"⛔ KILL-SWITCH: DD {dd_pct:.2f}% hit. Paused until next day.")
        if paused_for_dd or not within_trading_window(cfg):
            time.sleep(60)
            continue

        with shared_data_lock:
            latest_sentiment = deepcopy(shared_data["news_sentiment"])
            latest_events = deepcopy(shared_data["economic_events"])

        for s in symbols:
            # Rate limit check
            if trades_today[s] >= max_trades_day: continue

            # --- Step 1: Check for and action any pending entry signals ---
            if s in pending_signal and require_entry_confirmation:
                primary_tf = tfs[0]
                df = fetch_bars(s, tf_codes[primary_tf], count=2)
                if df.empty or df.index[-1] <= pending_signal[s]["bar_time"]:
                    continue # Wait for a new bar

                confirming_bar = df.iloc[-2] # The bar that just closed
                signal_side = pending_signal[s]["side"]

                confirmed = (signal_side > 0 and confirming_bar['close'] > confirming_bar['open']) or \
                            (signal_side < 0 and confirming_bar['close'] < confirming_bar['open'])

                if confirmed:
                    logger.info(f"[{s}] CONFIRMED pending {'LONG' if signal_side > 0 else 'SHORT'} signal.")
                    # Pop the signal and proceed to execution
                    signal_details = pending_signal.pop(s)
                    # Use the original ATR and features for sizing/logging
                    atr_use, last_feats = signal_details["atr"], signal_details["feats"]
                    tech_p_up, final_p_up = signal_details["tech_p_up"], signal_details["final_p_up"]

                    # --- Execute confirmed trade ---
                    tick = mt5.symbol_info_tick(s)
                    if not tick: continue
                    entry_price = tick.ask if signal_side > 0 else tick.bid
                    sl, tp = calc_sl_tp(signal_side, entry_price, atr_use, sl_mult, tp_mult)
                    sl, tp = enforce_min_distance(s, signal_side, entry_price, sl, tp)
                    stop_dist = abs(entry_price - sl)
                    equity = account_equity()
                    lots = lots_for_risk(s, equity, stop_dist, trade_bps)

                    if SKIP_IF_MINLOTS_EXCEEDS_RISK and estimate_risk_dollars(s, lots, stop_dist) > risk_dollars_from_bps(equity, trade_bps) * RISK_TOLERANCE_MULTIPLIER:
                        logger.warning(f"[{s}] SKIP confirmed trade: Min lot risk exceeds target.")
                        continue

                    resp = place_market_order(s, signal_side, lots, sl=sl, tp=tp)
                    logger.info(f"[{s}] EXECUTED confirmed {'LONG' if signal_side>0 else 'SHORT'} lots={lots:.2f}, resp={resp}")
                    if resp.get("ok"):
                        open_side[s], last_open_time[s] = signal_side, nowt_loop
                        trades_today[s] += 1
                        if resp.get("deal_id") and last_feats is not None:
                            log_features = {
                                'technical_p_up': tech_p_up, 'final_p_up': final_p_up,
                                'rsi14': last_feats["rsi14"].iloc[-1], 'macd_hist': last_feats["macd_hist"].iloc[-1], 'rv96': last_feats["rv96"].iloc[-1],
                                'ema_dist_norm': (last_feats["ema50"].iloc[-1] - last_feats["ema200"].iloc[-1]) / (last_feats["rv96"].iloc[-1] + 1e-9),
                                'base_sentiment': latest_sentiment.get(s,{}).get('sentiment_score'),
                            }
                            trade_logger.log_open_trade(resp["deal_id"], s, signal_side, lots, entry_price, sl, tp, log_features)
                else:
                    logger.info(f"[{s}] INVALIDATED pending signal by confirming bar.")
                    pending_signal.pop(s) # Cancel the signal

            # --- Step 2: Calculate new signals for the current tick ---
            tf_probs, tf_atr, updated_any, last_feats = {}, {}, False, None
            for tf in tfs:
                df = fetch_bars(s, tf_codes[tf], count=400)
                if df.empty or len(df) < 200 or df.index[-1] == last_bar_time[s][tf]: continue
                last_bar_time[s][tf], updated_any = df.index[-1], True
                feats = build_features(df.iloc[:-1])
                if feats.empty: continue
                last_feats = feats
                p = prob_up_from_feats(feats)
                tf_probs[tf], tf_atr[tf] = p, float(feats["atr14"].iloc[-1])
                tf_side[s][tf] = +1 if p >= 0.5 else -1

            if not updated_any: continue

            # --- Step 3: Make decision based on new signal ---
            sides = [tf_side[s][tf] for tf in tfs]
            raw_side = sides[0] if sides else 0
            atr_use = tf_atr.get(tfs[0], 0.0)
            if not atr_use: continue

            tech_p_up = float(np.median(list(tf_probs.values()))) if tf_probs else 0.5
            final_p_up, reason = get_final_trade_decision(tech_p_up, s, latest_sentiment, latest_events)

            current, desired = open_side[s], open_side[s]
            p_up, p_down = final_p_up, 1.0 - final_p_up

            if current == 0:
                if (nowt_loop - last_close_time[s]).total_seconds() / 60.0 >= cooldown_min:
                    if raw_side > 0 and p_up >= enter_thresh: desired = +1
                    elif raw_side < 0 and p_down >= enter_thresh: desired = -1
            elif (nowt_loop - last_open_time[s]).total_seconds() / 60.0 >= min_hold_min:
                if current > 0 and (p_up < exit_thresh or p_down >= enter_thresh): desired = 0
                elif current < 0 and (p_down < exit_thresh or p_up >= enter_thresh): desired = 0

            if desired == current: continue

            # --- Step 4: Action the decision (Exit, Enter, or set Pending) ---
            logger.info(f"[{s}] State change: {current}->{desired}. Reason: {reason}, Final p(up)={final_p_up:.3f}")

            if desired == 0: # Exit logic
                if current != 0 and not DRY_RUN:
                    res = close_symbol_positions(s)
                    if res.get("closed_tickets"): pending_alerts.update(res["closed_tickets"])
                open_side[s], last_close_time[s] = 0, nowt_loop
                if s in pending_signal: pending_signal.pop(s) # Clear any pending signals too
                continue

            # Entry logic
            if current != 0 and not DRY_RUN: # Flip existing position
                close_symbol_positions(s)

            if require_entry_confirmation:
                pending_signal[s] = {
                    "side": desired,
                    "bar_time": last_bar_time[s][tfs[0]],
                    "atr": atr_use,
                    "feats": last_feats,
                    "tech_p_up": tech_p_up,
                    "final_p_up": final_p_up
                }
                logger.info(f"[{s}] PENDING {'LONG' if desired > 0 else 'SHORT'} signal, waiting for confirmation on next bar.")
            else:
                # --- Execute trade immediately ---
                tick = mt5.symbol_info_tick(s)
                if not tick: continue
                entry_price = tick.ask if desired > 0 else tick.bid
                sl, tp = calc_sl_tp(desired, entry_price, atr_use, sl_mult, tp_mult)
                sl, tp = enforce_min_distance(s, desired, entry_price, sl, tp)
                stop_dist = abs(entry_price - sl)
                equity = account_equity()
                lots = lots_for_risk(s, equity, stop_dist, trade_bps)

                if SKIP_IF_MINLOTS_EXCEEDS_RISK and estimate_risk_dollars(s, lots, stop_dist) > risk_dollars_from_bps(equity, trade_bps) * RISK_TOLERANCE_MULTIPLIER:
                    logger.warning(f"[{s}] SKIP: Min lot risk exceeds target.")
                    continue

                resp = place_market_order(s, desired, lots, sl=sl, tp=tp)
                logger.info(f"[{s}] EXECUTED immediate {'LONG' if desired>0 else 'SHORT'} lots={lots:.2f}, resp={resp}")
                if resp.get("ok"):
                    open_side[s], last_open_time[s] = desired, nowt_loop
                    trades_today[s] += 1
                    if resp.get("deal_id") and last_feats is not None:
                        log_features = {
                            'technical_p_up': tech_p_up, 'final_p_up': final_p_up,
                            'rsi14': last_feats["rsi14"].iloc[-1], 'macd_hist': last_feats["macd_hist"].iloc[-1], 'rv96': last_feats["rv96"].iloc[-1],
                            'ema_dist_norm': (last_feats["ema50"].iloc[-1] - last_feats["ema200"].iloc[-1]) / (last_feats["rv96"].iloc[-1] + 1e-9),
                            'base_sentiment': latest_sentiment.get(s,{}).get('sentiment_score'),
                        }
                        trade_logger.log_open_trade(resp["deal_id"], s, desired, lots, entry_price, sl, tp, log_features)

        time.sleep(20)

if __name__ == "__main__":
    main()
