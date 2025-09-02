# src/execution/mt5_dual_tf_runner.py
# MT5 runner for low-equity FX on M5 with Telegram alerts and safety rails.

from __future__ import annotations
import os, time, argparse, datetime as dt
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
import MetaTrader5 as mt5

# Load .env for TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
load_dotenv()

from ..utils.config import load_config
from ..features.technical import rsi, macd, realized_vol, atr

# ================= Tunables =================
DRY_RUN        = False
ENTER_THRESH   = 0.65
EXIT_THRESH    = 0.55
MIN_HOLD_MIN   = 15
COOLDOWN_MIN   = 10
REQUIRE_CONSENSUS = False
SKIP_IF_MINLOTS_EXCEEDS_RISK = True
RISK_TOLERANCE_MULTIPLIER = 1.15 # allow ~15% over budget for min lots
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

# ================= Features & signal =================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi14"] = rsi(out["close"], 14)
    m = macd(out["close"])
    out = out.join(m)
    out["rv96"] = realized_vol(out["close"], 96)
    out["atr14"] = atr(out, 14)
    out.dropna(inplace=True)
    return out

def prob_up_from_feats(feats: pd.DataFrame) -> float:
    sig = (-(feats["rsi14"].iloc[-1] - 50)/20.0 +
           (feats["macd_hist"].iloc[-1] / (feats["rv96"].iloc[-1] + 1e-9)))
    return float(1.0 / (1.0 + np.exp(-sig)))

# ================= Account / execution =================
def account_equity() -> float:
    acc = mt5.account_info()
    if not acc:
        raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
    return float(acc.equity)

def symbol_info_or_raise(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info({symbol}) failed")
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return info

def close_symbol_positions(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f"positions_get({symbol}) failed: {mt5.last_error()}")
        return {"ok": False, "msg": str(mt5.last_error())}
    if len(positions) == 0:
        return {"ok": True, "msg": "no positions"}
    tick = mt5.symbol_info_tick(symbol)
    results = []
    for pos in positions:
        lot = pos.volume
        deal_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if deal_type == mt5.ORDER_TYPE_SELL else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": deal_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "close_by_python",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        r = mt5.order_send(req)
        results.append(f"{pos.ticket}:{r.retcode}")
    return {"ok": True, "msg": ";".join(results)}

def calc_sl_tp(side: int, price: float, atr_val: float, sl_mult: float, tp_mult: float) -> Tuple[float, float]:
    atr_val = float(atr_val or 0.0)
    if atr_val <= 0:
        point = 0.0001
        return (price - point, price + point) if side > 0 else (price + point, price - point)
    return (
        float(price - sl_mult * atr_val), float(price + tp_mult * atr_val)
    ) if side > 0 else (
        float(price + sl_mult * atr_val), float(price - tp_mult * atr_val)
    )

def place_market_order(symbol: str, side: int, lots: float, sl: Optional[float]=None, tp: Optional[float]=None) -> dict:
    info = symbol_info_or_raise(symbol)
    tick = mt5.symbol_info_tick(symbol)
    order_type = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
    price = tick.ask if side > 0 else tick.bid

    if sl is not None or tp is not None:
        point = float(getattr(info, "point", 0.0) or 0.0)
        if point > 0:
            if sl is not None: sl = round(sl / point) * point
            if tp is not None: tp = round(tp / point) * point

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "python_entry",
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    if sl is not None: req["sl"] = sl
    if tp is not None: req["tp"] = tp

    if DRY_RUN:
        return {"ok": True, "retcode": "DRY", "comment": "simulated"}

    result = mt5.order_send(req)
    return {"ok": result.retcode == mt5.TRADE_RETCODE_DONE,
            "retcode": result.retcode, "comment": result.comment}

# ---------- Telegram ----------
def _tg_creds(cfg):
    a = cfg.get("alerts", {}) or {}
    if not a.get("telegram_enabled", False):
        logger.debug("Telegram disabled in config.")
        return None, None
    token = os.getenv(a.get("telegram_bot_token_env", ""))
    chat_id = os.getenv(a.get("telegram_chat_id_env", ""))
    if not token or not chat_id:
        logger.warning("Telegram creds missing: token/chat_id not found in env.")
        return None, None
    return token, chat_id

def send_telegram_alert(cfg: dict, text: str) -> bool:
    token, chat_id = _tg_creds(cfg)
    if not token or not chat_id:
        return False
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/sendMessage",
            params={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        if not r.ok:
            logger.error(f"Telegram send failed: {r.status_code} {r.text}")
        else:
            logger.debug("Telegram alert sent.")
        return r.ok
    except Exception as e:
        logger.exception(f"Telegram send exception: {e}")
        return False

def human_pl(v: float) -> str:
    sign = "🟢" if v >= 0 else "🔴"
    return f"{sign} ${abs(v):.2f}"

def build_daily_summary(start_ts: dt.datetime, end_ts: dt.datetime, symbols: list[str]) -> dict:
    deals = mt5.history_deals_get(start_ts, end_ts) or []
    total = 0.0
    wins = 0
    trades = 0
    per_symbol: Dict[str, float] = {s: 0.0 for s in symbols}
    for d in deals:
        try:
            if getattr(d, "entry", None) != mt5.DEAL_ENTRY_OUT:
                continue
            sym = getattr(d, "symbol", "")
            if sym not in per_symbol:
                continue
            magic_ok = int(getattr(d, "magic", 0) or 0) == MAGIC_NUMBER
            comment  = (getattr(d, "comment", "") or "")
            if not (magic_ok or "python_entry" in comment or "close_by_python" in comment):
                continue
            p  = float(getattr(d, "profit", 0.0) or 0.0)
            c  = float(getattr(d, "commission", 0.0) or 0.0)
            sw = float(getattr(d, "swap", 0.0) or 0.0)
            net = p + c + sw
            total += net
            per_symbol[sym] += net
            trades += 1
            if net > 0: wins += 1
        except Exception:
            continue
    winrate = (wins / trades * 100.0) if trades else 0.0
    return {"total": total, "trades": trades, "wins": wins, "winrate": winrate, "per_symbol": per_symbol}

def format_daily_summary(start_ts: dt.datetime, end_ts: dt.datetime, sym_summary: dict) -> str:
    lines = []
    lines.append("<b>📊 Daily Summary</b>")
    lines.append(f"<b>Period:</b> {start_ts:%Y-%m-%d %H:%M} → {end_ts:%Y-%m-%d %H:%M} UTC")
    lines.append(f"<b>Trades:</b> {sym_summary['trades']}   <b>Win rate:</b> {sym_summary['winrate']:.0f}%")
    lines.append(f"<b>Net P/L:</b> {human_pl(sym_summary['total'])}")
    lines.append("<b>By symbol:</b>")
    for s, v in sym_summary["per_symbol"].items():
        lines.append(f"• {s}: {human_pl(v)}")
    return "\n".join(lines)

# ---------- Robust close + P/L (Patch A) ----------
# --- add this helper anywhere above close_positions_with_summary ---
def calc_position_pl(symbol: str, side: str, vol: float,
                     price_open: float, price_close: float) -> float:
    """
    Compute approximate P/L in account currency using tick_size/tick_value.
    side: "LONG" or "SHORT"
    """
    info = symbol_info_or_raise(symbol)
    tick_size = (getattr(info, "trade_tick_size", None) or
                 getattr(info, "tick_size", None) or
                 getattr(info, "point", 0.0))
    tick_value = (getattr(info, "trade_tick_value", None) or
                  getattr(info, "tick_value", None) or
                  (getattr(info, "point", 0.0) * getattr(info, "trade_contract_size", 0.0)))
    if not tick_size or not tick_value:
        return 0.0
    diff = (price_close - price_open)
    if side == "SHORT":
        diff = -diff
    # how many ticks moved * tick value * lots
    ticks = diff / float(tick_size)
    return float(ticks) * float(tick_value) * float(vol)

# ---------------- Main close with summary ----------------
def close_positions_with_summary(symbol: str,
                                 open_time_hint: Optional[dt.datetime] = None):
    """
    Close all positions for 'symbol' and return summary:
      {'total_profit': float, 'legs': [...], 'balance': float}

    Robust matching:
      - wait until positions are actually gone
      - search history window guided by open_time_hint if provided
      - match by position_id first
      - if no deals found, compute synthetic P/L from prices/tick_value
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return {"total_profit": 0.0, "legs": [], "balance": account_equity()}

    before_ts = now_utc()
    legs = []
    ticket_set = set()
    for p in positions:
        legs.append({
            "ticket": int(p.ticket),
            "symbol": p.symbol,
            "side": "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT",
            "volume": float(p.volume),
            "price_open": float(p.price_open),
        })
        ticket_set.add(int(p.ticket))

    # keep a guess of exit price in case broker doesn't record exit deals
    last_tick = mt5.symbol_info_tick(symbol)
    guessed_close_price = (last_tick.bid + last_tick.ask) / 2.0 if last_tick else None

    # close by market
    close_symbol_positions(symbol)

    # wait until flattened (max ~10s)
    for _ in range(40):
        time.sleep(0.25)
        if not (mt5.positions_get(symbol=symbol) or []):
            break

    # history window:
    # if we know when we opened, start a few hours before that; else a wide default
    if open_time_hint:
        start = (open_time_hint - dt.timedelta(hours=8)).replace(tzinfo=dt.timezone.utc)
    else:
        start = before_ts - dt.timedelta(hours=24)
    end = now_utc() + dt.timedelta(seconds=3)

    deals = mt5.history_deals_get(start, end) or []

    total_profit = 0.0
    exit_prices: Dict[int, float] = {}
    found_any = False

    # 1) primary: match exit deals by position_id, include commission/swap
    for d in deals:
        try:
            if getattr(d, "entry", None) != mt5.DEAL_ENTRY_OUT:
                continue
            pid = int(getattr(d, "position_id", 0) or 0)
            if pid in ticket_set:
                p  = float(getattr(d, "profit", 0.0) or 0.0)
                c  = float(getattr(d, "commission", 0.0) or 0.0)
                sw = float(getattr(d, "swap", 0.0) or 0.0)
                total_profit += (p + c + sw)
                exit_prices[pid] = float(getattr(d, "price", 0.0) or 0.0)
                found_any = True
        except Exception:
            continue

    # helper for synthetic P/L
    def _calc_pl(symbol: str, side: str, vol: float,
                 price_open: float, price_close: float) -> float:
        info = symbol_info_or_raise(symbol)
        tick_size = (getattr(info, "trade_tick_size", None) or
                     getattr(info, "tick_size", None) or
                     getattr(info, "point", 0.0))
        tick_value = (getattr(info, "trade_tick_value", None) or
                      getattr(info, "tick_value", None) or
                      (getattr(info, "point", 0.0) * getattr(info, "trade_contract_size", 0.0)))
        if not tick_size or not tick_value:
            return 0.0
        diff = (price_close - price_open)
        if side == "SHORT":
            diff = -diff
        ticks = diff / float(tick_size)
        return float(ticks) * float(tick_value) * float(vol)

    # 2) fallback: synthetic P/L if we didn't get exit deals
    if not found_any:
        for leg in legs:
            price_close = exit_prices.get(leg["ticket"])
            if not price_close:
                # try last exit deal by symbol
                sym_deals = [d for d in deals
                             if getattr(d, "symbol", "") == symbol and
                                getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT]
                if sym_deals:
                    price_close = float(getattr(sym_deals[-1], "price", 0.0) or 0.0)
            if not price_close:
                price_close = guessed_close_price or leg["price_open"]  # worst-case: flat
            leg["price_close"] = float(price_close)
            total_profit += _calc_pl(
                symbol=leg["symbol"],
                side=leg["side"],
                vol=leg["volume"],
                price_open=leg["price_open"],
                price_close=leg["price_close"],
            )
    else:
        for leg in legs:
            if leg["ticket"] in exit_prices:
                leg["price_close"] = exit_prices[leg["ticket"]]

    balance = account_equity()
    return {"total_profit": float(total_profit), "legs": legs, "balance": float(balance)}

def format_close_message(symbol: str, summary: dict, held_min: float, reason: str) -> str:
    lines = []
    lines.append(f"<b>✅ Trade Closed</b>")
    lines.append(f"<b>Symbol:</b> {symbol}")
    lines.append(f"<b>Reason:</b> {reason}")
    lines.append(f"<b>Held:</b> {held_min:.1f} min")
    if summary["legs"]:
        for i, leg in enumerate(summary["legs"], 1):
            entry = f'{leg["price_open"]:.5f}' if leg["price_open"] is not None else "?"
            exitp = f'{leg.get("price_close", 0.0):.5f}' if leg.get("price_close") else "?"
            lines.append(f"• {i}) {leg['side']} {leg['volume']:.2f} @ {entry} → {exitp}")
    lines.append(f"<b>P/L:</b> {human_pl(summary['total_profit'])}")
    lines.append(f"<b>Balance:</b> ${summary['balance']:.2f}")
    return "\n".join(lines)

# ================= Risk sizing =================
def first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def risk_dollars_from_bps(equity: float, bps: float) -> float:
    return equity * (bps / 10_000.0)

def clamp_to_step(x: float, step: float, min_v: float, max_v: float) -> float:
    if step <= 0:
        step = 0.01
    x = max(min_v, min(max_v, x))
    steps = round(x / step)
    return round(steps * step, 3)

def lots_for_risk(symbol: str, equity: float, stop_distance_price: float, bps: float) -> float:
    info = symbol_info_or_raise(symbol)
    tick_size = first_non_none(
        getattr(info, "trade_tick_size", None),
        getattr(info, "tick_size", None),
        getattr(info, "point", None),
    )
    tick_value = first_non_none(
        getattr(info, "trade_tick_value", None),
        getattr(info, "tick_value", None),
        getattr(info, "trade_tick_value_profit", None),
        getattr(info, "trade_tick_value_loss", None),
        (getattr(info, "point", 0.0) * getattr(info, "trade_contract_size", 0.0)) or None,
    )
    vol_min  = float(getattr(info, "volume_min", 0.01) or 0.01)
    vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)
    vol_max  = float(getattr(info, "volume_max", 100.0) or 100.0)

    if not tick_size or not tick_value or stop_distance_price <= 0:
        logger.warning(f"[{symbol}] sizing fallback — tick_size={tick_size}, tick_value={tick_value}, stop={stop_distance_price}")
        return clamp_to_step(0.01, vol_step, vol_min, vol_max)

    risk_per_lot = (stop_distance_price / float(tick_size)) * float(tick_value)
    risk_target  = risk_dollars_from_bps(equity, bps)
    if risk_per_lot <= 0:
        logger.warning(f"[{symbol}] risk_per_lot<=0 fallback; tick_size={tick_size}, tick_value={tick_value}")
        return clamp_to_step(0.01, vol_step, vol_min, vol_max)

    lots_raw = risk_target / risk_per_lot
    lots = clamp_to_step(lots_raw, vol_step, vol_min, vol_max)

    logger.info(f"[{symbol}] risk size: tick_size={tick_size} tick_value={tick_value} stop={stop_distance_price:.6f} "
                f"risk_per_lot=${risk_per_lot:.2f} target=${risk_target:.2f} → lots={lots}")
    return lots

def estimate_risk_dollars(symbol: str, lots: float, stop_distance_price: float) -> float:
    info = symbol_info_or_raise(symbol)
    tick_size = first_non_none(
        getattr(info, "trade_tick_size", None),
        getattr(info, "tick_size", None),
        getattr(info, "point", None),
    )
    tick_value = first_non_none(
        getattr(info, "trade_tick_value", None),
        getattr(info, "tick_value", None),
        getattr(info, "trade_tick_value_profit", None),
        getattr(info, "trade_tick_value_loss", None),
        (getattr(info, "point", 0.0) * getattr(info, "trade_contract_size", 0.0)) or None,
    )
    if not tick_size or not tick_value or stop_distance_price <= 0:
        return float("inf")
    risk_per_lot = (stop_distance_price / float(tick_size)) * float(tick_value)
    return float(risk_per_lot) * float(lots)

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None, help="Comma-separated override (e.g., EURUSD,GBPUSD)")
    args = ap.parse_args()

    cfg = load_config()
    symbols = (args.symbols.split(",") if args.symbols else cfg.get("symbols", []))
    tfs = cfg.get("timeframes", ["M5"])
    assert all(tf in TF_MAP for tf in tfs), f"Unsupported TF in config; allowed {list(TF_MAP)}"
    tf_codes = {tf: TF_MAP[tf] for tf in tfs}

    risk_cfg = cfg["risk"]
    sl_mult  = float(risk_cfg.get("stop_atr_mult", 1.8))
    tp_mult  = float(risk_cfg.get("takeprofit_atr_mult", 2.6))
    trade_bps = float(risk_cfg.get("max_risk_per_trade_bps", 100.0))

    kill = risk_cfg["kill_switch"]
    daily_dd_max   = float(kill.get("max_intraday_drawdown_pct", 2.0))
    max_trades_day = int(kill.get("max_daily_trades_per_symbol", 6))

    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5 initialize() failed: {mt5.last_error()}")
    logger.info(f"MT5 initialized. Symbols={symbols} TFs={tfs}")

    # ---- State ----
    last_bar_time: Dict[str, Dict[str, pd.Timestamp]] = {s: {tf: pd.Timestamp(0, tz="UTC") for tf in tfs} for s in symbols}
    tf_side: Dict[str, Dict[str, int]] = {s: {tf: 0 for tf in tfs} for s in symbols}
    open_side: Dict[str, int] = {s: 0 for s in symbols}
    trades_today: Dict[str, int] = {s: 0 for s in symbols}
    last_open_time: Dict[str, dt.datetime]  = {s: dt.datetime.min.replace(tzinfo=dt.timezone.utc) for s in symbols}
    last_close_time: Dict[str, dt.datetime] = {s: dt.datetime.min.replace(tzinfo=dt.timezone.utc) for s in symbols}

    # --- Sync with any live positions in MT5 (on startup) ---
    positions = mt5.positions_get()
    if positions:
        for p in positions:
            if p.symbol not in symbols:
                continue
            side = +1 if p.type == mt5.POSITION_TYPE_BUY else -1
            open_side[p.symbol] = side
            t_open = None
            if getattr(p, "time", None):
                try:
                    t_open = dt.datetime.fromtimestamp(int(p.time), tz=dt.timezone.utc)
                except Exception:
                    t_open = None
            if t_open:
                last_open_time[p.symbol] = t_open
            po = float(getattr(p, "price_open", 0.0) or 0.0)
            logger.info(
                f"{p.symbol}: detected {'LONG' if side>0 else 'SHORT'} {p.volume} lots "
                f"@ {po:.5f} (since {last_open_time[p.symbol]})"
            )

    day_start = today_utc_start()
    equity_day_open = account_equity()
    paused_for_dd = False

    logger.info("🟢 Bot started (M5 FX, low-equity safe).")

    while True:
        # Daily rollover summary + reset
        if now_utc() >= day_start + dt.timedelta(days=1):
            try:
                end_ts = now_utc()
                summary = build_daily_summary(day_start, end_ts, symbols)
                msg = format_daily_summary(day_start, end_ts, summary)
                send_telegram_alert(cfg, msg)
            except Exception as e:
                logger.exception(f"Daily summary failed: {e}")

            day_start = today_utc_start()
            equity_day_open = account_equity()
            trades_today = {s: 0 for s in symbols}
            paused_for_dd = False
            logger.info("🔄 New UTC day: counters reset")

        # Kill-switch
        eq = account_equity()
        dd_pct = (eq - equity_day_open) / max(equity_day_open, 1e-9) * 100.0
        if dd_pct <= -abs(daily_dd_max) and not paused_for_dd:
            if not DRY_RUN:
                for s in symbols:
                    close_symbol_positions(s)
            paused_for_dd = True
            logger.error(f"⛔ Kill-switch: DD {dd_pct:.2f}% ≤ {daily_dd_max}% — flattened & paused")
        if paused_for_dd:
            time.sleep(60)
            continue

        if not within_trading_window(cfg):
            time.sleep(10)
            continue

        for s in symbols:
            if trades_today[s] >= max_trades_day:
                continue

            # ---------- Patch B: broker/TP/SL watcher ----------
            cur_positions = mt5.positions_get(symbol=s) or []
            has_live = len(cur_positions) > 0
            if open_side[s] != 0 and not has_live:
                held_min = (now_utc() - last_open_time[s]).total_seconds() / 60.0
                summary = close_positions_with_summary(s, open_time_hint=last_open_time[s])
                msg = format_close_message(s, summary, held_min, reason="Broker exit (TP/SL/Manual)")
                send_telegram_alert(cfg, msg)
                last_close_time[s] = now_utc()
                open_side[s] = 0
                trades_today[s] += 1
                continue  # move to next symbol this cycle

            tf_probs = {}
            tf_atr   = {}
            tf_price = {}
            updated_any = False

            for tf in tfs:
                df = fetch_bars(s, tf_codes[tf], count=400)
                if df.empty or len(df) < 120:
                    continue
                last = df.index[-1]
                if last == last_bar_time[s][tf]:
                    continue
                last_bar_time[s][tf] = last
                updated_any = True

                feats = build_features(df.iloc[:-1])  # closed bar only
                if feats.empty:
                    continue
                p = prob_up_from_feats(feats)
                tf_probs[tf]  = p
                tf_atr[tf]    = float(feats["atr14"].iloc[-1])
                tf_price[tf]  = float(df["close"].iloc[-2])

                if p >= ENTER_THRESH: tf_side[s][tf] = +1
                elif p <= 1 - ENTER_THRESH: tf_side[s][tf] = -1
                else: tf_side[s][tf] = 0

            if not updated_any:
                continue

            # Combine TFs
            sides = [tf_side[s][tf] for tf in tfs]
            if REQUIRE_CONSENSUS and len(tfs) > 1:
                if len(set([x for x in sides if x != 0])) == 1 and any(x != 0 for x in sides):
                    raw_side = sides[0] if sides[0] != 0 else [x for x in sides if x != 0][0]
                    size_factor = 1.0
                else:
                    raw_side = 0; size_factor = 0.0
            else:
                if any(x != 0 for x in sides):
                    raw_side = [x for x in sides if x != 0][0]; size_factor = 1.0
                else:
                    raw_side = 0; size_factor = 0.0

            atr_vals= [v for v in tf_atr.values() if v is not None]
            prices  = [v for v in tf_price.values() if v is not None]
            if not atr_vals or not prices:
                continue
            atr_use   = atr_vals[-1]
            price_use = prices[-1]
            probs     = [p for p in tf_probs.values()]
            conf      = float(np.median(probs)) if probs else 0.5

            # Hysteresis / hold / cooldown
            now = now_utc()
            held_min = (now - last_open_time[s]).total_seconds() / 60.0
            cool_min = (now - last_close_time[s]).total_seconds() / 60.0
            current = open_side[s]
            desired = current

            if current == 0:
                if cool_min >= COOLDOWN_MIN:
                    if raw_side > 0 and conf >= ENTER_THRESH:  desired = +1
                    elif raw_side < 0 and conf >= ENTER_THRESH: desired = -1
            else:
                if held_min >= MIN_HOLD_MIN:
                    if conf <= EXIT_THRESH or raw_side == 0:
                        desired = 0
                    elif raw_side != current and conf >= ENTER_THRESH:
                        desired = raw_side

            # Execute
            if desired == current:
                pass

            elif desired == 0:
                if current != 0:
                    if not DRY_RUN:
                        summary = close_positions_with_summary(s, open_time_hint=last_open_time[s])
                        msg = format_close_message(s, summary, held_min, reason="Signal exit / flat")
                        send_telegram_alert(cfg, msg)
                    logger.info(f"{s}: FLAT (held {held_min:.1f}m, conf={conf:.2f})")
                last_close_time[s] = now
                open_side[s] = 0
                trades_today[s] += 1

            else:
                if current != 0 and desired != current:
                    if not DRY_RUN:
                        summary = close_positions_with_summary(s, open_time_hint=last_open_time[s])
                        msg = format_close_message(s, summary, held_min, reason="Flip")
                        send_telegram_alert(cfg, msg)
                    logger.info(f"{s}: FLIP {('LONG' if current>0 else 'SHORT')}→{('LONG' if desired>0 else 'SHORT')}")

                sl, tp = calc_sl_tp(desired, price_use, atr_use, sl_mult, tp_mult)
                stop_dist = abs(price_use - sl)
                equity = account_equity()

                lots_base = lots_for_risk(s, equity, stop_dist, trade_bps)
                lots = max(round(lots_base * size_factor, 3), 0.01)

                if SKIP_IF_MINLOTS_EXCEEDS_RISK:
                    risk_target = risk_dollars_from_bps(equity, trade_bps) * RISK_TOLERANCE_MULTIPLIER
                    est_risk = estimate_risk_dollars(s, lots, stop_dist)
                    if est_risk > risk_target + 1e-9:
                        logger.info(f"{s}: SKIP — min lot risk ${est_risk:.2f} > target ${risk_target:.2f} "
                                    f"(stop={stop_dist:.5f}, lots={lots})")
                        continue

                resp = {"ok": True, "retcode": "DRY", "comment": "simulated"} if DRY_RUN \
                       else place_market_order(s, desired, lots, sl=sl, tp=tp)
                logger.info(f"{s}: {'LONG' if desired>0 else 'SHORT'} lots={lots} conf={conf:.2f} "
                            f"SL={sl:.5f} TP={tp:.5f} (risk {trade_bps}bps) resp={resp}")
                last_open_time[s] = now
                open_side[s] = desired
                trades_today[s] += 1

        time.sleep(30)  # check twice a minute

if __name__ == "__main__":
    main()
