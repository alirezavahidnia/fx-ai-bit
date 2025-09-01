# src/execution/mt5_dual_tf_runner.py
# MT5 runner for low-equity FX on M5.
# - Direct MetaTrader5 API (no EA bridge)
# - Risk-based lot sizing + skip if min-lot exceeds budget
# - Hysteresis (enter/exit), min-hold, cooldown
# - Close/flip Telegram alerts (Trade info, P/L, Balance)
# - Daily drawdown kill-switch + daily trade cap
# - Trading window (UTC)

from __future__ import annotations
import os, time, argparse, datetime as dt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
import MetaTrader5 as mt5

# Load env containing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
load_dotenv("telegram_chat_id_env.env")

from ..utils.config import load_config
from ..features.technical import rsi, macd, realized_vol, atr


# ================= Tunables for M5 FX =================
DRY_RUN        = False          # True = simulate; False = send orders
ENTER_THRESH   = 0.62           # stronger to enter on noisy M5
EXIT_THRESH    = 0.55           # softer to exit
MIN_HOLD_MIN   = 15             # M5: hold at least 15m
COOLDOWN_MIN   = 10             # wait 10m after close before re-entering
REQUIRE_CONSENSUS = False       # single timeframe by default; keep False
SKIP_IF_MINLOTS_EXCEEDS_RISK = True
RISK_TOLERANCE_MULTIPLIER = 1.0 # 1.0 strict; 1.2 = allow 20% over budget


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
            "magic": 202501,
            "comment": "close_by_python",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        r = mt5.order_send(req)
        results.append(f"{pos.ticket}:{r.retcode}")
    return {"ok": True, "msg": ";".join(results)}

def calc_sl_tp(side: int, price: float, atr_val: float, sl_mult: float, tp_mult: float) -> Tuple[float, float]:
    if side > 0:
        return float(price - sl_mult * atr_val), float(price + tp_mult * atr_val)
    else:
        return float(price + sl_mult * atr_val), float(price - tp_mult * atr_val)

def place_market_order(symbol: str, side: int, lots: float, sl: float | None=None, tp: float | None=None) -> dict:
    info = symbol_info_or_raise(symbol)
    tick = mt5.symbol_info_tick(symbol)
    order_type = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
    price = tick.ask if side > 0 else tick.bid

    # snap SL/TP to symbol point grid
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
        "magic": 202501,
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


# ---------- Telegram alerts ----------
def _tg_creds(cfg):
    a = cfg.get("alerts", {}) or {}
    if not a.get("telegram_enabled", False):
        return None, None
    token = os.getenv(a.get("telegram_bot_token_env", ""))
    chat_id = os.getenv(a.get("telegram_chat_id_env", ""))
    if not token or not chat_id:
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
        return r.ok
    except Exception:
        return False

def human_pl(v: float) -> str:
    sign = "🟢" if v >= 0 else "🔴"
    return f"{sign} ${abs(v):.2f}"

def close_positions_with_summary(symbol: str):
    """
    Close all positions for 'symbol' and return summary:
    {'total_profit': float, 'legs': [...], 'balance': float}
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return {"total_profit": 0.0, "legs": [], "balance": account_equity()}

    legs = []
    for p in positions:
        legs.append({
            "ticket": p.ticket,
            "symbol": p.symbol,
            "side": "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT",
            "volume": float(p.volume),
            "price_open": float(p.price_open),
        })

    # Close by market
    close_symbol_positions(symbol)

    # Let MT5 register deals
    time.sleep(1.0)

    # Accumulate realized P/L for those tickets
    now = now_utc()
    deals = mt5.history_deals_get(now - dt.timedelta(hours=1), now) or []
    ticket_set = {leg["ticket"] for leg in legs}
    total_profit = 0.0
    exit_prices = {}

    for d in deals:
        if getattr(d, "position_id", None) in ticket_set and getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT:
            total_profit += float(getattr(d, "profit", 0.0))
            exit_prices[int(d.position_id)] = float(getattr(d, "price", 0.0))

    for leg in legs:
        leg["price_close"] = exit_prices.get(leg["ticket"], None)

    balance = account_equity()
    return {"total_profit": total_profit, "legs": legs, "balance": balance}

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


# ================= Robust lot sizing (FX-friendly) =================
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
    """
    Compute lots so loss at SL ≈ equity * (bps / 1e4), robust across MT5 builds.
    """
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
    trade_bps = float(risk_cfg.get("max_risk_per_trade_bps", 100.0))  # 1% default

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

    day_start = today_utc_start()
    equity_day_open = account_equity()
    paused_for_dd = False

    logger.info("🟢 Bot started (M5 FX, low-equity safe).")

    while True:
        # New day reset
        if now_utc() >= day_start + dt.timedelta(days=1):
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

            tf_probs = {}
            tf_atr   = {}
            tf_price = {}
            updated_any = False

            for tf in tfs:
                df = fetch_bars(s, tf_codes[tf], count=400)  # M5: 400 bars ~ 33h
                if df.empty or len(df) < 120:  # warmup
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
                    raw_side = 0
                    size_factor = 0.0
            else:
                # single TF or permissive multi-TF
                if any(x != 0 for x in sides):
                    raw_side = [x for x in sides if x != 0][0]
                    size_factor = 1.0
                else:
                    raw_side = 0
                    size_factor = 0.0

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

            # ---------------- Execute if change ----------------
            if desired == current:
                pass  # no action

            elif desired == 0:
                # Close & alert (going flat)
                if current != 0:
                    if not DRY_RUN:
                        summary = close_positions_with_summary(s)
                        msg = format_close_message(s, summary, held_min, reason="Signal exit / flat")
                        send_telegram_alert(cfg, msg)
                    logger.info(f"{s}: FLAT (held {held_min:.1f}m, conf={conf:.2f})")
                last_close_time[s] = now
                open_side[s] = 0
                trades_today[s] += 1

            else:
                # We want to be long/short now
                if current != 0 and desired != current:
                    # Flip: close existing first and alert
                    if not DRY_RUN:
                        summary = close_positions_with_summary(s)
                        msg = format_close_message(s, summary, held_min, reason="Flip")
                        send_telegram_alert(cfg, msg)
                    logger.info(f"{s}: FLIP {('LONG' if current>0 else 'SHORT')}→{('LONG' if desired>0 else 'SHORT')}")

                # SL/TP & risk
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

        time.sleep(30)  # M5 cadence, check twice a minute for bar-close


if __name__ == "__main__":
    main()
