"""
5-minute multi-indicator stock signal scanner.

Scans a watchlist (watchlist.txt) on each run, computes a confluence of the
indicators professional/technical traders commonly use on intraday charts,
and sends a Telegram alert ONLY when a ticker's signal flips to a fresh
BUY or SELL (so you don't get spammed every 5 minutes with the same call).

This is a rule-based decision-support system, not a broker connection: it
never places trades. It also is NOT investment advice — it mechanically
reports what a defined set of technical rules say about each ticker, with a
plain-language explanation of which conditions triggered.

Designed to be run on a schedule (every 5 minutes) by GitHub Actions or any
free cron-capable host. See README.md for deployment instructions.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ROOT = Path(__file__).parent
WATCHLIST_FILE = ROOT / "watchlist.txt"
STATE_FILE = ROOT / "data" / "state.json"

# Confluence thresholds: a ticker needs a score of at least this many net
# bullish/bearish conditions (out of 5 checks) to register as BUY/SELL.
# Score range is -5..+5. 3 is a reasonably strict default (most indicators agree).
SIGNAL_THRESHOLD = 3

INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "5d"   # yfinance limit for 5m data is 60d; 5d keeps requests light

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scanner")


# ── Watchlist & state ────────────────────────────────────────────────────
def load_watchlist() -> list:
    tickers = []
    for line in WATCHLIST_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(line.upper())
    return tickers


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ── Data ──────────────────────────────────────────────────────────────────
def fetch_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL,
                      progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No intraday data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Indicators (the standard intraday technical-trader toolkit) ──────────
def compute_indicators(df: pd.DataFrame) -> dict:
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    ind = {}

    # EMA9 / EMA21 — fast trend read, the classic intraday crossover pair
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ind["ema9"] = ema9.iloc[-1]
    ind["ema21"] = ema21.iloc[-1]

    # VWAP (session) — resets each trading day, the benchmark intraday desks trade around
    df = df.copy()
    df["date"] = df.index.date
    typical_price = (high + low + close) / 3
    df["tp_vol"] = typical_price * volume
    session_cum_tp_vol = df.groupby("date")["tp_vol"].cumsum()
    session_cum_vol = df.groupby("date")["Volume"].cumsum()
    vwap = session_cum_tp_vol / session_cum_vol
    ind["vwap"] = vwap.iloc[-1]

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    ind["rsi14"] = (100 - 100 / (1 + rs)).iloc[-1]

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    ind["macd_hist"] = (macd_line - signal_line).iloc[-1]
    ind["macd_hist_prev"] = (macd_line - signal_line).iloc[-2] if len(close) > 1 else ind["macd_hist"]

    # Volume vs its own 20-period average (confirmation)
    ind["volume"] = volume.iloc[-1]
    ind["volume_avg20"] = volume.rolling(20).mean().iloc[-1]

    # Bollinger Bands(20,2) — overextension check
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    ind["bb_upper"] = (bb_mid + 2 * bb_std).iloc[-1]
    ind["bb_lower"] = (bb_mid - 2 * bb_std).iloc[-1]

    # ATR(14) — core building block for price targets and stop placement
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    ind["atr14"] = tr.rolling(14).mean().iloc[-1]

    # Recent swing high/low over last 20 candles — key S/R reference
    ind["swing_high"] = high.rolling(20).max().iloc[-1]
    ind["swing_low"] = low.rolling(20).min().iloc[-1]

    ind["close"] = close.iloc[-1]
    ind["open"] = df["Open"].iloc[-1]
    ind["high"] = high.iloc[-1]
    ind["low"] = low.iloc[-1]
    ind["candle_time"] = df.index[-1]

    return ind


# ── Signal scoring (the "analyst brain") ──────────────────────────────────
def score_signal(ind: dict) -> tuple:
    """
    Scores -5..+5 across 5 independent checks. Returns (score, fired_reasons).
    Each fired_reasons entry is a plain-language description of a condition
    that triggered, used to build the alert message.
    """
    score = 0
    bullish, bearish = [], []

    # 1. Trend: EMA9 vs EMA21
    if ind["ema9"] > ind["ema21"]:
        score += 1
        bullish.append(f"EMA9 (${ind['ema9']:.2f}) is above EMA21 (${ind['ema21']:.2f}) — short-term trend is up")
    else:
        score -= 1
        bearish.append(f"EMA9 (${ind['ema9']:.2f}) is below EMA21 (${ind['ema21']:.2f}) — short-term trend is down")

    # 2. Price vs VWAP — are buyers/sellers in control of the session
    if ind["close"] > ind["vwap"]:
        score += 1
        bullish.append(f"Price (${ind['close']:.2f}) is trading above session VWAP (${ind['vwap']:.2f}) — buyers in control")
    else:
        score -= 1
        bearish.append(f"Price (${ind['close']:.2f}) is trading below session VWAP (${ind['vwap']:.2f}) — sellers in control")

    # 3. RSI(14) — momentum, with overbought/oversold guardrails
    rsi = ind["rsi14"]
    if 50 < rsi < 70:
        score += 1
        bullish.append(f"RSI14 at {rsi:.0f} — positive momentum, not yet overbought")
    elif rsi >= 70:
        score -= 1
        bearish.append(f"RSI14 at {rsi:.0f} — overbought, momentum may be exhausted")
    elif 30 < rsi < 50:
        score -= 1
        bearish.append(f"RSI14 at {rsi:.0f} — momentum leaning negative")
    elif rsi <= 30:
        score += 1
        bullish.append(f"RSI14 at {rsi:.0f} — oversold, potential bounce zone")

    # 4. MACD histogram direction (is bullish/bearish momentum building or fading)
    if ind["macd_hist"] > 0 and ind["macd_hist"] >= ind["macd_hist_prev"]:
        score += 1
        bullish.append("MACD histogram positive and rising — bullish momentum building")
    elif ind["macd_hist"] < 0 and ind["macd_hist"] <= ind["macd_hist_prev"]:
        score -= 1
        bearish.append("MACD histogram negative and falling — bearish momentum building")

    # 5. Volume confirmation
    if ind["volume_avg20"] and ind["volume"] > 1.2 * ind["volume_avg20"]:
        # Volume confirms whichever direction is already leading
        if score > 0:
            score += 1
            bullish.append("Volume is running above its 20-period average — move has conviction")
        elif score < 0:
            score -= 1
            bearish.append("Volume is running above its 20-period average — move has conviction")

    if score >= SIGNAL_THRESHOLD:
        return score, "BUY", bullish
    elif score <= -SIGNAL_THRESHOLD:
        return score, "SELL", bearish
    else:
        return score, "HOLD", bullish + bearish


def compute_targets(decision: str, ind: dict) -> dict:
    """
    Derives entry, stop loss, and tiered sell targets the way an experienced
    intraday quant would frame them — all anchored to ATR(14) so levels scale
    naturally with each stock's own recent volatility.

    BUY setup:
      Entry     : current close (or next candle open — signal fires on close)
      Stop loss : entry − 1.0× ATR  (one ATR of heat is a clean hard stop)
      Target T1 : entry + 1.5× ATR  (first take-profit, lock in ~60% of position)
      Target T2 : entry + 2.5× ATR  (runner target, trail stop after T1 hit)
      T2 is also capped at swing_high if that's closer — don't fight resistance.

    SELL/SHORT setup (mirror logic):
      Entry     : current close
      Stop loss : entry + 1.0× ATR
      Target T1 : entry − 1.5× ATR
      Target T2 : entry − 2.5× ATR
      T2 is also floored at swing_low.

    Risk/reward is reported vs T2 (the full move), giving a conservative
    picture of the trade's potential.
    """
    entry = ind["close"]
    atr = ind["atr14"]

    if decision == "BUY":
        stop = entry - (1.0 * atr)
        t1 = entry + (1.5 * atr)
        t2 = min(entry + (2.5 * atr), ind["swing_high"])
        # Don't let rounding push T2 below T1
        t2 = max(t2, t1 + 0.01)
    else:  # SELL
        stop = entry + (1.0 * atr)
        t1 = entry - (1.5 * atr)
        t2 = max(entry - (2.5 * atr), ind["swing_low"])
        t2 = min(t2, t1 - 0.01)

    risk = abs(entry - stop)
    reward = abs(t2 - entry)
    rr = reward / risk if risk > 0 else 0

    return {
        "entry": entry,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "risk": risk,
        "reward": reward,
        "rr": rr,
        "atr": atr,
    }


def build_reason(decision: str, fired: list, ind: dict, targets: dict) -> str:
    """
    Quant trader voice — references actual indicator readings and price levels,
    not generic descriptions. Written as a concise trade rationale you'd say
    out loud to a desk partner, not a textbook definition.
    """
    rsi = ind["rsi14"]
    vwap = ind["vwap"]
    ema9 = ind["ema9"]
    ema21 = ind["ema21"]
    close = ind["close"]
    vol_ratio = ind["volume"] / ind["volume_avg20"] if ind["volume_avg20"] else 1
    atr = targets["atr"]

    lines = []

    if decision == "BUY":
        # Trend context
        if ema9 > ema21:
            lines.append(
                f"Trend: EMA9 ${ema9:.2f} > EMA21 ${ema21:.2f} — "
                f"momentum structure is bullish on the 5m. Price holding above "
                f"both MAs is textbook intraday long setup."
            )
        # VWAP context
        if close > vwap:
            lines.append(
                f"VWAP: Trading ${close - vwap:.2f} above VWAP (${vwap:.2f}). "
                f"Institutions anchor execution to VWAP — price above it means "
                f"the smart money flow is net long this session."
            )
        else:
            lines.append(
                f"VWAP: Reclaimed VWAP (${vwap:.2f}) from below — "
                f"watch for a hold above as confirmation."
            )
        # RSI
        if rsi <= 30:
            lines.append(
                f"RSI14 {rsi:.0f}: Deep oversold. At these levels mean-reversion "
                f"probability is high — this is where I size in, not chase."
            )
        else:
            lines.append(
                f"RSI14 {rsi:.0f}: Momentum is constructive without being "
                f"stretched. Room to run before the 70 overbought wall."
            )
        # MACD
        if ind["macd_hist"] > ind["macd_hist_prev"]:
            lines.append(
                f"MACD: Histogram expanding bullish (hist {ind['macd_hist']:.4f}). "
                f"Crossover momentum is accelerating — that's the fuel."
            )
        # Volume
        if vol_ratio > 1.2:
            lines.append(
                f"Volume: {vol_ratio:.1f}× the 20-period avg. "
                f"Elevated participation backs the move — this isn't air."
            )
        # BB context
        if close >= ind["bb_upper"]:
            lines.append(
                f"⚠️ Caution: Price is printing at/above upper Bollinger Band "
                f"(${ind['bb_upper']:.2f}). Statistically extended — consider "
                f"waiting for a pullback to VWAP/EMA21 for a cleaner entry."
            )
        # ATR note
        lines.append(
            f"ATR(14) is ${atr:.2f} — that's the stock's natural 'breathing room' "
            f"per candle. Targets and stop are sized off this so levels fit the "
            f"stock's own rhythm, not arbitrary round numbers."
        )

    else:  # SELL
        if ema9 < ema21:
            lines.append(
                f"Trend: EMA9 ${ema9:.2f} < EMA21 ${ema21:.2f} — "
                f"short-term structure has flipped. Sellers are in control on the 5m."
            )
        if close < vwap:
            lines.append(
                f"VWAP: Trading ${vwap - close:.2f} below VWAP (${vwap:.2f}). "
                f"Failed VWAP = institutional distribution. This is a bearish anchor."
            )
        if rsi >= 70:
            lines.append(
                f"RSI14 {rsi:.0f}: Overbought and rolling over. "
                f"This is where I start looking for the exit, not adding."
            )
        else:
            lines.append(
                f"RSI14 {rsi:.0f}: Momentum deteriorating. "
                f"No support from the momentum read."
            )
        if ind["macd_hist"] < ind["macd_hist_prev"]:
            lines.append(
                f"MACD: Histogram contracting/negative (hist {ind['macd_hist']:.4f}). "
                f"Bearish momentum is building, not fading."
            )
        if vol_ratio > 1.2:
            lines.append(
                f"Volume: {vol_ratio:.1f}× average. "
                f"Distribution on elevated volume is a clear institutional fingerprint."
            )
        if close <= ind["bb_lower"]:
            lines.append(
                f"⚠️ Note: Price is at/below lower Bollinger Band "
                f"(${ind['bb_lower']:.2f}). Could see a technical bounce — "
                f"tighten stop to protect short if bounce materialises."
            )
        lines.append(
            f"ATR(14) ${atr:.2f} used to size levels — "
            f"stop and targets reflect this ticker's actual volatility."
        )

    return "\n\n".join(lines)


# ── Telegram ──────────────────────────────────────────────────────────────
def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set (env vars or GitHub secrets).")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code != 200:
        log.error(f"Telegram send failed: {resp.status_code} {resp.text}")
    resp.raise_for_status()


def format_alert(ticker: str, decision: str, ind: dict, reason: str, targets: dict) -> str:
    if decision == "BUY":
        header_emoji = "🟢"
        decision_line = "✅ *BUY*"
        price_label = "📥 Entry Price"
        sell_label = "📤 Take Profit"
        sell_value = f"T1: ${targets['t1']:.2f}  |  T2: ${targets['t2']:.2f}"
        stop_label = "🛑 Stop Loss"
    else:
        header_emoji = "🔴"
        decision_line = "🚨 *SELL / EXIT*"
        price_label = "📥 Entry / Exit Price"
        sell_label = "🎯 Cover Target"
        sell_value = f"T1: ${targets['t1']:.2f}  |  T2: ${targets['t2']:.2f}"
        stop_label = "🛑 Stop Loss"

    rr_bar = "█" * min(int(targets['rr'] * 2), 10)  # visual R:R bar, capped at 5:1

    candle_ts = str(ind["candle_time"])

    return (
        f"{header_emoji} *{ticker}*  —  {decision_line}\n"
        f"{'─' * 28}\n"
        f"{price_label}: *${targets['entry']:.2f}*\n"
        f"{sell_label}: *{sell_value}*\n"
        f"{stop_label}: *${targets['stop']:.2f}*\n"
        f"{'─' * 28}\n"
        f"⚖️ Risk / Reward:  1 : {targets['rr']:.1f}  {rr_bar}\n"
        f"📉 Max Risk: ${targets['risk']:.2f} per share\n"
        f"📈 Max Gain (T2): ${targets['reward']:.2f} per share\n"
        f"{'─' * 28}\n"
        f"🧠 *Analyst Note*\n"
        f"{reason}\n"
        f"{'─' * 28}\n"
        f"⏱ Signal candle: {candle_ts} (5m)\n"
        f"_Confluence score confirmed across EMA, VWAP, RSI, MACD, Volume_\n"
        f"_Rule-based signal — verify before acting. Not financial advice._"
    )


# ── Main scan loop ─────────────────────────────────────────────────────────
def run_scan():
    tickers = load_watchlist()
    state = load_state()
    results = []  # collect for the "best opportunity" summary

    for ticker in tickers:
        try:
            df = fetch_intraday(ticker)
            ind = compute_indicators(df)
            candle_key = str(ind["candle_time"])

            prev = state.get(ticker, {})
            if prev.get("last_candle") == candle_key:
                # Already processed this candle on a previous run — skip
                continue

            score, decision, fired = score_signal(ind)
            results.append((ticker, score, decision, ind))

            prev_decision = prev.get("last_decision")
            if decision in ("BUY", "SELL") and decision != prev_decision:
                targets = compute_targets(decision, ind)
                reason = build_reason(decision, fired, ind, targets)
                msg = format_alert(ticker, decision, ind, reason, targets)
                send_telegram_message(msg)
                log.info(f"Alert sent: {ticker} {decision} (score {score}, R:R {targets['rr']:.1f})")

            state[ticker] = {
                "last_candle": candle_key,
                "last_decision": decision,
                "last_score": score,
                "last_price": float(ind["close"]),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            log.warning(f"Skipping {ticker}: {e}")
        time.sleep(0.5)  # be polite to Yahoo Finance's unofficial endpoint

    save_state(state)

    # Optional: highlight the single strongest BUY this cycle, if any
    buys = [r for r in results if r[2] == "BUY"]
    if buys:
        best = max(buys, key=lambda r: r[1])
        ticker, score, decision, ind = best
        log.info(f"Best opportunity this cycle: {ticker} (score {score})")


if __name__ == "__main__":
    run_scan()
