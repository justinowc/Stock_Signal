"""
Agentic Stock Signal Bot — Full upgrade.

What this does:
  • Multi-timeframe analysis: 4H (bias) → 1H (direction) → 5M (entry)
  • Claude (Anthropic API) as the analyst brain — real reasoning, not templates
  • Pre-market brief, post-market recap, and 24hr news monitoring
  • 3 take-profit levels + stop loss + R:R + position sizing note
  • iPhone-optimised Telegram messages
  • Watchlist news scan every 2 hours around the clock

Run modes (auto-detected from ET clock, or pass --mode <name>):
  market_scan      — 9:30am–4:00pm ET, every 15 min
  premarket_brief  — 8:45am ET snapshot
  postmarket_brief — 4:30pm ET recap
  news_check       — overnight / weekend, every 3 hours

Fallback: if ANTHROPIC_API_KEY is not set, the bot reverts to the
rule-based analyst note so Telegram alerts still fire.
"""

import os, sys, json, time, logging, re
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

ROOT           = Path(__file__).parent
WATCHLIST_FILE = ROOT / "watchlist.txt"
STATE_FILE     = ROOT / "data" / "state.json"
ET             = ZoneInfo("America/New_York")

# Multi-TF confluence threshold (out of max 7 points across 3 timeframes)
SIGNAL_THRESHOLD = 4

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot")

# ── Company name lookup ───────────────────────────────────────────────────
NAMES = {
    "AXTI":"AXT Inc",        "NVDA":"NVIDIA",
    "AMD":"AMD",              "MSFT":"Microsoft",
    "GOOGL":"Alphabet",       "AMZN":"Amazon",
    "META":"Meta",            "TSLA":"Tesla",
    "AAPL":"Apple",           "ORCL":"Oracle",
    "TSM":"TSMC",             "AVGO":"Broadcom",
    "ASML":"ASML",            "MU":"Micron",
    "ON":"ON Semi",           "ACMR":"ACM Research",
    "COHR":"Coherent",        "LITE":"Lumentum",
    "SIMO":"Silicon Motion",  "MRVL":"Marvell",
    "CRDO":"Credo Tech",      "CIEN":"Ciena",
    "PENG":"Penguin Solutions","IONQ":"IonQ",
    "RGTI":"Rigetti",         "QUBT":"Quantum Computing",
    "QBTS":"D-Wave",          "SMCI":"Super Micro",
    "APLD":"Applied Digital", "CRWV":"CoreWeave",
    "PLTR":"Palantir",        "SOUN":"SoundHound",
    "CBRS":"CBRS",            "SPCX":"SPCX",
    "SPY":"S&P 500 ETF",      "QQQ":"Nasdaq 100 ETF",
}

# ── Watchlist & state ─────────────────────────────────────────────────────
def load_watchlist():
    rows = []
    for line in WATCHLIST_FILE.read_text().splitlines():
        l = line.strip()
        if l and not l.startswith("#"):
            rows.append(l.upper())
    return rows

def load_state():
    if STATE_FILE.exists():
        try: return json.loads(STATE_FILE.read_text())
        except: return {}
    return {}

def save_state(s):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))

# ── Session & mode detection ──────────────────────────────────────────────
def get_session():
    from datetime import time as T
    t = datetime.now(ET).time()
    if T(4,0) <= t < T(9,30):  return "PRE_MARKET"
    if T(9,30) <= t < T(16,0): return "MARKET"
    if T(16,0) <= t < T(20,0): return "POST_MARKET"
    return "CLOSED"

def get_mode():
    from datetime import time as T
    t = datetime.now(ET).time()
    if T(8,30) <= t < T(9,30):  return "premarket_brief"
    if T(9,30) <= t < T(16,0):  return "market_scan"
    if T(16,0) <= t < T(17,30): return "postmarket_brief"
    return "news_check"

# ── Data fetching (multi-timeframe) ───────────────────────────────────────
def fetch_data(ticker):
    kw = dict(progress=False, auto_adjust=True)
    df5m = yf.download(ticker, period="5d",  interval="5m",  **kw)
    df1h = yf.download(ticker, period="60d", interval="1h",  **kw)
    for df in [df5m, df1h]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    if df5m.empty or df1h.empty:
        raise ValueError(f"No data for {ticker}")
    df4h = df1h.resample("4h", closed="left", label="left").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna()
    return {"5m": df5m, "1h": df1h, "4h": df4h}

# ── Indicators (one timeframe) ────────────────────────────────────────────
def compute_indicators(df):
    if df is None or len(df) < 20:
        return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    ind = {}

    # EMAs
    ind["ema9"]  = close.ewm(span=9,  adjust=False).mean().iloc[-1]
    ind["ema20"] = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ind["ema50"] = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(close)>=50 else float("nan")

    # VWAP (session reset by date)
    df2 = df.copy(); df2["_d"] = pd.to_datetime(df2.index).date
    tp = (high + low + close) / 3
    df2["_tv"] = tp * vol
    ind["vwap"] = (df2.groupby("_d")["_tv"].cumsum() /
                   df2.groupby("_d")["Volume"].cumsum()).iloc[-1]

    # RSI(14)
    d = close.diff()
    ag = d.clip(lower=0).rolling(14).mean()
    al = (-d.clip(upper=0)).rolling(14).mean()
    ind["rsi14"] = (100 - 100/(1 + ag/al.replace(0,np.nan))).iloc[-1]

    # MACD(12,26,9)
    ml = close.ewm(span=12,adjust=False).mean() - close.ewm(span=26,adjust=False).mean()
    sl2 = ml.ewm(span=9,adjust=False).mean()
    ind["macd_hist"]      = (ml - sl2).iloc[-1]
    ind["macd_hist_prev"] = (ml - sl2).iloc[-2] if len(close)>1 else ind["macd_hist"]

    # Bollinger Bands(20,2)
    bm = close.rolling(20).mean(); bs = close.rolling(20).std()
    ind["bb_upper"] = (bm + 2*bs).iloc[-1]
    ind["bb_mid"]   = bm.iloc[-1]
    ind["bb_lower"] = (bm - 2*bs).iloc[-1]

    # ATR(14)
    pc = close.shift(1)
    tr = pd.concat([high-low,(high-pc).abs(),(low-pc).abs()],axis=1).max(axis=1)
    ind["atr14"] = tr.rolling(14).mean().iloc[-1]

    # Volume
    ind["volume"]    = vol.iloc[-1]
    ind["vol_avg20"] = vol.rolling(20).mean().iloc[-1]
    ind["vol_ratio"] = ind["volume"]/ind["vol_avg20"] if ind["vol_avg20"] else 1

    # Price
    ind["close"]   = close.iloc[-1]
    ind["high20"]  = high.rolling(20).max().iloc[-1]
    ind["low20"]   = low.rolling(20).min().iloc[-1]
    ind["pct_chg"] = (close.iloc[-1]/close.iloc[-2]-1)*100 if len(close)>1 else 0

    # Trend label
    if ind["ema9"] > ind["ema20"]:
        ind["trend"] = "STRONG_BULL" if (not np.isnan(ind["ema50"]) and ind["ema20"]>ind["ema50"]) else "BULL"
    elif ind["ema9"] < ind["ema20"]:
        ind["trend"] = "STRONG_BEAR" if (not np.isnan(ind["ema50"]) and ind["ema20"]<ind["ema50"]) else "BEAR"
    else:
        ind["trend"] = "NEUTRAL"
    return ind

# ── Multi-timeframe scoring ───────────────────────────────────────────────
def score_mtf(i5m, i1h, i4h):
    """
    Score across 3 TFs. 4H and 1H each have weight 2, 5M has weight 1.
    Volume spike adds 1. Max = 7.
    BUY >= +4, SELL <= -4, WATCH = ±2..3, HOLD otherwise.
    """
    score = 0; bulls = []; bears = []

    def check(ind, label, w):
        nonlocal score
        if ind is None: return
        # Trend check
        if ind["ema9"] > ind["ema20"]:
            score += w; bulls.append(f"{label} EMA trend ▲")
        else:
            score -= w; bears.append(f"{label} EMA trend ▼")
        # MACD + RSI combo
        macd_up = ind["macd_hist"] > 0 and ind["macd_hist"] >= ind["macd_hist_prev"]
        rsi = ind["rsi14"]
        if macd_up and rsi < 70:
            score += w; bulls.append(f"{label} MACD+RSI({rsi:.0f}) bullish")
        elif not macd_up and rsi > 30:
            score -= w; bears.append(f"{label} MACD+RSI({rsi:.0f}) bearish")

    check(i4h, "4H", 2)
    check(i1h, "1H", 2)
    check(i5m, "5M", 1)

    # Volume confirmation
    if i5m and i5m["vol_ratio"] > 1.5:
        if score > 0: score += 1; bulls.append(f"Vol spike {i5m['vol_ratio']:.1f}x")
        elif score < 0: score -= 1; bears.append(f"Vol spike {i5m['vol_ratio']:.1f}x")

    if score >= SIGNAL_THRESHOLD:    return score, "BUY",  bulls, bears
    elif score <= -SIGNAL_THRESHOLD: return score, "SELL", bulls, bears
    elif abs(score) >= 2:            return score, "WATCH",bulls, bears
    else:                            return score, "HOLD", bulls, bears

# ── News fetching ─────────────────────────────────────────────────────────
def fetch_news(ticker, limit=6):
    items = []
    for url in [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en",
    ]:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:limit]:
                items.append({"title": e.title, "published": e.get("published","")})
            if len(items) >= limit: break
        except: continue
    return items[:limit]

def fetch_market_news():
    items = []
    for url in [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ&region=US&lang=en-US",
        "https://news.google.com/rss/search?q=stock+market+today+US&hl=en-US&gl=US&ceid=US:en",
    ]:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:5]:
                items.append({"title": e.title, "published": e.get("published","")})
        except: continue
    return items[:8]

# ── Claude brain (Anthropic API) ──────────────────────────────────────────
def claude_signal(ticker, company, i5m, i1h, i4h, news, score, session):
    """Call Claude for a structured trade analysis. Returns dict or None."""
    if not ANTHROPIC_API_KEY:
        return None

    def fmt(ind, label):
        if not ind: return f"{label}: no data"
        ab = "ABOVE" if ind["close"] > ind["vwap"] else "BELOW"
        return (
            f"{label} | Trend:{ind['trend']} | Close:${ind['close']:.2f}\n"
            f"  EMA9/20:${ind['ema9']:.2f}/${ind['ema20']:.2f} | "
            f"RSI:{ind['rsi14']:.0f} | MACDh:{ind['macd_hist']:.4f}\n"
            f"  VWAP:${ind['vwap']:.2f}({ab}) | ATR:${ind['atr14']:.2f} | "
            f"Vol:{ind['vol_ratio']:.1f}x | BB:{ind['bb_lower']:.2f}-{ind['bb_upper']:.2f}"
        )

    news_txt = "\n".join(f"- {n['title']}" for n in news[:5]) or "No recent news."
    now_et   = datetime.now(ET).strftime("%a %b %d %Y, %I:%M %p ET")

    prompt = f"""You are a professional quant trader with 20+ years experience. Singapore-based retail trader client.

DATETIME: {now_et} | SESSION: {session} | TICKER: {ticker} ({company})
CONFLUENCE SCORE: {score}/7 ({'BULLISH' if score>0 else 'BEARISH'})

MULTI-TIMEFRAME DATA:
{fmt(i4h,'4H (macro bias)')}
{fmt(i1h,'1H (trade direction)')}
{fmt(i5m,'5M (entry timing)')}

RECENT NEWS:
{news_txt}

TASK: Synthesise the multi-timeframe picture with news. Consider:
1. Do 4H, 1H, 5M all agree? Disagreement = reduce size.
2. Does news support or contradict the technicals?
3. What is the optimal entry, stop (ATR-based), and 3 profit targets?
4. Recommend best timeframe for THIS setup (scalp=5M, intraday=1H+5M, swing=4H+1H).
5. Risk management: max 2% account risk per trade. Scale position accordingly.
6. Note any economic indicators, earnings dates, or sector themes relevant now.

Return ONLY valid JSON (no markdown, no preamble):
{{
  "decision": "BUY or SELL or HOLD or WATCH",
  "entry_price": <number>,
  "tp1": <1.5x ATR from entry>,
  "tp2": <2.5x ATR or key resistance>,
  "tp3": <4x ATR or prior swing extreme>,
  "stop_loss": <1x ATR on wrong side of entry>,
  "best_timeframe": "e.g. 4H+1H or 1H+5M or 5M",
  "timeframe_note": "one sentence — why this TF for this setup",
  "confidence": <1-10>,
  "risk_level": "Low or Medium or High",
  "position_size": "Full or Half or Quarter — based on conviction",
  "analyst_note": "<MAX 280 chars: setup rationale, key catalyst, key risk>",
  "key_catalysts": ["one", "two"],
  "economic_context": "any relevant macro indicator or event in 1 sentence"
}}"""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json",
                     "x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version":"2023-06-01"},
            json={"model":"claude-sonnet-4-6","max_tokens":900,
                  "messages":[{"role":"user","content":prompt}]},
            timeout=35,
        )
        r.raise_for_status()
        raw = r.json()["content"][0]["text"].strip()
        raw = re.sub(r"```json?|```","", raw).strip()
        return json.loads(raw)
    except Exception as e:
        log.error(f"Claude signal API error ({ticker}): {e}")
        return None

def claude_brief(mode, ticker_news, mkt_news):
    """Call Claude for a market brief. Returns dict or None."""
    if not ANTHROPIC_API_KEY:
        return None
    label = {"premarket_brief":"PRE-MARKET","postmarket_brief":"POST-MARKET",
             "news_check":"24HR NEWS CHECK"}.get(mode,"MARKET BRIEF")
    now_et = datetime.now(ET).strftime("%a %b %d %Y, %I:%M %p ET")
    mkt = "\n".join(f"- {n['title']}" for n in mkt_news[:5])
    watch_txt = ""
    for tk, news in list(ticker_news.items())[:18]:
        hls = "\n".join(f"  · {n['title']}" for n in news[:2])
        if hls: watch_txt += f"\n{NAMES.get(tk,tk)} ({tk}):\n{hls}\n"

    prompt = f"""You are a professional market analyst. Provide a concise {label} brief.
DATETIME: {now_et}

BROAD MARKET NEWS:
{mkt}

WATCHLIST NEWS:
{watch_txt or 'No watchlist headlines found.'}

Return ONLY valid JSON:
{{
  "market_mood": "Bullish or Bearish or Mixed or Cautious",
  "mood_reason": "one sentence",
  "key_themes": ["theme1","theme2","theme3"],
  "top_opportunities": [
    {{"ticker":"X","company":"Y","signal":"Opportunity or Watch or Avoid","reason":"brief"}}
  ],
  "risk_factors": ["risk1","risk2"],
  "economic_watch": "Any key macro event or indicator today? 1 sentence.",
  "trader_action": "What should a Singapore trader do this session? 2 sentences max.",
  "sector_rotation": "Any notable sector strength/weakness? 1 sentence."
}}"""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json",
                     "x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version":"2023-06-01"},
            json={"model":"claude-sonnet-4-6","max_tokens":700,
                  "messages":[{"role":"user","content":prompt}]},
            timeout=35,
        )
        r.raise_for_status()
        raw = r.json()["content"][0]["text"].strip()
        raw = re.sub(r"```json?|```","", raw).strip()
        return json.loads(raw)
    except Exception as e:
        log.error(f"Claude brief API error: {e}")
        return None

# ── Fallback targets (no API key) ─────────────────────────────────────────
def fallback_analysis(decision, i5m, i1h, i4h, bulls, bears):
    if not i5m: return None
    atr = i5m["atr14"]; entry = i5m["close"]
    if decision == "BUY":
        sl = entry - 1.0*atr
        tp1 = entry + 1.5*atr
        tp2 = min(entry + 2.5*atr, i5m["high20"])
        tp3 = entry + 4.0*atr
        fired = bulls
    else:
        sl = entry + 1.0*atr
        tp1 = entry - 1.5*atr
        tp2 = max(entry - 2.5*atr, i5m["low20"])
        tp3 = entry - 4.0*atr
        fired = bears
    note = "Rule-based signal: " + " | ".join(fired[:3])
    return {"decision":decision,"entry_price":entry,"tp1":tp1,"tp2":tp2,"tp3":tp3,
            "stop_loss":sl,"best_timeframe":"1H+5M","timeframe_note":"Intraday setup",
            "confidence":5,"risk_level":"Medium","position_size":"Half",
            "analyst_note":note[:280],"key_catalysts":fired[:2],"economic_context":""}

# ── Telegram message — signal alert (iPhone-optimised) ────────────────────
def fmt_signal_msg(ticker, company, decision, analysis, i5m, i1h, i4h):
    entry = analysis.get("entry_price", i5m["close"] if i5m else 0)
    tp1   = analysis.get("tp1", 0)
    tp2   = analysis.get("tp2", 0)
    tp3   = analysis.get("tp3", 0)
    sl    = analysis.get("stop_loss", 0)
    conf  = int(analysis.get("confidence", 5))
    risk  = analysis.get("risk_level", "Medium")
    note  = analysis.get("analyst_note", "")
    tf    = analysis.get("best_timeframe", "1H+5M")
    tf_n  = analysis.get("timeframe_note", "")
    cats  = analysis.get("key_catalysts", [])
    size  = analysis.get("position_size", "")
    eco   = analysis.get("economic_context", "")

    def pct(t):
        if entry <= 0: return ""
        p = (t - entry)/entry*100
        return f"{'+'if p>=0 else ''}{p:.1f}%"

    risk_px   = abs(entry - sl)
    reward_px = abs(tp2 - entry)
    rr        = reward_px/risk_px if risk_px > 0 else 0

    conf_bar  = "█"*conf + "░"*(10-conf)
    risk_e    = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk,"🟡")
    dec_e     = {"BUY":"🟢","SELL":"🔴","WATCH":"🟡"}.get(decision,"⚪")
    dec_txt   = {"BUY":"BUY  ✅","SELL":"SELL 🚨","WATCH":"WATCH 👀"}.get(decision,decision)

    def tf_row(ind, label):
        if not ind: return f"{label}: —"
        t = ind.get("trend","")
        e = "▲" if "BULL" in t else "▼" if "BEAR" in t else "→"
        m = "MACD+" if ind["macd_hist"]>0 else "MACD-"
        return f"{label} {e}  RSI {ind['rsi14']:.0f}  {m}"

    sep  = "─"*28
    now  = datetime.now(ET).strftime("%d %b · %I:%M%p ET")

    lines = [
        f"{dec_e} *{company}*",
        f"`${ticker}`  ·  *{dec_txt}*",
        sep,
        f"⏱  *{tf}*",
        f"_{tf_n}_" if tf_n else None,
        f"💪  Conf: `{conf_bar}` {conf}/10",
        f"{risk_e}  Risk: {risk}  |  Size: {size}",
        sep,
        f"📥  Entry    *${entry:.2f}*",
        f"🎯  TP1      *${tp1:.2f}*   `{pct(tp1)}`",
        f"🎯  TP2      *${tp2:.2f}*   `{pct(tp2)}`",
        f"🎯  TP3      *${tp3:.2f}*   `{pct(tp3)}`",
        f"🛑  Stop     *${sl:.2f}*   `{pct(sl)}`",
        f"⚖️   R:R  1 : {rr:.1f}",
        sep,
        "📊  *SIGNALS*",
        tf_row(i4h, "4H"),
        tf_row(i1h, "1H"),
        tf_row(i5m, "5M"),
        sep,
        "🧠  *ANALYST NOTE*",
        note,
    ]
    if cats:
        lines += ["", "📌  *Key Catalysts*"]
        for c in cats[:2]: lines.append(f"  ·  {c}")
    if eco:
        lines += [f"📊  {eco}"]
    lines += [
        sep,
        f"🕐  {now}",
        "_Signal · Verify before acting_",
        "_Not financial advice_",
    ]
    return "\n".join(l for l in lines if l is not None)

# ── Telegram message — news brief ─────────────────────────────────────────
def fmt_brief_msg(brief, mode):
    if not brief:
        return "📰 Market brief unavailable (check API key)."
    label = {"premarket_brief":"PRE-MARKET","postmarket_brief":"POST-MARKET",
             "news_check":"24HR NEWS"}.get(mode,"BRIEF")
    mood  = brief.get("market_mood","")
    mood_e= {"Bullish":"🟢","Bearish":"🔴","Mixed":"🟡","Cautious":"🟠"}.get(mood,"⚪")
    sep   = "─"*28
    now   = datetime.now(ET).strftime("%d %b · %I:%M%p ET")
    lines = [
        f"📰  *{label} BRIEF*",
        f"🕐  {now}",
        sep,
        f"{mood_e}  *{mood}*",
        f"_{brief.get('mood_reason','')}_",
    ]
    themes = brief.get("key_themes",[])
    if themes:
        lines += ["","🔑  *Key Themes*"]
        for t in themes[:3]: lines.append(f"  ·  {t}")
    opps = brief.get("top_opportunities",[])
    if opps:
        lines += ["","👀  *Watchlist*"]
        sig_e = {"Opportunity":"🟢","Watch":"🟡","Avoid":"🔴"}
        for o in opps[:4]:
            e = sig_e.get(o.get("signal",""),"⚪")
            lines.append(f"{e}  {o.get('company','')}  `{o.get('ticker','')}`")
            lines.append(f"  _{o.get('reason','')}_")
    risks = brief.get("risk_factors",[])
    if risks:
        lines += ["","⚠️  *Risk Factors*"]
        for r in risks[:2]: lines.append(f"  ·  {r}")
    sector = brief.get("sector_rotation","")
    if sector: lines += ["",f"🔄  {sector}"]
    eco = brief.get("economic_watch","")
    if eco: lines += [f"📊  {eco}"]
    action = brief.get("trader_action","")
    if action: lines += ["","💼  *Trader Action*", action]
    lines += [sep, "_Agentic Signal Bot · Not financial advice_"]
    return "\n".join(l for l in lines if l is not None)

# ── Telegram send ─────────────────────────────────────────────────────────
def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram credentials not set.")
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
              "parse_mode":"Markdown","disable_web_page_preview":True},
        timeout=15,
    )
    if r.status_code != 200:
        log.error(f"Telegram error: {r.status_code} {r.text}")
    r.raise_for_status()

# ── Market scan (runs during market hours) ────────────────────────────────
def run_market_scan():
    tickers = load_watchlist(); state = load_state()
    session = get_session()
    log.info(f"Market scan: {len(tickers)} tickers | session: {session}")

    for ticker in tickers:
        try:
            company = NAMES.get(ticker, ticker)
            dfs = fetch_data(ticker)
            i5m = compute_indicators(dfs["5m"])
            i1h = compute_indicators(dfs["1h"])
            i4h = compute_indicators(dfs["4h"])
            if i5m is None: continue

            candle_key   = str(dfs["5m"].index[-1])
            prev         = state.get(ticker, {})
            if prev.get("last_candle") == candle_key:
                continue  # same candle, skip

            score, decision, bulls, bears = score_mtf(i5m, i1h, i4h)

            if decision in ("BUY","SELL") and decision != prev.get("last_decision"):
                news     = fetch_news(ticker)
                analysis = claude_signal(ticker, company, i5m, i1h, i4h,
                                         news, score, session)
                if analysis is None:
                    analysis = fallback_analysis(decision, i5m, i1h, i4h, bulls, bears)
                if analysis:
                    final = analysis.get("decision", decision)
                    msg   = fmt_signal_msg(ticker, company, final,
                                           analysis, i5m, i1h, i4h)
                    send_telegram(msg)
                    log.info(f"Alert: {ticker} {final} | score {score} | "
                             f"conf {analysis.get('confidence','?')}")

            state[ticker] = {
                "last_candle":   candle_key,
                "last_decision": decision,
                "last_score":    score,
                "last_price":    float(i5m["close"]),
                "updated_at":    datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            log.warning(f"Skip {ticker}: {e}")
        time.sleep(0.6)

    save_state(state)

# ── News brief (pre/post market and 24hr) ─────────────────────────────────
def run_news_brief(mode):
    tickers = load_watchlist()
    mkt_news = fetch_market_news()
    ticker_news = {}
    for tk in tickers[:22]:
        news = fetch_news(tk, limit=3)
        if news: ticker_news[tk] = news
        time.sleep(0.3)

    brief = claude_brief(mode, ticker_news, mkt_news)
    msg   = fmt_brief_msg(brief, mode)
    send_telegram(msg)
    log.info(f"Brief sent: {mode}")

# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Allow CLI override: python scanner.py --mode premarket_brief
    mode = get_mode()
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--mode" and i+1 < len(sys.argv[1:]):
            mode = sys.argv[i+2]
    log.info(f"Mode: {mode} | ET: {datetime.now(ET).strftime('%H:%M')} | session: {get_session()}")

    if mode == "market_scan":
        run_market_scan()
    elif mode in ("premarket_brief", "postmarket_brief", "news_check"):
        run_news_brief(mode)
    else:
        run_news_brief("news_check")
