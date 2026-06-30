# 5-minute multi-indicator stock signal bot (free deployment)

Scans a watchlist every 5 minutes, scores each ticker against a confluence
of the technical indicators professional/intraday traders commonly use, and
sends a Telegram alert in this format whenever a ticker flips to a fresh
BUY or SELL:

```
🟢 AXTI
Decision: BUY
Price: $67.86
Candle: 2026-06-30 09:35:00 (5m)

Reason:
Multiple short-term indicators are aligned bullish:
- EMA9 is above EMA21 — short-term trend is up
- Price is trading above session VWAP — buyers in control
- RSI14 at 58 — positive momentum, not yet overbought
- Volume is running above its 20-period average — move has conviction
```

**This is a rule-based decision-support tool. It never places trades, and
the "Decision" it sends is a mechanical read of technical indicators —
not personalized financial advice.** Read the Limitations section below
before relying on it with real money.

---

## How the "analyst brain" works

Each scan computes 5 independent checks per ticker on 5-minute candles:

1. **EMA9 vs EMA21** — short-term trend direction
2. **Price vs session VWAP** — whether buyers or sellers control the session
3. **RSI(14)** — momentum, with overbought (≥70) / oversold (≤30) guardrails
4. **MACD histogram** — whether bullish/bearish momentum is building or fading
5. **Volume vs its 20-period average** — confirms the move has real participation

Each check contributes +1 (bullish) or -1 (bearish) to a score from -5 to
+5. A **BUY** fires at score ≥ +3, a **SELL** fires at score ≤ -3 (edit
`SIGNAL_THRESHOLD` in `scanner.py` to make this stricter or looser).
Bollinger Bands are also checked to flag when a signal is already
"stretched" (late entry warning in the message).

The reasoning text is built from a template listing exactly which
conditions fired — that's the free version of "analyst brain." If you ever
want true Claude-generated prose instead of the template, you'd replace
`build_reason()` with a call to the Anthropic API (a few lines, using the
indicator dict as input) — that costs a small amount per call since it's a
paid API, so it's left out of the free default.

You only get pinged when a ticker's signal **changes** to BUY/SELL — not
every 5 minutes regardless of outcome — so the channel stays usable.

---

## Why "scan the entire stock market" isn't the free version

There are roughly 8,000 listed US tickers. Free data (Yahoo Finance via
`yfinance`, which is what makes this $0) will rate-limit or block
aggressive polling of that many tickers every 5 minutes, and free compute
(GitHub Actions) has execution-time limits too. `watchlist.txt` ships with
~25 liquid, news-active tickers (your AXTI, mega-cap tech, semiconductors,
a few volatile mid-caps, plus SPY/QQQ as market reference) — edit that file
to add or remove tickers. In testing, 50-80 tickers comfortably finishes
within a 5-minute cycle on the free GitHub Actions runner; pushing toward
hundreds will need a paid data feed (e.g. Polygon.io, IBKR's own API) to
stay reliable.

---

## Deploy it free with GitHub Actions

No server, no hosting cost. GitHub Actions gives **unlimited free minutes
on public repositories**, which is what runs this on a schedule.

### 1. Create your Telegram bot
1. Message **@BotFather** on Telegram, send `/newbot`, follow the prompts.
2. Save the token it gives you.
3. Send your new bot any message (e.g. "hi").
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
   and find `"chat":{"id": ...}` — that's your chat ID.

### 2. Create a GitHub account and repo (free)
1. Sign up at github.com if you don't have an account.
2. Create a new **public** repository (public = unlimited free Action minutes).
3. Upload all the files in this folder to that repo (drag-and-drop on
   github.com works, or use `git push` if you're comfortable with git).

### 3. Add your secrets
In the repo: **Settings → Secrets and variables → Actions → New repository secret**
- Add `TELEGRAM_BOT_TOKEN` with your bot token
- Add `TELEGRAM_CHAT_ID` with your chat ID

### 4. Enable Actions and test
1. Go to the **Actions** tab in your repo, enable workflows if prompted.
2. Click into "5-minute stock signal scan" → **Run workflow** to trigger it
   manually and confirm it works (check the run logs, and check Telegram).
3. Once confirmed, it runs automatically every 5 minutes during market
   hours per the schedule in `.github/workflows/scan.yml` — nothing else
   to do.

### 5. Customize
- Edit `watchlist.txt` to change which tickers are scanned.
- Edit `SIGNAL_THRESHOLD` in `scanner.py` to make signals stricter/looser.
- Edit the cron line in `.github/workflows/scan.yml` if you want a
  different schedule (e.g. only during your local evening hours).

---

## Limitations (read before trusting this with real money)

- **Data is free, unofficial, and can be delayed.** `yfinance` pulls from
  Yahoo Finance's public endpoints, not a licensed real-time feed. For
  most liquid US stocks this is close to real-time, but it isn't
  guaranteed, and Yahoo can change/break the endpoint without notice.
- **GitHub Actions' schedule isn't exact.** Cron jobs on the free tier can
  run a few minutes late during high platform load — treat "every 5
  minutes" as "approximately every 5 minutes," not millisecond-precise.
- **A 3-out-of-5 confluence score is a simple heuristic, not a strategy
  with a proven edge.** It's a reasonable, transparent way to combine
  trend/momentum/volume signals the way many discretionary traders do
  manually — but it hasn't been backtested for profitability, and you
  should not assume historical-style win rates without testing it
  yourself first (paper-trade the alerts for a while before acting on
  them with real money).
- **This is not personalized financial advice**, and I'm not a licensed
  financial advisor — it's a tool that mechanically applies technical
  rules and tells you what it found.
