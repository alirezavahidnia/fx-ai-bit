# FX-AI-Bot

An automated **Forex trading bot** built with Python and MetaTrader 5.
Optimized for **low-equity accounts** and tuned for the **M5 timeframe**.

---

## ✨ Features

* **Technical analysis signals**:

  * RSI, MACD, realized volatility, ATR-based stop-loss & take-profit.
* **Hysteresis logic**:

  * Prevents rapid flip-flopping with entry/exit confidence thresholds.
* **Risk management**:

  * Risk-based lot sizing (scaled to equity and stop distance).
  * Daily drawdown kill-switch.
  * Daily trade cap per symbol.
* **Trading controls**:

  * Minimum hold and cooldown periods.
  * Trading window (restricts trading to UTC hours).
* **Telegram integration**:

  * Trade close alerts with P/L, balance, and details.
  * Daily summary at UTC rollover (trades, winrate, net P/L per symbol).
* **Live position sync**:

  * Detects existing MT5 positions on startup and resumes management.
* **Configurable**:

  * All parameters in `config.yaml`.
  * Secrets (Telegram token, chat ID) in `.env`.

---

## 📦 Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourname/fx-ai-bot.git
   cd fx-ai-bot
   ```

2. **Create a virtual environment & install dependencies**:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure MetaTrader 5 & risk settings**:

   * Edit `config.yaml` to set your trading symbols, timeframes, and risk controls.

4. **Add environment variables**:
   Create a `.env` file in the repo root:

   ```ini
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

5. **Run the bot**:

   ```bash
   python -m src.execution.mt5_dual_tf_runner
   ```

---

## ⚠️ Disclaimer

This bot is provided for **educational and research purposes only**.
Trading financial markets involves significant risk. Use at your own discretion.
