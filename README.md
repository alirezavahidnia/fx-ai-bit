# FX-AI-Bot

An AI-powered trading bot for MetaTrader 5 (MT5).
It uses technical indicators + probabilistic signals on short timeframes (M5) to trade FX pairs automatically, with built-in risk management and Telegram alerts.

------------------------------------------------------------

FEATURES
- Connects directly to MetaTrader 5 via Python (MetaTrader5 library)
- Works on M5 timeframe (configurable)
- Technical indicators:
  * RSI (14)
  * MACD
  * ATR
  * Realized Volatility
- Probabilistic signal model with hysteresis (enter / exit / flip)
- Risk-based lot sizing (auto-calculated from account equity)
- Daily kill-switch (max drawdown % and max trades per symbol)
- SL/TP auto-calculated from ATR
- Telegram alerts whenever a trade is closed:
  * Symbol, direction, volume
  * Entry → Exit price
  * Profit/Loss and account balance
  * Reason for closing (flat or flip)

------------------------------------------------------------

PROJECT STRUCTURE
fx-ai-bot/
├── src/
│   ├── execution/
│   │   └── mt5_dual_tf_runner.py   # main runner script
│   ├── features/                   # technical indicators
│   └── utils/                      # config helpers
├── config.example.yaml              # safe config template
├── .env.example                     # template for Telegram secrets
├── requirements.txt                 # Python dependencies
└── README.md

------------------------------------------------------------

INSTALLATION
1. Requirements
- Windows with MetaTrader 5 installed (and logged into your broker account, e.g. RoboForex)
- Python 3.11+
- Git

2. Clone the repo
    git clone https://github.com/<your-username>/fx-ai-bot.git
    cd fx-ai-bot

3. Virtual environment
    py -3.11 -m venv .venv
    .\.venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt

4. Configuration
- Copy the examples:
    cp .env.example .env
    cp config.example.yaml config.yaml

- Edit .env → add your Telegram bot token & chat ID:
    TELEGRAM_BOT_TOKEN=your-token-here
    TELEGRAM_CHAT_ID=your-chat-id-here

- Edit config.yaml → set your pairs, risk settings, trading window, etc.

------------------------------------------------------------

RUNNING THE BOT
Open MetaTrader 5 and log in to your account. Then run:

    .\.venv\Scripts\python -m src.execution.mt5_dual_tf_runner

The bot will:
- Fetch bars from MT5
- Evaluate signals
- Place/cancel trades
- Send Telegram messages whenever a trade closes

------------------------------------------------------------

VPS SETUP
For 24/7 operation, you can host the bot on an AWS EC2 Windows Server:

- Recommended: t3a.medium (2 vCPU, 4 GB RAM, 40 GB disk)
- Install MT5 + Python + this bot
- Configure MT5 to auto-login
- Use Task Scheduler or NSSM to auto-run the bot on startup

------------------------------------------------------------

DISCLAIMER
This bot is provided for educational purposes only.
Trading Forex involves substantial risk. Use at your own risk.
The authors are not responsible for any financial losses.

------------------------------------------------------------

CONTACT
- Telegram alerts are built in — configure via .env
- Open issues or contribute via GitHub Pull Requests
