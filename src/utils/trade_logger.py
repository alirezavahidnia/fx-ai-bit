import csv
import os
from typing import Dict, Any
from loguru import logger
import threading
import datetime as dt

class TradeLogger:
    def __init__(self, filepath: str = 'trade_log.csv'):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.header = [
            'timestamp_utc', 'event_type', 'position_id', 'symbol',
            'direction', 'volume', 'price', 'sl', 'tp', 'pnl',
            'technical_p_up', 'final_p_up',
            'rsi14', 'macd_hist', 'rv96', 'ema_dist_norm',
            'base_sentiment', 'quote_sentiment', 'net_sentiment'
        ]
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.filepath):
            with self.lock:
                if not os.path.exists(self.filepath):
                    with open(self.filepath, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(self.header)
                        logger.info(f"Created trade log file at: {self.filepath}")

    def log_event(self, event_data: Dict[str, Any]):
        """Logs a generic event to the CSV file."""
        with self.lock:
            try:
                # Ensure all header columns are present in the data, with None as default
                row = [event_data.get(h) for h in self.header]

                with open(self.filepath, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                logger.exception(f"Failed to write to trade log: {e}")

    def log_open_trade(self, position_id: int, symbol: str, direction: int, volume: float, price: float, sl: float, tp: float, features: dict):
        """Logs the opening of a trade."""
        event_data = {
            'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'event_type': 'open',
            'position_id': position_id,
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'price': price,
            'sl': sl,
            'tp': tp,
            **features # Add all the feature data
        }
        self.log_event(event_data)

    def log_close_trade(self, position_id: int, pnl: float):
        """Logs the closing of a trade."""
        event_data = {
            'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'event_type': 'close',
            'position_id': position_id,
            'pnl': pnl,
        }
        self.log_event(event_data)
