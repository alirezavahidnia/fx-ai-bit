import pandas as pd
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
        with self.lock:
            if not os.path.exists(self.filepath):
                pd.DataFrame(columns=self.header).to_csv(self.filepath, index=False)
                logger.info(f"Created trade log file at: {self.filepath}")

    def log_open_trade(self, position_id: int, symbol: str, direction: int, volume: float, price: float, sl: float, tp: float, features: dict):
        """Logs the opening of a trade to a new row."""
        with self.lock:
            try:
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
                    **features
                }

                df = pd.DataFrame([event_data], columns=self.header)
                df.to_csv(self.filepath, mode='a', header=False, index=False)

            except Exception as e:
                logger.exception(f"Failed to log open trade: {e}")

    def log_close_trade(self, position_id: int, pnl: float):
        """
        Finds the trade opening row by position_id, updates it with the PNL,
        and marks it as closed using a safe, atomic write.
        """
        with self.lock:
            # Prevent trying to log a close if the file doesn't exist or is empty.
            if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
                logger.warning("Trade log file is empty or does not exist. Cannot log close.")
                return

            temp_filepath = self.filepath + '.tmp'
            try:
                # Use dtype to ensure correct matching and handle potential NaNs in position_id column
                df = pd.read_csv(self.filepath, dtype={'position_id': 'Int64'})

                mask = (df['position_id'] == position_id) & (df['event_type'] == 'open')

                if not mask.any():
                    logger.warning(f"Could not find opening log for position {position_id} to record PNL.")
                    return

                # In case of duplicates (e.g., bot restart), update the last open entry
                idx = df.index[mask].tolist()[-1]
                df.loc[idx, 'pnl'] = pnl
                df.loc[idx, 'event_type'] = 'closed'

                # Atomically write the updated dataframe to the file
                df.to_csv(temp_filepath, index=False)
                os.replace(temp_filepath, self.filepath)

                logger.info(f"Updated log for closed position {position_id} with PNL {pnl:.2f}")

            except Exception as e:
                logger.exception(f"Failed to log close trade for position {position_id}: {e}")
                # Clean up temp file on error to prevent inconsistent state
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except OSError as err:
                        logger.error(f"Error removing temp log file {temp_filepath}: {err}")
