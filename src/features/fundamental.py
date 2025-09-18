import investpy
import pandas as pd
from loguru import logger
from datetime import datetime, timezone
from threading import Thread
from queue import Queue

def _fetch_investpy_in_thread(q: Queue, countries: list, importances: list, time_zone: str):
    """Helper function to run the blocking investpy call in a thread."""
    try:
        result = investpy.economic_calendar(
            countries=countries,
            importances=importances,
            time_zone=time_zone
        )
        q.put(result)
    except Exception as e:
        q.put(e)

def get_economic_calendar(countries: list = None, timeout_seconds: int = 30):
    """
    Fetches high-impact economic calendar events with a hard timeout
    to prevent the call from hanging indefinitely.
    """
    if countries is None:
        countries = [
            'united states', 'euro zone', 'japan', 'united kingdom', 'canada',
            'australia', 'new zealand', 'switzerland', 'china'
        ]

    try:
        q = Queue()
        thread = Thread(
            target=_fetch_investpy_in_thread,
            args=(q, countries, ['high'], 'GMT')
        )
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            logger.warning(f"investpy call timed out after {timeout_seconds} seconds.")
            return []

        result = q.get()
        if isinstance(result, Exception):
            raise result # Re-raise exception from the thread to be logged

        df = result
        if df.empty:
            return []

        df = df[['date', 'time', 'event', 'importance']].copy()
        today_str = datetime.now(timezone.utc).strftime('%d/%m/%Y')
        df = df[df['date'] == today_str]

        df = df[df['time'] != 'All Day']
        if df.empty:
            return []

        df['datetime_utc'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%d/%m/%Y %H:%M'
        ).dt.tz_localize('UTC')

        return df[['datetime_utc', 'event', 'importance']].to_dict('records')

    except Exception as e:
        logger.exception(f"Failed to fetch or process economic calendar: {e}")
        return []
