import investpy
import pandas as pd
from loguru import logger
from datetime import datetime, timezone

def get_economic_calendar(countries: list = None):
    """
    Fetches high-impact economic calendar events for the current day from major economies.
    Uses GMT/UTC for time.
    """
    if countries is None:
        countries = [
            'united states', 'euro zone', 'japan', 'united kingdom', 'canada',
            'australia', 'new zealand', 'switzerland', 'china'
        ]

    try:
        # Fetch high-impact events for today
        df = investpy.economic_calendar(
            countries=countries,
            importances=['high'],
            time_zone='GMT'
        )
        if df.empty:
            return []

        # We only need a few columns
        df = df[['date', 'time', 'event', 'impact']].copy()

        # The date from investpy is 'dd/mm/yyyy'. Filter for today UTC.
        today_str = datetime.now(timezone.utc).strftime('%d/%m/%Y')
        df = df[df['date'] == today_str]

        # Ignore 'All Day' events as they don't have a specific time
        df = df[df['time'] != 'All Day']
        if df.empty:
            return []

        # Create a proper UTC datetime object for easier comparison
        df['datetime_utc'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%d/%m/%Y %H:%M'
        ).dt.tz_localize('UTC')

        # Return relevant columns as a list of dictionaries
        return df[['datetime_utc', 'event', 'impact']].to_dict('records')

    except Exception as e:
        logger.exception(f"Failed to fetch or process economic calendar: {e}")
        # investpy is known to have connection issues sometimes
        if "Max retries exceeded" in str(e) or "ECONNRESET" in str(e):
            logger.error("investpy failed with a connection error, which is a known issue. Skipping this cycle.")
        return []
