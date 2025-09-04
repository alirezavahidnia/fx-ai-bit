from __future__ import annotations
import pandas as pd
import investpy
from loguru import logger
from typing import List, Dict

def get_economic_calendar(currencies: List[str], importances: List[str]) -> pd.DataFrame:
    """
    Fetches economic calendar data for specified currencies and importance levels.

    Args:
        currencies: A list of currency codes (e.g., ["USD", "EUR", "GBP"]).
        importances: A list of importance levels (e.g., ["high", "medium"]).

    Returns:
        A pandas DataFrame with the economic calendar data, or an empty DataFrame on error.
    """
    try:
        # The investpy library expects full country names, not currency codes.
        # We need to map currencies to the countries investpy expects.
        # This is a simplified mapping.
        currency_to_country_map = {
            "USD": "united states",
            "EUR": "euro zone",
            "GBP": "united kingdom",
            "JPY": "japan",
            "CAD": "canada",
            "AUD": "australia",
            "NZD": "new zealand",
            "CHF": "switzerland",
            "CNY": "china",
        }

        countries = [currency_to_country_map.get(c.upper()) for c in currencies]
        countries = [c for c in countries if c is not None]

        if not countries:
            logger.warning("No valid countries found for the given currencies.")
            return pd.DataFrame()

        logger.info(f"Fetching economic calendar for countries: {countries} with importances: {importances}")

        # Fetch data for today
        calendar_df = investpy.economic_calendar(
            countries=countries,
            importances=importances
        )

        if calendar_df.empty:
            logger.info("No economic events found for the specified criteria today.")
            return pd.DataFrame()

        # The 'time' column is just a string like '02:00', '14:30' or 'All Day'.
        # We need to parse it and combine with the date for proper timestamping.
        # For now, we will just return the dataframe as is.
        # Further processing will be needed in the main bot.

        logger.info(f"Successfully fetched {len(calendar_df)} economic events.")
        return calendar_df

    except Exception as e:
        # The investpy library sometimes fails if there's no data or if the website structure changes.
        # It can raise a variety of exceptions. We'll catch them and log the error.
        logger.error(f"An error occurred while fetching the economic calendar: {e}")
        # A common error is "Connection Error: 'economic_calendar' object has no attribute 'find_all'".
        # This usually means the scrape failed.
        if "economic_calendar' object has no attribute 'find_all'" in str(e):
             logger.warning("Investpy might be having trouble scraping Investing.com. This can be intermittent.")
        return pd.DataFrame()
