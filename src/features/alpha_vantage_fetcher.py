from __future__ import annotations
from typing import List, Dict, Any
import datetime as dt
from loguru import logger
import requests

class AlphaVantageFetcher:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Alpha Vantage API key is required.")
        self.api_key = api_key

    def fetch_forex_news(self, topics: List[str] = None, tickers: List[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches news and sentiment from Alpha Vantage.

        Args:
            topics: A list of topics to filter by (e.g., ['economy_monetary', 'forex']).
            tickers: A list of tickers to filter by (e.g., ['FOREX:USD', 'FOREX:EUR']).

        Returns:
            A list of standardized news article dictionaries.
        """
        try:
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": 100, # Fetch a good number of articles
            }
            if topics:
                params["topics"] = ",".join(topics)
            if tickers:
                params["tickers"] = ",".join(tickers)

            logger.info(f"Fetching news from Alpha Vantage with params: {params}")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "feed" not in data:
                logger.warning(f"No 'feed' key in Alpha Vantage response. Response: {data}")
                return []

            articles = []
            for article in data.get("feed", []):
                main_sentiment_score = 0.0
                mentioned_currencies = []

                if article.get("ticker_sentiment"):
                    sorted_tickers = sorted(article["ticker_sentiment"], key=lambda x: float(x.get('relevance_score', 0)), reverse=True)
                    if sorted_tickers:
                        main_sentiment_score = float(sorted_tickers[0].get('ticker_sentiment_score', 0.0))

                    for sent in article["ticker_sentiment"]:
                        ticker = sent.get('ticker', '')
                        if 'FOREX' in ticker:
                            currency = ticker.split(':')[1]
                            if currency not in mentioned_currencies:
                                mentioned_currencies.append(currency)

                articles.append({
                    "source": article.get("source"),
                    "title": article.get("title"),
                    "description": article.get("summary"),
                    "url": article.get("url"),
                    "publishedAt": article.get("time_published"),
                    "sentiment": main_sentiment_score,
                    "currencies": mentioned_currencies,
                    "timestamp": self._parse_timestamp(article.get("time_published")),
                })

            logger.info(f"Fetched and processed {len(articles)} articles from Alpha Vantage.")
            return articles

        except requests.exceptions.RequestException as e:
            logger.exception(f"An error occurred while fetching news from Alpha Vantage: {e}")
            return []
        except (ValueError, KeyError, TypeError) as e:
            logger.exception(f"Error processing data from Alpha Vantage: {e}")
            return []

    def _parse_timestamp(self, ts_str: str) -> dt.datetime:
        # Alpha Vantage timestamp is like "20220410T013000"
        return dt.datetime.strptime(ts_str, '%Y%m%dT%H%M%S').replace(tzinfo=dt.timezone.utc)
