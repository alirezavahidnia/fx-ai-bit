import requests
from loguru import logger
from collections import defaultdict

def fetch_news_sentiments(api_key: str, symbols: list[str]) -> dict:
    """
    Fetches news and sentiment scores from Alpha Vantage for a list of symbols.
    It calculates a relevance-weighted average sentiment score per symbol.

    API docs: https://www.alphavantage.co/documentation/#news-sentiment
    """
    if not api_key:
        logger.warning("Alpha Vantage API key not provided. Skipping news fetch.")
        return {}

    # For Forex pairs, we can use the FOREX topic and the symbols as tickers.
    # For Gold (XAUUSD), news is more related to broader economic topics.
    forex_symbols = [s for s in symbols if "USD" in s and s != "XAUUSD"] # A simple filter for forex pairs

    # Broad topics relevant to all assets, especially commodities
    topics = "financial_markets,economy_macro,finance"

    # Build the tickers string, including a general FOREX topic ticker
    tickers_list = ["FOREX"] + forex_symbols
    tickers_str = ",".join(tickers_list)

    url = (
        f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
        f'&tickers={tickers_str}'
        f'&topics={topics}'
        f'&apikey={api_key}'
        f'&limit=50'  # Fetch last 50 relevant articles
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        if "feed" not in data or not data["feed"]:
            logger.info(f"No news found for tickers: {tickers_str}")
            return {}

        # Process the feed to get an average sentiment per ticker
        # We will store a tuple of (weighted_score, relevance_sum)
        sentiment_data = defaultdict(lambda: [0.0, 0.0])

        for article in data.get('feed', []):
            for ticker_sentiment in article.get('ticker_sentiment', []):
                symbol = ticker_sentiment.get('ticker')
                # The API uses FOREX:EURUSD format, we just want EURUSD
                symbol_clean = symbol.split(':')[-1]

                if symbol_clean in symbols:
                    try:
                        relevance = float(ticker_sentiment.get('relevance_score', 0.0))
                        sentiment = float(ticker_sentiment.get('ticker_sentiment_score', 0.0))

                        # Accumulate weighted score and total relevance
                        sentiment_data[symbol_clean][0] += sentiment * relevance
                        sentiment_data[symbol_clean][1] += relevance
                    except (ValueError, TypeError):
                        continue

        # Calculate the final average sentiment for each symbol
        avg_sentiments = {}
        for symbol, (weighted_score_sum, relevance_sum) in sentiment_data.items():
            if relevance_sum > 0:
                avg_score = weighted_score_sum / relevance_sum

                # Determine a qualitative label
                if avg_score >= 0.15:
                    label = 'BULLISH'
                elif avg_score <= -0.15:
                    label = 'BEARISH'
                else:
                    label = 'NEUTRAL'

                avg_sentiments[symbol] = {
                    "sentiment_score": round(avg_score, 4),
                    "sentiment_label": label,
                    "relevance_sum": round(relevance_sum, 4)
                }

        if avg_sentiments:
            logger.info(f"Fetched and processed sentiments from Alpha Vantage: {avg_sentiments}")
        return avg_sentiments

    except requests.exceptions.RequestException as e:
        logger.exception(f"Failed to fetch news from Alpha Vantage: {e}")
    except Exception as e:
        logger.exception(f"Error processing Alpha Vantage data: {e}")

    return {}
