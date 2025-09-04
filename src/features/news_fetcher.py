from __future__ import annotations
from typing import List, Dict, Any
from newsapi import NewsApiClient
from loguru import logger

class NewsFetcher:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("NewsAPI key is required.")
        self.client = NewsApiClient(api_key=api_key)

    def fetch_forex_news(self, keywords: List[str], sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches news articles based on keywords and optional sources.
        """
        try:
            query = " OR ".join(keywords)
            response = self.client.get_everything(
                q=query,
                sources=",".join(sources) if sources else None,
                language="en",
                sort_by="publishedAt",
                page_size=50  # Get a decent number of recent articles
            )

            if response.get("status") == "ok":
                articles = []
                for article in response.get("articles", []):
                    articles.append({
                        "source": article.get("source", {}).get("name"),
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "url": article.get("url"),
                        "publishedAt": article.get("publishedAt"),
                    })
                logger.info(f"Fetched {len(articles)} articles for keywords: {keywords}")
                return articles
            else:
                logger.error(f"Error fetching news from NewsAPI: {response.get('message')}")
                return []

        except Exception as e:
            logger.exception(f"An error occurred while fetching news: {e}")
            return []
