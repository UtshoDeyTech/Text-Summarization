"""
Web Query Cache Manager
Caches web search results to reduce API calls and improve performance
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path
from app.service.log_client import logger

# Cache directory path
CACHE_DIR = Path(__file__).parent / "web_query_cache"
CACHE_FILE = CACHE_DIR / "web_query_results.json"

class WebQueryCacheManager:
    """Manages caching of web search results to reduce API calls"""

    def __init__(self, cache_ttl_minutes: int = 120):
        """
        Initialize web query cache manager

        Args:
            cache_ttl_minutes: Time-to-live for cache entries in minutes (default: 120)
        """
        self.cache_dir = CACHE_DIR
        self.cache_file = CACHE_FILE
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._ensure_cache_directory()

    def _ensure_cache_directory(self):
        """Ensure cache directory exists"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[WEB QUERY CACHE] Cache directory ready: {self.cache_dir}")
        except Exception as e:
            logger.error(f"[WEB QUERY CACHE] Failed to create cache directory: {e}")
            raise

    def _generate_cache_key(self, question: str, user_id: Optional[str]) -> str:
        """
        Generate unique cache key for a web query

        Args:
            question: User question
            user_id: User ID

        Returns:
            str: MD5 hash as cache key
        """
        normalized_question = question.lower().strip()
        cache_string = f"{normalized_question}|{user_id or 'anonymous'}"
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
        return cache_key

    def get_cached_result(self, question: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a web query

        Args:
            question: User question
            user_id: User ID

        Returns:
            Cached response dictionary if not expired, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(question, user_id)
            cache_data = self._load_cache()

            if cache_key in cache_data:
                entry = cache_data[cache_key]
                cached_time = datetime.fromisoformat(entry["cached_at"])
                if datetime.now() - cached_time < self.cache_ttl:
                    logger.info(f"[WEB QUERY CACHE] ✓ Cache hit for question: {question[:50]}...")
                    return entry["response"]
                else:
                    logger.info(f"[WEB QUERY CACHE] Cache expired for question: {question[:50]}...")
                    del cache_data[cache_key]
                    self._save_cache(cache_data)

            logger.debug(f"[WEB QUERY CACHE] Cache miss for question: {question[:50]}...")
            return None

        except Exception as e:
            logger.error(f"[WEB QUERY CACHE] Error getting cached result: {e}")
            return None

    def cache_result(self, question: str, user_id: Optional[str], response: Dict[str, Any]) -> bool:
        """
        Cache a web query result

        Args:
            question: User question
            user_id: User ID
            response: The WebSearchResponse as a dictionary

        Returns:
            bool: True if cached successfully, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(question, user_id)
            cache_data = self._load_cache()

            cache_data[cache_key] = {
                "question": question,
                "user_id": user_id,
                "response": response,
                "cached_at": datetime.now().isoformat(),
            }

            if len(cache_data) > 500:  # Limit cache size
                self._cleanup_old_entries(cache_data)

            self._save_cache(cache_data)
            logger.info(f"[WEB QUERY CACHE] ✓ Cached result for question: {question[:50]}...")
            return True

        except Exception as e:
            logger.error(f"[WEB QUERY CACHE] Failed to cache result: {e}")
            return False

    def _cleanup_old_entries(self, cache_data: Dict, max_entries: int = 500):
        """Remove oldest cache entries"""
        try:
            sorted_entries = sorted(cache_data.items(), key=lambda x: x[1].get("cached_at", ""), reverse=True)
            cleaned_data = dict(sorted_entries[:max_entries])
            removed_count = len(cache_data) - len(cleaned_data)
            logger.info(f"[WEB QUERY CACHE] Removed {removed_count} old entries")
            cache_data.clear()
            cache_data.update(cleaned_data)
        except Exception as e:
            logger.error(f"[WEB QUERY CACHE] Error during cleanup: {e}")

    def clear_cache(self) -> bool:
        """Clear the entire web query cache"""
        try:
            logger.info(f"[WEB QUERY CACHE] Clearing all cache")
            self._save_cache({})
            return True
        except Exception as e:
            logger.error(f"[WEB QUERY CACHE] Failed to clear cache: {e}")
            return False

    def _load_cache(self) -> Dict:
        """Load cache from file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[WEB QUERY CACHE] Failed to load cache: {e}")
            return {}

    def _save_cache(self, cache_data: Dict) -> bool:
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"[WEB QUERY CACHE] Failed to save cache: {e}")
            return False

# Global instance
web_query_cache = WebQueryCacheManager()
