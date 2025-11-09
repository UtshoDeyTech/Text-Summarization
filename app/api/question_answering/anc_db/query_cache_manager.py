"""
Query Cache Manager
Caches query results to reduce API calls and improve performance
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from app.service.log_client import logger

# Cache directory path
CACHE_DIR = Path(__file__).parent / "query_cache"
CACHE_FILE = CACHE_DIR / "query_results.json"

class QueryCacheManager:
    """Manages caching of query results to reduce API calls"""

    def __init__(self, cache_ttl_minutes: int = 60):
        """
        Initialize query cache manager

        Args:
            cache_ttl_minutes: Time-to-live for cache entries in minutes (default: 60)
        """
        self.cache_dir = CACHE_DIR
        self.cache_file = CACHE_FILE
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._ensure_cache_directory()

    def _ensure_cache_directory(self):
        """Ensure cache directory exists"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[QUERY CACHE] Cache directory ready: {self.cache_dir}")
        except Exception as e:
            logger.error(f"[QUERY CACHE] Failed to create cache directory: {e}")
            raise

    def _generate_cache_key(self, question: str, lead_type: str, agent_id: str, agency_id: str) -> str:
        """
        Generate unique cache key for a query

        Args:
            question: User question
            lead_type: Lead type
            agent_id: Agent ID
            agency_id: Agency ID

        Returns:
            str: MD5 hash as cache key
        """
        # Normalize question (lowercase, strip whitespace)
        normalized_question = question.lower().strip()

        # Create unique string combining all parameters
        cache_string = f"{normalized_question}|{lead_type}|{agent_id}|{agency_id}"

        # Generate MD5 hash
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()

        return cache_key

    def get_cached_result(self, question: str, lead_type: str, agent_id: str, agency_id: str) -> Optional[Tuple[str, List[str], bool]]:
        """
        Get cached result for a query

        Args:
            question: User question
            lead_type: Lead type
            agent_id: Agent ID
            agency_id: Agency ID

        Returns:
            Tuple of (answer, suggested_questions, found_data) if cached and not expired, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(question, lead_type, agent_id, agency_id)
            cache_data = self._load_cache()

            if cache_key in cache_data:
                entry = cache_data[cache_key]

                # Check if cache entry is expired
                cached_time = datetime.fromisoformat(entry["cached_at"])
                if datetime.now() - cached_time < self.cache_ttl:
                    logger.info(f"[QUERY CACHE] ✓ Cache hit for question: {question[:50]}...")
                    logger.info(f"[QUERY CACHE] Cache age: {(datetime.now() - cached_time).seconds}s")
                    # Handle backward compatibility - old cache entries might not have found_data
                    found_data = entry.get("found_data", True)
                    return (entry["answer"], entry["suggested_questions"], found_data)
                else:
                    logger.info(f"[QUERY CACHE] Cache expired for question: {question[:50]}...")
                    # Remove expired entry
                    del cache_data[cache_key]
                    self._save_cache(cache_data)

            logger.debug(f"[QUERY CACHE] Cache miss for question: {question[:50]}...")
            return None

        except Exception as e:
            logger.error(f"[QUERY CACHE] Error getting cached result: {e}")
            return None

    def cache_result(self, question: str, lead_type: str, agent_id: str, agency_id: str,
                    answer: str, suggested_questions: List[str], found_data: bool = True) -> bool:
        """
        Cache query result

        Args:
            question: User question
            lead_type: Lead type
            agent_id: Agent ID
            agency_id: Agency ID
            answer: Query answer
            suggested_questions: Suggested follow-up questions
            found_data: Whether relevant data was found (default: True for backward compatibility)

        Returns:
            bool: True if cached successfully, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(question, lead_type, agent_id, agency_id)
            cache_data = self._load_cache()

            # Add new entry
            cache_data[cache_key] = {
                "question": question,
                "lead_type": lead_type,
                "agent_id": agent_id,
                "agency_id": agency_id,
                "answer": answer,
                "suggested_questions": suggested_questions,
                "found_data": found_data,
                "cached_at": datetime.now().isoformat(),
                "cache_key": cache_key
            }

            # Clean up old entries (keep only last 1000)
            if len(cache_data) > 1000:
                logger.info(f"[QUERY CACHE] Cache size exceeded, cleaning up...")
                self._cleanup_old_entries(cache_data)

            # Save to file
            self._save_cache(cache_data)

            logger.info(f"[QUERY CACHE] ✓ Cached result for question: {question[:50]}... (found_data={found_data})")
            return True

        except Exception as e:
            logger.error(f"[QUERY CACHE] Failed to cache result: {e}")
            return False

    def _cleanup_old_entries(self, cache_data: Dict, max_entries: int = 1000):
        """
        Remove oldest cache entries to keep cache size manageable

        Args:
            cache_data: Current cache data
            max_entries: Maximum number of entries to keep
        """
        try:
            # Sort by cached_at time
            sorted_entries = sorted(
                cache_data.items(),
                key=lambda x: x[1].get("cached_at", ""),
                reverse=True
            )

            # Keep only the most recent entries
            cleaned_data = dict(sorted_entries[:max_entries])

            removed_count = len(cache_data) - len(cleaned_data)
            logger.info(f"[QUERY CACHE] Removed {removed_count} old entries")

            cache_data.clear()
            cache_data.update(cleaned_data)

        except Exception as e:
            logger.error(f"[QUERY CACHE] Error during cleanup: {e}")

    def clear_cache(self, lead_type: Optional[str] = None, agent_id: Optional[str] = None) -> bool:
        """
        Clear query cache

        Args:
            lead_type: Specific lead type to clear, or None to clear all
            agent_id: Specific agent to clear, or None to clear all

        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            if lead_type or agent_id:
                logger.info(f"[QUERY CACHE] Clearing cache for lead_type={lead_type}, agent_id={agent_id}")
                cache_data = self._load_cache()

                # Filter out matching entries
                filtered_data = {
                    key: value for key, value in cache_data.items()
                    if not (
                        (lead_type and value.get("lead_type") == lead_type) or
                        (agent_id and value.get("agent_id") == agent_id)
                    )
                }

                removed_count = len(cache_data) - len(filtered_data)
                self._save_cache(filtered_data)
                logger.info(f"[QUERY CACHE] ✓ Cleared {removed_count} cache entries")
            else:
                logger.info(f"[QUERY CACHE] Clearing all cache")
                self._save_cache({})
                logger.info(f"[QUERY CACHE] ✓ All cache cleared")

            return True

        except Exception as e:
            logger.error(f"[QUERY CACHE] Failed to clear cache: {e}")
            return False

    def get_cache_info(self) -> Dict:
        """
        Get information about the cache

        Returns:
            dict: Cache information including entry count and size
        """
        cache_data = self._load_cache()

        # Calculate cache statistics
        total_entries = len(cache_data)
        entries_by_lead_type = {}
        expired_count = 0

        for entry in cache_data.values():
            lead_type = entry.get("lead_type", "unknown")
            entries_by_lead_type[lead_type] = entries_by_lead_type.get(lead_type, 0) + 1

            # Check if expired
            cached_time = datetime.fromisoformat(entry.get("cached_at", datetime.now().isoformat()))
            if datetime.now() - cached_time >= self.cache_ttl:
                expired_count += 1

        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_count,
            "expired_entries": expired_count,
            "entries_by_lead_type": entries_by_lead_type,
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists(),
            "cache_ttl_minutes": self.cache_ttl.seconds // 60
        }

    def _load_cache(self) -> Dict:
        """
        Load cache from file

        Returns:
            dict: Cached data or empty dict if file doesn't exist
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.debug(f"[QUERY CACHE] Cache file does not exist, returning empty cache")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"[QUERY CACHE] Invalid JSON in cache file: {e}")
            return {}
        except Exception as e:
            logger.error(f"[QUERY CACHE] Failed to load cache: {e}")
            return {}

    def _save_cache(self, cache_data: Dict) -> bool:
        """
        Save cache to file

        Args:
            cache_data: Data to save

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"[QUERY CACHE] Failed to save cache: {e}")
            return False


# Global instance with 120-minute TTL (increased for better cache hit rate)
query_cache = QueryCacheManager(cache_ttl_minutes=120)
