"""
Schema Cache Manager
Manages caching of database table schemas to improve query performance
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from app.service.log_client import logger

# Cache directory path
CACHE_DIR = Path(__file__).parent / "schema_cache"
CACHE_FILE = CACHE_DIR / "table_schemas.json"

class SchemaCacheManager:
    """Manages schema caching for database tables"""

    def __init__(self):
        """Initialize schema cache manager"""
        self.cache_dir = CACHE_DIR
        self.cache_file = CACHE_FILE
        self._ensure_cache_directory()

    def _ensure_cache_directory(self):
        """Ensure cache directory exists"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[SCHEMA CACHE] Cache directory ready: {self.cache_dir}")
        except Exception as e:
            logger.error(f"[SCHEMA CACHE] Failed to create cache directory: {e}")
            raise

    def save_schema(self, table_name: str, schema_info: str, column_list: list = None) -> bool:
        """
        Save schema information for a table to cache

        Args:
            table_name: Name of the table
            schema_info: Raw schema information string
            column_list: List of column names (optional)

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            logger.info(f"[SCHEMA CACHE] Saving schema for table: {table_name}")

            # Load existing cache or create new one
            cache_data = self._load_cache()

            # Update cache with new schema
            cache_data[table_name] = {
                "schema_info": schema_info,
                "column_list": column_list or [],
                "last_updated": datetime.now().isoformat(),
                "column_count": len(column_list) if column_list else 0
            }

            # Save to file
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[SCHEMA CACHE] ✓ Schema saved for {table_name} ({len(column_list) if column_list else 0} columns)")
            return True

        except Exception as e:
            logger.error(f"[SCHEMA CACHE] Failed to save schema for {table_name}: {e}")
            return False

    def get_schema(self, table_name: str) -> Optional[Dict]:
        """
        Get cached schema information for a table

        Args:
            table_name: Name of the table

        Returns:
            dict: Schema information or None if not found
        """
        try:
            cache_data = self._load_cache()

            if table_name in cache_data:
                logger.info(f"[SCHEMA CACHE] ✓ Schema found in cache for {table_name}")
                return cache_data[table_name]
            else:
                logger.warning(f"[SCHEMA CACHE] Schema not found in cache for {table_name}")
                return None

        except Exception as e:
            logger.error(f"[SCHEMA CACHE] Failed to get schema for {table_name}: {e}")
            return None

    def get_formatted_schema(self, table_name: str) -> Optional[str]:
        """
        Get formatted schema summary from cache

        Args:
            table_name: Name of the table

        Returns:
            str: Formatted schema summary or None if not found
        """
        schema_data = self.get_schema(table_name)

        if not schema_data:
            return None

        column_list = schema_data.get("column_list", [])

        if column_list:
            # Format as readable list
            schema_summary = "Available columns:\n" + "\n".join([f"  - {col}" for col in column_list[:50]])
            if len(column_list) > 50:
                schema_summary += f"\n  ... and {len(column_list) - 50} more columns"
            return schema_summary
        else:
            # Fallback to raw schema info
            return schema_data.get("schema_info", "")[:1000]

    def save_all_schemas(self, schemas: Dict[str, Dict]) -> bool:
        """
        Save multiple table schemas at once

        Args:
            schemas: Dictionary mapping table names to schema data

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            logger.info(f"[SCHEMA CACHE] Saving schemas for {len(schemas)} tables")

            cache_data = {}
            for table_name, schema_data in schemas.items():
                cache_data[table_name] = {
                    "schema_info": schema_data.get("schema_info", ""),
                    "column_list": schema_data.get("column_list", []),
                    "last_updated": datetime.now().isoformat(),
                    "column_count": len(schema_data.get("column_list", []))
                }

            # Save to file
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[SCHEMA CACHE] ✓ All schemas saved successfully")
            return True

        except Exception as e:
            logger.error(f"[SCHEMA CACHE] Failed to save all schemas: {e}")
            return False

    def get_all_schemas(self) -> Dict:
        """
        Get all cached schemas

        Returns:
            dict: All cached schemas
        """
        return self._load_cache()

    def clear_cache(self, table_name: Optional[str] = None) -> bool:
        """
        Clear schema cache

        Args:
            table_name: Specific table to clear, or None to clear all

        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            if table_name:
                logger.info(f"[SCHEMA CACHE] Clearing cache for table: {table_name}")
                cache_data = self._load_cache()
                if table_name in cache_data:
                    del cache_data[table_name]
                    with open(self.cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"[SCHEMA CACHE] ✓ Cache cleared for {table_name}")
                else:
                    logger.warning(f"[SCHEMA CACHE] Table {table_name} not found in cache")
            else:
                logger.info(f"[SCHEMA CACHE] Clearing all cache")
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                logger.info(f"[SCHEMA CACHE] ✓ All cache cleared")

            return True

        except Exception as e:
            logger.error(f"[SCHEMA CACHE] Failed to clear cache: {e}")
            return False

    def is_cached(self, table_name: str) -> bool:
        """
        Check if schema is cached for a table

        Args:
            table_name: Name of the table

        Returns:
            bool: True if cached, False otherwise
        """
        cache_data = self._load_cache()
        return table_name in cache_data

    def get_cache_info(self) -> Dict:
        """
        Get information about the cache

        Returns:
            dict: Cache information including table count and last updated times
        """
        cache_data = self._load_cache()

        return {
            "total_tables": len(cache_data),
            "tables": {
                table_name: {
                    "column_count": data.get("column_count", 0),
                    "last_updated": data.get("last_updated", "Unknown")
                }
                for table_name, data in cache_data.items()
            },
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists()
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
                logger.debug(f"[SCHEMA CACHE] Cache file does not exist, returning empty cache")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"[SCHEMA CACHE] Invalid JSON in cache file: {e}")
            return {}
        except Exception as e:
            logger.error(f"[SCHEMA CACHE] Failed to load cache: {e}")
            return {}


# Global instance
schema_cache = SchemaCacheManager()
