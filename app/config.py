"""
Configuration settings for the API
"""
import os

# Server configuration
MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "1000"))
KEEP_ALIVE_TIMEOUT = int(os.getenv("KEEP_ALIVE", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Cache configuration
CACHE_MAX_SIZE = int(os.getenv("CACHE_SIZE", "1000"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

# Performance settings
ENABLE_CACHING = os.getenv("ENABLE_CACHE", "true").lower() == "true"
LOG_SLOW_REQUESTS = os.getenv("LOG_SLOW", "true").lower() == "true"
SLOW_REQUEST_THRESHOLD = float(os.getenv("SLOW_THRESHOLD", "1.0"))  # seconds

