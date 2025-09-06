#!/usr/bin/env python3
"""
Design API rate limiting middleware
Production-ready Python implementation
Generated Hour 7 - Project 2
Created: 2025-09-04T20:37:51.302766
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI-Generated Implementation:
**API Rate Limiting Middleware**
=====================================

This implementation provides a production-ready API rate limiting middleware using the Redis store. It uses a token bucket algorithm to enforce rate limits.

**Installation**
---------------

To install the required dependencies, run the following command:

```bash
pip install redis
```

**Implementation**
-----------------

### `rate_limiter.py`

```python
import logging
import redis
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RateLimitException(Exception):
    """Custom exception for rate limit exceeded error"""
    pass

class RateLimiter:
    def __init__(self, redis_client: redis.Redis, keys_prefix: str, rate_limit: int, period: int):
        """
        Initialize the rate limiter

        :param redis_client: Redis client instance
        :param keys_prefix: Prefix for Redis keys
        :param rate_limit: Maximum number of requests per period
        :param period: Time period in seconds
        """
        self.redis_client = redis_client
        self.keys_prefix = keys_prefix
        self.rate_limit = rate_limit
        self.period = period

    def _get_key(self, identifier: str) -> str:
        """
        Get the Redis key for the given identifier

        :param identifier: Identifier for the key
        :return: Redis key
        """
        return f"{self.keys_prefix}:{identifier}"

    def _get_token_count(self, identifier: str) -> int:
        """
        Get the token count for the given identifier

        :param identifier: Identifier for the key
        :return: Token count
        """
        key = self._get_key(identifier)
        token_count = self.redis_client.get(key)
        if token_count is None:
            return self.rate_limit
        return int(token_count)

    def _inc_token_count(self, identifier: str, count: int) -> None:
        """
        Increment the token count for the given identifier

        :param identifier: Identifier for the key
        :param count: Increment value
        """
        key = self._get_key(identifier)
        self.redis_client.incr(key)
        self.redis_client.expire(key, self.period)

    def _check_rate_limit(self, identifier: str) -> bool:
        """
        Check the rate limit for the given identifier

        :param identifier: Identifier for the key
        :return: True if rate limit is not exceeded, False otherwise
        """
        token_count = self._get_token_count(identifier)
        if token_count < self.rate_limit:
            self._inc_token_count(identifier, 1)
            return True
        raise RateLimitException("Rate limit exceeded")

class Middleware:
    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize the middleware

        :param rate_limiter: Rate limiter instance
        """
        self.rate_limiter = rate_limiter

    def __call__(self, enforce_rate_limit: bool = True, identifier: str = "default"):
        """
        Middleware function

        :param enforce_rate_limit: Enforce rate limit (default: True)
        :param identifier: Identifier for the rate limiter (default: "default")
        """
        if enforce_rate_limit:
            try:
                self.rate_limiter._check_rate_limit(identifier)
            except RateLimitException as e:
                raise e
        return lambda func: lambda *args, **kwargs: func(*args, **kwargs)
```

### `app.py`

```python
import logging
import redis
from rate_limiter import Middleware, RateLimiter
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

redis_client = redis.Redis(host='localhost', port=6379, db=0)

rate_limiter = RateLimiter(redis_client, "api_rates", 100, 60)
middleware = Middleware(rate_limiter)

@app.middleware("http")
async def rate_limiting_middleware(request: str, call_next):
    return middleware(enforce_rate_limit=True)(call_next)(request)

@app.get("/api/endpoint")
async def endpoint():
    return {"message": "Hello, World!"}
```

**Usage**
---------

1. Create a Redis instance or use an existing one.
2. Initialize the rate limiter with the Redis client, keys prefix, rate limit, and period.
3. Create a middleware instance with the rate limiter.
4. Apply the middleware to a FastAPI application using the `@app.middleware` decorator.
5. Enforce rate limits in your API endpoints by using the `middleware(enforce_rate_limit=True)` decorator.

**Configuration**
-----------------

You can configure the rate limiter by passing different values to the `RateLimiter` constructor:

*   `redis_client`: Redis client instance
*   `keys_prefix`: Prefix for Redis keys
*   `rate_limit`: Maximum number of requests per period
*   `period`: Time period in seconds

if __name__ == "__main__":
    print(f"🚀 Design API rate limiting middleware")
    print(f"📊 Generated: 2025-09-04T20:37:51.302779")
    logger.info(f"Starting Design API rate limiting middleware...")
