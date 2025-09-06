#!/usr/bin/env python3
"""
Design API rate limiting middleware
Production-ready Python implementation
Generated Hour 2 - Project 5
Created: 2025-09-04T20:15:05.771789
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
================================

This implementation provides a production-ready Python middleware for API rate limiting. It includes classes for configuration management, rate limiter, and error handling.

**Installation**
---------------

To use this middleware, install the required packages using pip:
```bash
pip install python-memcached
```
**Implementation**
-----------------

### `config.py`

This file contains the configuration settings for the rate limiter.

```python
from typing import Dict

class RateLimitConfig:
    def __init__(self, max_requests: int, time_window: int) -> None:
        """
        Initialize the rate limit configuration.

        :param max_requests: Maximum number of requests allowed within the time window.
        :param time_window: Time window in seconds.
        """
        self.max_requests = max_requests
        self.time_window = time_window

    def to_dict(self) -> Dict:
        """
        Convert the configuration to a dictionary.

        :return: Configuration dictionary.
        """
        return {"max_requests": self.max_requests, "time_window": self.time_window}
```

### `rate_limiter.py`

This file contains the rate limiter implementation.

```python
import memcache
import logging
from typing import Dict, Any

from config import RateLimitConfig

class RateLimiter:
    def __init__(self, config: RateLimitConfig, memcached_host: str) -> None:
        """
        Initialize the rate limiter.

        :param config: Rate limit configuration.
        :param memcached_host: Memcached host.
        """
        self.config = config
        self.memcached = memcache.Client([memcached_host])
        self.logger = logging.getLogger(__name__)

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if the client is allowed to make a request.

        :param client_id: Client ID.
        :return: True if allowed, False otherwise.
        """
        key = f"{client_id}:{self.config.max_requests}"
        value = self.memcached.get(key)
        if value is not None:
            self.memcached.incr(key)
            return value == str(self.config.max_requests)
        else:
            self.memcached.set(key, 1, self.config.time_window)
            return True

    def get_config(self) -> Dict:
        """
        Get the rate limit configuration.

        :return: Configuration dictionary.
        """
        return self.config.to_dict()
```

### `middleware.py`

This file contains the API rate limiting middleware.

```python
import logging
from typing import Callable, Any
from rate_limiter import RateLimiter

class RateLimitMiddleware:
    def __init__(self, rate_limiter: RateLimiter) -> None:
        """
        Initialize the rate limiting middleware.

        :param rate_limiter: Rate limiter instance.
        """
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)

    def __call__(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Handle incoming requests.

        :param func: Request handler function.
        :param args: Function arguments.
        :param kwargs: Function keyword arguments.
        :return: Function return value.
        """
        client_id = kwargs.get("client_id")
        if client_id is None:
            self.logger.error("Client ID is not provided")
            return {"error": "Client ID is not provided"}, 400

        if not self.rate_limiter.is_allowed(client_id):
            self.logger.error(f"Rate limit exceeded for client {client_id}")
            return {"error": "Rate limit exceeded"}, 429

        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error handling request for client {client_id}: {str(e)}")
            raise
```

### `app.py`

This file contains the example API application.

```python
import logging
from fastapi import FastAPI
from middleware import RateLimitMiddleware
from rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO)

app = FastAPI()

config = RateLimitConfig(max_requests=10, time_window=60)
rate_limiter = RateLimiter(config, "localhost:11211")
middleware = RateLimitMiddleware(rate_limiter)

@app.get("/")
async def root(client_id: str):
    return {"message": "Hello World"}

@app.get("/protected")
async def protected(client_id: str):
    return {"message": "Protected resource"}

app.add_middleware(middleware.__class__)
```

**Usage**
---------

To use this middleware, create an instance of the `RateLimiter` class with the desired configuration and memcached host. Then, create an instance of the `RateLimitMiddleware` class with the `RateLimiter` instance. Finally, add the middleware to the FastAPI application using the `add_middleware` method.

Note that this implementation uses memcached as the storage backend for the rate limiter. You can replace it with other storage solutions, such as Redis or a database, if needed.

**Error Handling**
-----------------

The middleware handles the following error cases:

*   Missing client ID: Returns a 400 error with a message indicating that the client ID is not provided.
*   Rate limit exceeded: Returns a 429 error with a message indicating that the rate limit has been exceeded.
*   Internal server error: Logs the error and raises it to the caller.

if __name__ == "__main__":
    print(f"🚀 Design API rate limiting middleware")
    print(f"📊 Generated: 2025-09-04T20:15:05.771804")
    logger.info(f"Starting Design API rate limiting middleware...")
