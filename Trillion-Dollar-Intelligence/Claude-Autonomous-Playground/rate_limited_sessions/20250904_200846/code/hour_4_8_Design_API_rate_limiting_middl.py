#!/usr/bin/env python3
"""
Design API rate limiting middleware
Production-ready Python implementation
Generated Hour 4 - Project 8
Created: 2025-09-04T20:24:48.019075
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
Here's a Python implementation of API rate limiting middleware. This example uses Redis as the backend to store and retrieve rate limit information.

**rate_limiter.py**
```python
import logging
import time
from typing import Dict, Union
from redis import Redis

logger = logging.getLogger(__name__)

class RateLimitException(Exception):
    """Custom exception for rate limiting errors"""
    pass

class RateLimiter:
    """Rate Limiter class"""
    def __init__(self, redis: Redis, namespace: str, max_requests: int, period: int):
        """
        Initialize the RateLimiter class

        Args:
            redis (Redis): Redis instance for storage
            namespace (str): Namespace for rate limit information
            max_requests (int): Maximum number of requests allowed within the period
            period (int): Time period in seconds
        """
        self.redis = redis
        self.namespace = namespace
        self.max_requests = max_requests
        self.period = period

    def is_rate_limited(self, client_ip: str, resource_path: str) -> bool:
        """
        Check if the client is rate limited

        Args:
            client_ip (str): Client IP address
            resource_path (str): Resource path

        Returns:
            bool: True if rate limited, False otherwise
        """
        key = f"{self.namespace}:{client_ip}:{resource_path}"
        value = self.redis.get(key)

        if value:
            timestamp, count = map(int, value.decode().split(":"))
            if time.time() - timestamp < self.period:
                if count >= self.max_requests:
                    return True
                self.redis.incr(key, 1)
            else:
                self.redis.set(key, f"{int(time.time())}:{1}")
        else:
            self.redis.setex(key, self.period, f"{int(time.time())}:{1}")

        return False

class RateLimitMiddleware:
    """Rate Limit Middleware class"""
    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize the RateLimitMiddleware class

        Args:
            rate_limiter (RateLimiter): RateLimiter instance
        """
        self.rate_limiter = rate_limiter

    def __call__(self, func):
        """
        Call the decorated function

        Args:
            func (function): Decorated function

        Returns:
            function: Wrapped function
        """
        def wrapper(*args, **kwargs):
            client_ip = kwargs.get("client_ip")
            resource_path = kwargs.get("resource_path")

            if self.rate_limiter.is_rate_limited(client_ip, resource_path):
                raise RateLimitException("Rate limit exceeded")

            return func(*args, **kwargs)

        return wrapper
```

**config.py**
```python
class Config:
    """Configuration class"""
    def __init__(self):
        """
        Initialize the Config class
        """
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.redis_password = None
        self.namespace = "rate_limit"
        self.max_requests = 100
        self.period = 60  # seconds

    def get_redis_url(self) -> str:
        """
        Get the Redis URL

        Returns:
            str: Redis URL
        """
        if self.redis_password:
            password = f":{self.redis_password}"
        else:
            password = ""

        return f"redis://{self.redis_host}:{self.redis_port}{password}"

    def get_redis_kwargs(self) -> Dict[str, Union[str, int]]:
        """
        Get the Redis connection kwargs

        Returns:
            Dict[str, Union[str, int]]: Redis connection kwargs
        """
        return {"host": self.redis_host, "port": self.redis_port, "password": self.redis_password}
```

**main.py**
```python
import logging
from rate_limiter import RateLimiter, RateLimitException
from rate_limiter import RateLimitMiddleware
from config import Config
from redis import Redis

logging.basicConfig(level=logging.INFO)

def main():
    config = Config()
    redis = Redis(**config.get_redis_kwargs())

    rate_limiter = RateLimiter(redis, config.namespace, config.max_requests, config.period)
    middleware = RateLimitMiddleware(rate_limiter)

    @middleware
    def hello_world(client_ip: str, resource_path: str) -> str:
        """
        Hello World endpoint

        Args:
            client_ip (str): Client IP address
            resource_path (str): Resource path

        Returns:
            str: Hello World response
        """
        return f"Hello World from {client_ip}"

    client_ip = "192.168.1.100"
    resource_path = "/hello"

    try:
        response = hello_world(client_ip, resource_path)
        logging.info(response)
    except RateLimitException as e:
        logging.error(e)

if __name__ == "__main__":
    main()
```

In this example, we have three files:

1.  `rate_limiter.py`: Contains the `RateLimiter` class, which is responsible for checking if a client is rate limited. It uses Redis to store and retrieve rate limit information.
2.  `config.py`: Contains the `Config` class, which is responsible for storing and retrieving configuration settings.
3.  `main.py`: Contains the main function, which demonstrates how to use the `RateLimitMiddleware` class to rate limit API requests.

To run the example, you need to install the `redis` library by running `pip install redis`. You also need to configure the `config.py` file to point to your Redis instance.

if __name__ == "__main__":
    print(f"🚀 Design API rate limiting middleware")
    print(f"📊 Generated: 2025-09-04T20:24:48.019091")
    logger.info(f"Starting Design API rate limiting middleware...")
