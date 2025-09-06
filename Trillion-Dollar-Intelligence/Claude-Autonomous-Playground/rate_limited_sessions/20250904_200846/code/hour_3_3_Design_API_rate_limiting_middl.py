#!/usr/bin/env python3
"""
Design API rate limiting middleware
Production-ready Python implementation
Generated Hour 3 - Project 3
Created: 2025-09-04T20:19:29.821111
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

This implementation provides a production-ready API rate limiting middleware using Python. It utilizes the `logging` module for logging, `configparser` for configuration management, and `redis` for storing the rate limit counts.

**Installation**
---------------

To install the required libraries, run the following command:

```bash
pip install redis
```

**Implementation**
-----------------

### `config.py`

```python
import configparser

class Config:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get(self, section, key):
        return self.config.get(section, key)

    def get_int(self, section, key):
        return int(self.config.get(section, key))

    def get_bool(self, section, key):
        return self.config.getboolean(section, key)
```

### `rate_limiter.py`

```python
import logging
import redis
from rate_limiter.config import Config
from typing import Dict, Optional

class RateLimiter:
    def __init__(self, config_file: str, redis_host: str, redis_port: int):
        self.config = Config(config_file)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.logger = logging.getLogger(__name__)

    def _get_rate_limit(self, ip_address: str, limit_key: str) -> Optional[Dict]:
        try:
            rate_limit = self.redis_client.hgetall(limit_key)
            if rate_limit:
                return dict(rate_limit)
            else:
                return None
        except redis.exceptions.RedisError as e:
            self.logger.error(f"Error retrieving rate limit: {e}")
            return None

    def _set_rate_limit(self, ip_address: str, limit_key: str, rate_limit: Dict):
        try:
            self.redis_client.hmset(limit_key, rate_limit)
            self.redis_client.expire(limit_key, int(self.config.get('rate_limit', 'expires_in')))
        except redis.exceptions.RedisError as e:
            self.logger.error(f"Error setting rate limit: {e}")

    def _increment_rate_limit(self, ip_address: str, limit_key: str):
        rate_limit = self._get_rate_limit(ip_address, limit_key)
        if rate_limit:
            rate_limit['count'] = rate_limit['count'] + 1
            self._set_rate_limit(ip_address, limit_key, rate_limit)

    def is_rate_limited(self, ip_address: str, limit_key: str) -> bool:
        rate_limit = self._get_rate_limit(ip_address, limit_key)
        if rate_limit:
            return rate_limit['count'] >= int(self.config.get('rate_limit', 'limit'))
        else:
            return False

    def rate_limit(self, ip_address: str, limit_key: str):
        if self.is_rate_limited(ip_address, limit_key):
            raise Exception("Rate limit exceeded")
        else:
            self._increment_rate_limit(ip_address, limit_key)
```

### `middleware.py`

```python
import logging
from rate_limiter.rate_limiter import RateLimiter
from typing import Callable, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class RateLimitMiddleware:
    def __init__(self, app: FastAPI, config_file: str, redis_host: str, redis_port: int):
        self.app = app
        self.rate_limiter = RateLimiter(config_file, redis_host, redis_port)
        self.logger = logging.getLogger(__name__)

    def __call__(self, receive, send):
        async def middleware(request: Request):
            try:
                if request.method in ['GET', 'HEAD']:
                    limit_key = f"{request.client.host}:{request.path}"
                else:
                    limit_key = f"{request.client.host}:{request.method}:{request.path}"
                self.rate_limiter.rate_limit(request.client.host, limit_key)
                return await send(request)
            except Exception as e:
                return JSONResponse(status_code=429, content={"error": str(e)})

        return middleware

def create_rate_limit_middleware(app: FastAPI, config_file: str, redis_host: str, redis_port: int) -> FastAPI:
    middleware = RateLimitMiddleware(app, config_file, redis_host, redis_port)
    return middleware(app)
```

### `main.py`

```python
import logging
from fastapi import FastAPI
from rate_limiter.middleware import create_rate_limit_middleware
from rate_limiter.config import Config

logging.basicConfig(level=logging.INFO)

app = FastAPI()

config = Config('config.ini')
redis_host = config.get('redis', 'host')
redis_port = config.get_int('redis', 'port')

app = create_rate_limit_middleware(app, 'config.ini', redis_host, redis_port)

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Configuration**
----------------

Create a `config.ini` file with the following configuration:

```ini
[rate_limit]
limit = 100
expires_in = 60

[redis]
host = localhost
port = 6379
```

**Usage**
---------

To use the rate limiting middleware, create an instance of the `RateLimitMiddleware` class and pass it to the `FastAPI` instance. You can configure the rate limiting settings by modifying the `config.ini` file.

**Error Handling**
-----------------

The rate limiting middleware raises an exception when the rate limit is exceeded. You can catch this exception and return a custom response to the client.

**Type Hints**
--------------

The code uses type hints to indicate the expected types of function arguments and return values. This makes the code more readable and self-documenting.

**Logging**
---------

The code uses the `logging` module to log events and errors. You can configure the logging settings by modifying the `logging.basicConfig` call.

**Redis**
--------

The code uses Redis to store the rate limit counts. You can configure the Redis settings by modifying the `redis_host` and `redis_port` variables.

if __name__ == "__main__":
    print(f"🚀 Design API rate limiting middleware")
    print(f"📊 Generated: 2025-09-04T20:19:29.821125")
    logger.info(f"Starting Design API rate limiting middleware...")
