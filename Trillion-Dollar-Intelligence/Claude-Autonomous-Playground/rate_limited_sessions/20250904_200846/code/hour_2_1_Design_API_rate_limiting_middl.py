#!/usr/bin/env python3
"""
Design API rate limiting middleware
Production-ready Python implementation
Generated Hour 2 - Project 1
Created: 2025-09-04T20:14:37.301593
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
Here's an example of production-ready Python code for designing API rate limiting middleware using Flask. This implementation includes classes, error handling, documentation, type hints, logging, and configuration management.

**rate_limiting_middleware.py**

```python
import logging
import time
from typing import Dict, Callable, Any

from flask import Flask, request, jsonify
from flask.logging import default_handler

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler and set level to INFO
file_handler = logging.FileHandler('rate_limiting.log')
file_handler.setLevel(logging.INFO)

# Create formatter and attach it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to the logger
logger.addHandler(file_handler)

class RateLimitExceededError(Exception):
    """
    Custom exception for rate limit exceeded error.
    """
    pass

class RateLimitMiddleware:
    """
    API rate limiting middleware.

    This middleware checks the number of requests from the same IP address
    within a specified time window and raises a RateLimitExceededError if the
    rate limit is exceeded.
    """
    def __init__(self, app: Flask, config: Dict[str, Any]):
        """
        Initialize the rate limiting middleware.

        Args:
        - app: Flask application instance.
        - config: Middleware configuration dictionary.
            - 'rate_limit': Maximum number of requests within the time window.
            - 'time_window': Time window in seconds.
        """
        self.app = app
        self.config = config
        self.ip_rate_limits: Dict[str, float] = {}

    def __call__(self, func: Callable) -> Callable:
        """
        Call the decorated function with rate limiting checks.

        Args:
        - func: Decorated function.

        Returns:
        - Decorated function with rate limiting checks.
        """
        def decorated_function(*args, **kwargs) -> Any:
            # Get the client IP address from the request headers
            client_ip = request.remote_addr

            # Check if the client IP address exists in the rate limit dictionary
            if client_ip not in self.ip_rate_limits:
                # Initialize the rate limit dictionary for the client IP address
                self.ip_rate_limits[client_ip] = {'timestamp': time.time(), 'requests': 0}

            # Get the current timestamp and increment the request count
            current_timestamp = time.time()
            self.ip_rate_limits[client_ip]['requests'] += 1

            # Check if the rate limit is exceeded
            if self.ip_rate_limits[client_ip]['requests'] >= self.config['rate_limit']:
                # Calculate the time elapsed since the last request
                time_elapsed = current_timestamp - self.ip_rate_limits[client_ip]['timestamp']

                # Check if the time window has passed
                if time_elapsed < self.config['time_window']:
                    # Raise a RateLimitExceededError
                    raise RateLimitExceededError('Rate limit exceeded')

                # Reset the rate limit dictionary for the client IP address
                self.ip_rate_limits[client_ip] = {'timestamp': current_timestamp, 'requests': 1}

            # Update the rate limit dictionary for the client IP address
            self.ip_rate_limits[client_ip]['timestamp'] = current_timestamp

            # Call the decorated function
            return func(*args, **kwargs)

        return decorated_function

def configure_rate_limiting_middleware(app: Flask, config: Dict[str, Any]) -> RateLimitMiddleware:
    """
    Configure the rate limiting middleware.

    Args:
    - app: Flask application instance.
    - config: Middleware configuration dictionary.

    Returns:
    - RateLimitMiddleware instance.
    """
    rate_limit_middleware = RateLimitMiddleware(app, config)
    return rate_limit_middleware

def register_rate_limiting_middleware(app: Flask, rate_limit_middleware: RateLimitMiddleware) -> None:
    """
    Register the rate limiting middleware with the Flask application.

    Args:
    - app: Flask application instance.
    - rate_limit_middleware: RateLimitMiddleware instance.
    """
    app.wsgi_app = rate_limit_middleware(app.wsgi_app)

# Example usage
if __name__ == '__main__':
    app = Flask(__name__)

    # Configure rate limiting middleware
    config = {
        'rate_limit': 5,  # Maximum number of requests within the time window
        'time_window': 60  # Time window in seconds
    }
    rate_limit_middleware = configure_rate_limiting_middleware(app, config)

    # Register rate limiting middleware with the Flask application
    register_rate_limiting_middleware(app, rate_limit_middleware)

    @app.route('/example', methods=['GET'])
    @rate_limit_middleware
    def example():
        return jsonify({'message': 'Example endpoint'}), 200

    @app.route('/exceed_rate_limit', methods=['GET'])
    def exceed_rate_limit():
        # Intentionally exceed the rate limit
        return jsonify({'message': 'Rate limit exceeded'}), 429

    if __name__ == '__main__':
        app.run(debug=True)
```

This implementation provides a `RateLimitMiddleware` class that checks the number of requests from the same IP address within a specified time window and raises a `RateLimitExceededError` if the rate limit is exceeded. The middleware is configured using a dictionary that specifies the maximum number of requests within the time window and the time window in seconds. The middleware is registered with the Flask application using the `register_rate_limiting_middleware` function.

The `exception` module is used to define a custom `RateLimitExceededError` exception that is raised when the rate limit is exceeded.

The `logging` module is used to configure the logger and log rate limiting events.

The example usage demonstrates how to configure and register the rate limiting middleware with the Flask application.

To run the example, simply execute the script using Python:
```bash
python rate_limiting_middleware.py
```
This will start the Flask development server, and you can access the example endpoints using a tool like `curl` or a web browser.

Note that this implementation assumes a simple rate limiting strategy based on IP addresses. In a production environment, you may want to consider using a more sophisticated rate limiting strategy, such as token bucket or leaky bucket algorithms. Additionally, you may want to consider using a caching layer, such as Redis or Memcached, to store rate limiting information.

if __name__ == "__main__":
    print(f"🚀 Design API rate limiting middleware")
    print(f"📊 Generated: 2025-09-04T20:14:37.301633")
    logger.info(f"Starting Design API rate limiting middleware...")
